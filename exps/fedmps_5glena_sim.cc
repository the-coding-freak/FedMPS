#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/spectrum-module.h"
#include "ns3/nr-module.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <map>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("FedMps5GLenaTraceGenerator");

// Structure to hold telemetry logs per client per round
struct TelemetryRecord {
    uint32_t round;
    uint32_t client_id;
    uint32_t mec_id;
    double uplink_mbps;
    double latency_ms;
    double packet_loss;
};

std::vector<TelemetryRecord> g_telemetry_records;

// Periodic callback to collect telemetry for the current round
void CollectTelemetry (uint32_t round, 
                       double round_duration, 
                       Ptr<FlowMonitor> monitor, 
                       Ptr<Ipv4FlowClassifier> classifier,
                       NetDeviceContainer ueDevices)
{
    NS_LOG_UNCOND ("Collecting 5G-LENA Telemetry for Round " << round);

    // Force update of flow monitor statistics
    monitor->CheckForLostPackets ();
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats ();

    // Map IP flow back to Client ID
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin (); i != stats.end (); ++i)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (i->first);
        
        // Uplink traffic to Remote Host (1.0.0.2)
        if (t.destinationAddress == Ipv4Address ("1.0.0.2"))
        {
            uint8_t ip_octet = t.sourceAddress.Get () & 0xFF;
            uint32_t client_id = ip_octet - 2; // Assuming UE IPs start at 10.0.0.2 / 7.0.0.2

            double txPackets = i->second.txPackets;
            double rxPackets = i->second.rxPackets;
            double lostPackets = i->second.lostPackets;
            
            double packet_loss = 0.0;
            if (txPackets > 0)
            {
                packet_loss = lostPackets / txPackets;
            }

            double latency_ms = 0.0;
            if (rxPackets > 0)
            {
                latency_ms = (i->second.delaySum.GetSeconds () / rxPackets) * 1000.0;
            }

            double throughput_mbps = 0.0;
            if (round_duration > 0)
            {
                throughput_mbps = (i->second.rxBytes * 8.0) / (round_duration * 1e6);
            }

            // Query UE's serving 5G NR cell (MEC ID)
            uint32_t mec_id = 0;
            if (client_id < ueDevices.GetN())
            {
                Ptr<NetDevice> ueDev = ueDevices.Get(client_id);
                Ptr<NrUeNetDevice> nrUeDev = ueDev->GetObject<NrUeNetDevice> ();
                if (nrUeDev)
                {
                    mec_id = nrUeDev->GetPhy (0)->GetCellId ();
                }
            }

            TelemetryRecord record;
            record.round = round;
            record.client_id = client_id;
            record.mec_id = mec_id;
            record.uplink_mbps = throughput_mbps;
            record.latency_ms = latency_ms;
            record.packet_loss = packet_loss;

            g_telemetry_records.push_back (record);
        }
    }

    monitor->SerializeToXmlFile ("temp_5glena_stats.xml", true, true);
    NS_LOG_UNCOND ("Round " << round << ": Collected " << g_telemetry_records.size() << " total 5G-LENA telemetry rows so far.");
}

int main (int argc, char *argv[])
{
    uint32_t num_users = 20;
    uint32_t rounds = 500;
    double round_duration = 1.0; 
    std::string output_file = "fedmps_5glena_trace.csv";

    CommandLine cmd (__FILE__);
    cmd.AddValue ("num_users", "Number of client UEs", num_users);
    cmd.AddValue ("rounds", "Number of FL rounds", rounds);
    cmd.AddValue ("round_duration", "Duration of each FL round in seconds", round_duration);
    cmd.AddValue ("output", "Path to save output trace CSV", output_file);
    cmd.Parse (argc, argv);

    // Create EPC Helper for 5G NR Core
    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper> ();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper> ();
    nrHelper->SetEpcHelper (epcHelper);

    // 5G NR Spectrum & Beamforming setup (3.5 GHz Sub-6 band, 100 MHz Channel)
    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;
    CcBwpCreator::SimpleOperationBandConf bandConf (3.5e9, 100e6, numCcPerBand);
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc (bandConf);

    // Configure NrChannelHelper with 3GPP channel factory
    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper> ();
    channelHelper->ConfigureFactories ("UMi", "Default", "ThreeGpp");
    channelHelper->SetChannelConditionModelAttribute ("UpdatePeriod", TimeValue (MilliSeconds (0)));
    channelHelper->SetPathlossAttribute ("ShadowingEnabled", BooleanValue (false));
    channelHelper->AssignChannelsToBands ({band});

    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper> ();
    nrHelper->SetBeamformingHelper (idealBeamformingHelper);

    allBwps = CcBwpCreator::GetAllBwps ({band});

    Ptr<Node> pgw = epcHelper->GetPgwNode ();

    // Create Remote Host (Cloud Orchestrator)
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create (1);
    Ptr<Node> remoteHost = remoteHostContainer.Get (0);
    InternetStackHelper internet;
    internet.Install (remoteHostContainer);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("100Gbps")));
    p2ph.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
    NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);

    // Create 3 MEC/gNodeB base stations
    NodeContainer gNbNodes;
    gNbNodes.Create (3);

    // Create Client UEs
    NodeContainer ueNodes;
    ueNodes.Create (num_users);

    // Base Station Positions (Fixed locations)
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
    positionAlloc->Add (Vector (50.0, 50.0, 10.0));   // MEC 1
    positionAlloc->Add (Vector (100.0, 100.0, 10.0)); // MEC 2
    positionAlloc->Add (Vector (150.0, 150.0, 10.0)); // MEC 3
    mobility.SetPositionAllocator (positionAlloc);
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    mobility.Install (gNbNodes);

    // Client UE Mobility (Random Walk inside 200m x 200m area at 5 to 20 m/s)
    mobility.SetPositionAllocator ("ns3::RandomBoxPositionAllocator",
                                   "X", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=200.0]"),
                                   "Y", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=200.0]"));
    mobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel",
                               "Bounds", StringValue ("0|200|0|200"),
                               "Speed", StringValue ("ns3::UniformRandomVariable[Min=5.0|Max=20.0]"),
                               "Distance", DoubleValue (20.0));
    mobility.Install (ueNodes);

    // Install 5G NR NetDevices
    NetDeviceContainer gNbDevices = nrHelper->InstallGnbDevice (gNbNodes, allBwps);
    NetDeviceContainer ueDevices = nrHelper->InstallUeDevice (ueNodes, allBwps);

    // Set Antennas (gNB: 8x8 MIMO array, UE: 2x2 array)
    nrHelper->SetGnbAntennaAttribute ("NumRows", UintegerValue (8));
    nrHelper->SetGnbAntennaAttribute ("NumColumns", UintegerValue (8));
    nrHelper->SetUeAntennaAttribute ("NumRows", UintegerValue (2));
    nrHelper->SetUeAntennaAttribute ("NumColumns", UintegerValue (2));

    internet.Install (ueNodes);
    Ipv4InterfaceContainer ueIpIfaces = epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueDevices));

    // Attach UEs to gNBs
    for (uint32_t u = 0; u < ueDevices.GetN (); ++u)
    {
        uint32_t gnb_idx = u % gNbDevices.GetN ();
        nrHelper->AttachToGnb (ueDevices.Get (u), gNbDevices.Get (gnb_idx));
    }

    // Routing setup
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
    {
        Ptr<Node> ueNode = ueNodes.Get (u);
        Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ueNode->GetObject<Ipv4> ());
        ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);
    }

    // UDP Traffic Applications (5G Uplink)
    uint16_t dlPort = 1234;
    PacketSinkHelper dlPacketSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), dlPort));
    ApplicationContainer serverApps = dlPacketSinkHelper.Install (remoteHost);
    serverApps.Start (Seconds (0.0));

    for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
    {
        OnOffHelper clientHelper ("ns3::UdpSocketFactory", Address (InetSocketAddress (internetIpIfaces.GetAddress (1), dlPort)));
        clientHelper.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1.0]"));
        clientHelper.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0.0]"));
        clientHelper.SetAttribute ("DataRate", DataRateValue (DataRate ("2Mbps")));
        clientHelper.SetAttribute ("PacketSize", UintegerValue (1024));

        ApplicationContainer clientApps = clientHelper.Install (ueNodes.Get (u));
        clientApps.Start (Seconds (0.5));
        clientApps.Stop (Seconds (rounds * round_duration));
    }

    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor> monitor = flowmonHelper.InstallAll ();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmonHelper.GetClassifier ());

    for (uint32_t r = 0; r < rounds; ++r)
    {
        double scheduled_time = (r + 1) * round_duration - 0.1;
        Simulator::Schedule (Seconds (scheduled_time), 
                             &CollectTelemetry, 
                             r, 
                             round_duration, 
                             monitor, 
                             classifier, 
                             ueDevices);
    }

    NS_LOG_UNCOND ("Starting 5G-LENA Simulation for " << rounds << " rounds...");
    Simulator::Stop (Seconds (rounds * round_duration));
    Simulator::Run ();

    std::ofstream csv;
    csv.open (output_file.c_str (), std::ios::out);
    csv << "round,client_id,mec_id,uplink_mbps,latency_ms,packet_loss\n";
    for (size_t i = 0; i < g_telemetry_records.size (); ++i)
    {
        csv << g_telemetry_records[i].round << ","
            << g_telemetry_records[i].client_id << ","
            << g_telemetry_records[i].mec_id << ","
            << g_telemetry_records[i].uplink_mbps << ","
            << g_telemetry_records[i].latency_ms << ","
            << g_telemetry_records[i].packet_loss << "\n";
    }
    csv.close ();
    NS_LOG_UNCOND ("Successfully wrote " << g_telemetry_records.size() << " 5G-LENA network telemetry rows to " << output_file);

    Simulator::Destroy ();
    return 0;
}

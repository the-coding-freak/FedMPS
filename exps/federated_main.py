#!/usr/bin/env python
# -*- coding: utf-8 -*-hello
# Python version: 3.6

import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
from pathlib import Path
from torch.utils.data import TensorDataset
import datetime
import logging


lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from lib.options import *
from lib.update import *
from lib.models.models import *
from lib.utils import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# Record console output
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'w')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass

def Fedavg(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, summary_writer,logger,logdir):

    idxs_users = np.arange(args.num_users)
    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses= [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w= local_model.update_weights_fedavg( idx=idx,model=copy.deepcopy(local_model_list[idx]))
            local_weights.append(copy.deepcopy(w))

        # aggregate local weights
        w_avg = copy.deepcopy(local_weights[0])
        for k in w_avg.keys():
            for i in range(1, len(local_weights)):
                w_avg[k] += local_weights[i][k]
            w_avg[k] = torch.div(w_avg[k], len(local_weights))

        # Update each local model with the globally averaged parameters
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model

        # test
        acc_list_l, loss_list_l= test_inference_fedavg(args,round, local_model_list, test_dataset, user_groups_lt,logger,summary_writer)
        print('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        logger.info('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round
            net = copy.deepcopy(local_model_list[0])
            torch.save(net.state_dict(), logdir + '/localmodel0.pth')

    print('best results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round,best_acc,best_std))
    logger.info('best results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round,best_acc,best_std))


def Fedprox(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,logdir):

    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses= [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        acc_list_train=[]
        loss_list_train=[]
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, idx_acc = local_model.update_weights_prox(args,idx, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            acc_list_train.append(idx_acc)
            loss_list_train.append(loss)
            local_weights.append(copy.deepcopy(w))

        # update global weights
        local_weights_list = local_weights
        w_avg = copy.deepcopy(local_weights_list[0])
        for k in w_avg.keys():
            for i in range(1, len(local_weights_list)):
                w_avg[k] += local_weights_list[i][k]
            w_avg[k] = torch.div(w_avg[k], len(local_weights_list))

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model

        # test
        acc_list_l, loss_list_l = test_inference_fedavg(args, round, local_model_list, test_dataset,user_groups_lt, logger, summary_writer)
        print('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        logger.info('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        summary_writer.add_scalars('scalar/Total_Avg_Accuracy', {'train':np.mean(acc_list_train),'test':np.mean(acc_list_l)}, round)

        logger.info('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        summary_writer.add_scalars('scalar/Total_Avg_Loss',{'train': np.mean(loss_list_train), 'test': np.mean(loss_list_l)}, round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round
            net = copy.deepcopy(local_model_list[0])
            torch.save(net.state_dict(), logdir + '/localmodel0.pth')

        print('best results:')
        print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))
        logger.info('best results:')
        logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))

def Moon(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,global_model,logger,summary_writer,logdir):

    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0
    old_nets_pool=[]#1

    if len(old_nets_pool) < 1:
        old_nets = copy.deepcopy(local_model_list)
        for net in old_nets:
            net.eval()
            for param in net.parameters():
                param.requires_grad = False

    party_list_this_round = [i for i in range(args.num_users)]
    for round in tqdm(range(args.rounds)):

        global_model.eval()
        for param in global_model.parameters():
            param.requires_grad = False

        local_weights, local_losses= [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        acc_list_train=[]
        loss_list_train=[]

        for idx in idxs_users:
            prev_models = []
            for i in range(len(old_nets_pool)):
                prev_models.append(old_nets_pool[i][idx])
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, idx_acc = local_model.update_weights_moon(args,idx, model=copy.deepcopy(local_model_list[idx]),global_model=global_model,previous_models=prev_models, global_round=round)
            acc_list_train.append(idx_acc)
            loss_list_train.append(loss)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        total_data_points = sum([len(user_groups[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(user_groups[r]) / total_data_points for r in party_list_this_round]

        local_weights_list = local_weights
        w_avg = copy.deepcopy(local_weights_list[0])
        for key, value in w_avg.items():
            w_avg[key] = value * fed_avg_freqs[0]
        for k in w_avg.keys():
            for i in range(1, len(local_weights_list)):
                w_avg[k] += local_weights_list[i][k]*fed_avg_freqs[i]

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model

        global_model.load_state_dict(w_avg)

        if len(old_nets_pool) < args.model_buffer_size:
            old_nets = copy.deepcopy(local_model_list)
            for  net in old_nets:
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            old_nets_pool.append(old_nets)
        elif args.pool_option == 'FIFO':
            old_nets = copy.deepcopy(local_model_list)
            for net in old_nets:
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            for i in range(args.model_buffer_size - 2, -1, -1):
                old_nets_pool[i] = old_nets_pool[i + 1]
            old_nets_pool[args.model_buffer_size - 1] = old_nets

        acc_list_l, loss_list_l,acc_list_g, loss_list,loss_total_list = test_inference_new_het_lt(args, local_model_list, test_dataset,classes_list, user_groups_lt)

        print('| ROUND: {} | For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_l), np.std(acc_list_l)))
        logger.info('| ROUND: {} | For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_l), np.std(acc_list_l)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round

    print('best results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))
    logger.info('best results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))

def fedntd(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, summary_writer,logger,logdir):

    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses= [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w= local_model.update_weights_fedntd(args, idx=idx,model=copy.deepcopy(local_model_list[idx]))
            local_weights.append(copy.deepcopy(w))

        # update global weights
        w_avg = copy.deepcopy(local_weights[0])
        for k in w_avg.keys():
            for i in range(1, len(local_weights)):
                w_avg[k] += local_weights[i][k]
            w_avg[k] = torch.div(w_avg[k], len(local_weights))

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model

        # test
        acc_list_l, loss_list_l= test_inference_fedavg(args,round, local_model_list, test_dataset, user_groups_lt,logger,summary_writer)
        print('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        logger.info('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round

    print('best results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round,best_acc,best_std))
    logger.info('best results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round,best_acc,best_std))

def fedgkd(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, logdir):
    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0

    models_buffer = []
    ensemble_model = None

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        acc_list_train = []
        loss_list_train = []
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, idx_acc = local_model.update_weights_gkd(args, idx, model=copy.deepcopy(local_model_list[idx]), global_round=round, avg_teacher=ensemble_model)
            acc_list_train.append(idx_acc)
            loss_list_train.append(loss)
            local_weights.append(copy.deepcopy(w))

        # update global avg weights for this round
        local_weights_list = local_weights
        w_avg = copy.deepcopy(local_weights_list[0])
        for k in w_avg.keys():
            for i in range(1, len(local_weights_list)):
                w_avg[k] += local_weights_list[i][k]
            w_avg[k] = torch.div(w_avg[k], len(local_weights_list))

        # update global ensemble weights
        if len(models_buffer) >= args.buffer_length:
            models_buffer.pop(0)
        models_buffer.append(copy.deepcopy(w_avg))

        ensemble_w = copy.deepcopy(models_buffer[0])
        for k in ensemble_w.keys():
            for i in range(1, len(models_buffer)):
                ensemble_w[k] += models_buffer[i][k]
            ensemble_w[k] = torch.div(ensemble_w[k], len(models_buffer))

        if ensemble_model is None:
            ensemble_model = copy.deepcopy(local_model_list[0])
        ensemble_model.load_state_dict(ensemble_w, strict=True)

        # provide the client with the average model, not the ensemble model
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model

        # test
        acc_list_l, loss_list_l = test_inference_fedavg(args, round, local_model_list, test_dataset, user_groups_lt, logger, summary_writer)
        print('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean( acc_list_l), np.std( acc_list_l)))
        logger.info('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean( acc_list_l), np.std( acc_list_l)))
        summary_writer.add_scalars('scalar/Total_Avg_Accuracy',{'train': np.mean(acc_list_train), 'test': np.mean(acc_list_l)}, round)

        logger.info('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean( acc_list_l), np.std( acc_list_l)))
        summary_writer.add_scalars('scalar/Total_Avg_Loss',{'train': np.mean(loss_list_train), 'test': np.mean(loss_list_l)}, round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round

        print('best results:')
        print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))
        logger.info('best results:')
        logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))
def Fedproc(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):

    idxs_users = np.arange(args.num_users)
    party_list_this_round = [i for i in range(args.num_users)]
    global_protos = []

    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0
    for round in tqdm(range(args.rounds)):

        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        acc_list_train=[]
        loss_list_train=[]

        for idx in idxs_users:

            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos, idx_acc = local_model.update_weights_fedproc(args,idx, model=copy.deepcopy(local_model_list[idx]),global_protos=global_protos, global_round=round)
            acc_list_train.append(idx_acc)
            loss_list_train.append(loss['1'])
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_protos[idx] = agg_protos

        # update global protos
        global_protos = proto_aggregation(local_protos)
        # update global weights
        total_data_points = sum([len(user_groups[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(user_groups[r]) / total_data_points for r in party_list_this_round]

        local_weights_list = local_weights
        w_avg = copy.deepcopy(local_weights_list[0])
        for key, value in w_avg.items():
            w_avg[key] = value * fed_avg_freqs[0]
        for k in w_avg.keys():
            for i in range(1, len(local_weights_list)):
                w_avg[k] += local_weights_list[i][k]*fed_avg_freqs[i]

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model


        acc_list_l, loss_list_l,acc_list_g, loss_list,loss_total_list = test_inference_new_het_lt(args, local_model_list, test_dataset,classes_list, user_groups_lt)
        print('| ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
            round, np.mean(acc_list_l), np.std(acc_list_l)))
        logger.info(
            '| ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
                round, np.mean(acc_list_l), np.std(acc_list_l)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round

    print('best results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
        best_round, best_acc, best_std))
    logger.info('best results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
        best_round, best_acc, best_std))


def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, summary_writer,logger,logdir):

    global_protos = []
    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf')
    best_std = -float('inf')
    best_acc_w=-float('inf')
    best_std_w=-float('inf')
    best_round = 0
    best_round_w=0
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w,protos = local_model.update_weights_fedproto(args, idx, global_protos,model=copy.deepcopy(local_model_list[idx]),global_round=round)
            agg_protos = agg_func(protos)
            local_weights.append(copy.deepcopy(w))
            local_protos[idx] = agg_protos

        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global weights
        global_protos = proto_aggregation(local_protos)

        # test
        acc_list_l, acc_list_g= test_inference_fedproto(args,logger, local_model_list, test_dataset,classes_list, user_groups_lt, global_protos)

        print('| ROUND: {} | For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l), np.std(acc_list_l)))
        logger.info('| ROUND: {} | For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        print('| ROUND: {} | For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_g), np.std(acc_list_g)))
        logger.info('| ROUND: {} | For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_g), np.std(acc_list_g)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy_wp', np.mean(acc_list_g), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round
        if np.mean(acc_list_g) > best_acc_w:
            best_acc_w = np.mean(acc_list_g)
            best_std_w = np.std(acc_list_g)
            best_round_w = round

    print('best wo results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))
    logger.info('best wo results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))

    print('best w results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round_w , best_acc_w ,best_std_w ))
    logger.info('best w results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round_w ,best_acc_w ,best_std_w ))


def FedMPS(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,summary_writer,logger):

    # global model: shares the same structure as the output layer of each local model
    global_model = GlobalFedmps(args)
    global_model.to(args.device)
    global_model.train()

    global_high_protos = {}
    global_low_protos = {}
    global_logits = {}
    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf') # best results wo protos
    best_std = -float('inf')
    best_round = 0
    best_acc_w = -float('inf')  # best results w protos
    best_std_w = -float('inf')
    best_round_w = 0
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_high_protos, local_low_protos = [], [], {}, {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        acc_list_train = []
        loss_list_train = []
        for idx in idxs_users:
            # local model updating
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, high_protos, low_protos, idx_acc = local_model.update_weights_fedmps(args, idx, global_high_protos, global_low_protos, global_logits, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            acc_list_train.append(idx_acc)
            loss_list_train.append(loss['total'])
            agg_high_protos = agg_func(high_protos)
            agg_low_protos = agg_func(low_protos)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_high_protos[idx] = agg_high_protos
            local_low_protos[idx] = agg_low_protos

        # aggregate local multi-level prototypes instead of local weights
        local_weights_list = local_weights
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        global_high_protos = proto_aggregation(local_high_protos)
        global_low_protos = proto_aggregation(local_low_protos)

        # global model training:
        # create inputs: local high-level prototypes
        global_data, global_label = get_global_input(local_high_protos)
        dataset = TensorDataset(global_data, global_label)
        train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        # begin training and output global logits
        global_logits = train_global_proto_model(global_model, train_dataloader)

        # test
        acc_list_l, loss_list_l, acc_list_g, loss_list, loss_total_list = test_inference_new_het_lt(args,local_model_list,test_dataset,classes_list,user_groups_lt,global_high_protos)

        print('| ROUND: {} | For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_l), np.std(acc_list_l)))
        logger.info('| ROUND: {} | For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_l), np.std(acc_list_l)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        print('| ROUND: {} | For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_g), np.std(acc_list_g)))
        logger.info('| ROUND: {} | For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_g), np.std(acc_list_g)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy_wp', np.mean(acc_list_g), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round
        if np.mean(acc_list_g) > best_acc_w:
            best_acc_w = np.mean(acc_list_g)
            best_std_w = np.std(acc_list_g)
            best_round_w = round

    print('best wo results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))
    logger.info('best wo results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))

    print('best w results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round_w, best_acc_w, best_std_w))
    logger.info('best w results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round_w, best_acc_w, best_std_w))



if __name__ == '__main__':
    args = args_parser()
    exp_details(args)
    logdir = os.path.join('../newresults', args.alg, str(datetime.datetime.now().strftime("%Y-%m-%d/%H.%M.%S"))+'_'+args.dataset+'_n'+str(args.ways))
    mkdirs(logdir)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(logdir, 'log.log'),
        format='[%(levelname)s](%(asctime)s) %(message)s',
        datefmt='%Y/%m/%d/ %I:%M:%S %p', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    print("**Basic Setting...")
    logger.info("**Basic Setting...")
    print('  ', args)
    logging.info(args)

    summary_writer = SummaryWriter(logdir)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load dataset and user groups
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)# Minimum 2 classes; cannot exceed the total number of classes  List of the number of classes owned by each client: [,,,]
    if args.dataset == 'mnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset =='cifar100':
        k_list = np.random.randint(args.shots- args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset=='tinyimagenet':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'realwaste':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'flowers':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'defungi' or args.dataset == 'fashion':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'imagenet':  # The number of samples in the category with the fewest samples in the training set is 732, and the number of samples in the category with the fewest samples in the test set is 50.
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)

    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)
    # user_groups: dictionary where
    #   - key = client ID
    #   - value = ndarray of selected sample IDs for the client’s chosen classes (class IDs sorted in ascending order)
    # user_groups_lt: test set sample ID dictionary
    # classes_list: list of lists representing the classes assigned to each client

    # Build models
    local_model_list = []
    for i in range(args.num_users):
        if args.dataset == 'mnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20

            local_model = CNNMnist(args=args)

        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20
            local_model = CNNFemnist(args=args)

        elif args.dataset == 'cifar10' or args.dataset=='cifar100' or args.dataset == 'flowers'  or args.dataset == 'defungi' :
            local_model = CNNCifar(args=args)
        elif args.dataset=='tinyimagenet':
            args.num_classes = 200
            local_model = ModelCT(out_dim=256, n_classes=args.num_classes)
        elif args.dataset=='realwaste':
            local_model = CNNCifar(args=args)
        elif args.dataset=='fashion':
            local_model=CNNFashion_Mnist(args=args)
        elif args.dataset == 'imagenet':
            local_model = ResNetWithFeatures(base='resnet18', num_classes=args.num_classes)

        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)


    if args.alg=='fedavg':
        Fedavg(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, summary_writer,logger,logdir)
    elif args.alg=='fedprox':
        Fedprox(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,logdir)
    elif args.alg=='moon':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'femnist':
            global_model = CNNFemnist(args=args)
        elif args.dataset == 'cifar10' or args.dataset=='realwaste' or args.dataset == 'flowers' or args.dataset == 'defungi':
            global_model = CNNCifar(args=args)
        elif args.dataset=='cifar100':
            args.num_classes = 100
            local_model = ModelCT( out_dim=256, n_classes=args.num_classes)
        elif args.dataset=='tinyimagenet':
            args.num_classes = 200
            local_model = ModelCT(out_dim=256, n_classes=args.num_classes)
        elif args.dataset=='fashion':
            global_model=CNNFashion_Mnist(args=args)
        elif args.dataset == 'imagenet':
            global_model = ResNetWithFeatures(base='resnet18')
        global_model.to(args.device)
        global_model.train()
        Moon(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,global_model,logger,summary_writer,logdir)
    elif args.alg == 'fedntd':
        fedntd(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, summary_writer, logger, logdir)
    elif args.alg == 'fedgkd':
        fedgkd(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, logdir)
    elif args.alg=='fedproc':
        Fedproc(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
    elif args.alg=='fedproto':
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, summary_writer,logger,logdir)
    elif args.alg=='ours':#FedMPS
        FedMPS(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,summary_writer,logger)




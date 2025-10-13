#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch.utils.data import DataLoader, Dataset
from lib.conloss import *
from lib.utils import *
from lib.ntdloss import *

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(self.device)
        self.ntd_criterion = NTD_Loss(args.num_classes, args.ntd_tau, args.ntd_beta)
        self.gkd_criterion = nn.CrossEntropyLoss(reduction="mean")

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(1 * len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=True)

        return trainloader

    def update_weights_fedavg(self, idx,model):
        # Set mode to train model
        model.train()

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            correct=0
            total=0
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                _,log_probs, _,_ = model(images)
                loss = self.criterion(log_probs, labels)

                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total +=labels.size(0)

                batch_loss.append(loss.item())
            epoch_loss=sum(batch_loss)/len(batch_loss)
            train_acc = correct / total
            print(' User: %d Epoch: %d  Loss: %f ||  train_acc: %f ' % ( idx, iter, epoch_loss, train_acc))

        return model.state_dict()

    def update_weights_prox(self, args, idx, model, global_round):
        '''
        Based on https://github.com/litian96/FedProx
        '''
        # Set mode to train model
        model.train()
        epoch_loss = []
        global_weight_collector = copy.deepcopy(list(model.to(args.device).parameters()))

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            correct = 0
            total = 0
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                _, log_probs, _, _ = model(images)
                loss = self.criterion(log_probs, labels)

                fed_prox_reg = 0.0
                # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
                for param_index, param in enumerate(model.parameters()):
                    fed_prox_reg += ((args.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg

                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total += labels.size(0)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            acc_last_epoch = correct / total
            print(' User: %d Epoch: %d  Loss: %f ||  train_acc: %f ' % (
            idx, iter, sum(batch_loss) / len(batch_loss), acc_last_epoch))
        epoch_loss = sum(epoch_loss) / len(epoch_loss)

        return model.state_dict(), epoch_loss, acc_val.item(), acc_last_epoch



    def update_weights_moon(self,args, idx, model, global_model,previous_models,global_round):
        '''
        Based on https://github.com/Xtra-Computing/MOON
        '''
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        cos = torch.nn.CosineSimilarity(dim=-1)
        criterion = nn.CrossEntropyLoss().to(args.device)
        for iter in range(self.args.train_ep):
            correct=0
            total=0
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)
                model.zero_grad()
                _,log_probs, protos,_ = model(images)

                loss1 = self.criterion(log_probs, labels)

                _,_, pro2,_ = global_model(images)
                posi = cos(protos, pro2)
                logits = posi.reshape(-1, 1)

                for previous_model in previous_models:
                    previous_model.to(args.device)
                    _,_, pro3,_ = previous_model(images)
                    nega = cos(protos, pro3)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                logits /= args.temperature
                labels_m = torch.zeros(images.size(0)).to(args.device).long()

                loss2 = args.mu * criterion(logits, labels_m)

                loss = loss1+loss2
                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total +=labels.size(0)
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            acc_last_epoch = correct / total


        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item(),acc_last_epoch

    def update_weights_fedntd(self, args, idx, model):
        '''
        Based on https://github.com/Lee-Gihun/FedNTD
        '''
        # keep global
        dg_model = copy.deepcopy(model)
        dg_model.to(args.device)

        for params in dg_model.parameters():
            params.requires_grad = False

        # Set mode to train model
        model.train()

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            correct = 0
            total = 0
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                logits, _, _, _ = model(images)
                with torch.no_grad():
                    dg_logits, _, _, _ = dg_model(images)

                loss = self.ntd_criterion(logits, labels, dg_logits)

                loss.backward()
                optimizer.step()

                _, y_hat = logits.max(1)
                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total += labels.size(0)

                batch_loss.append(loss.item())
            epoch_loss = sum(batch_loss) / len(batch_loss)
            train_acc = correct / total

            print(' User: %d Epoch: %d  Loss: %f ||  train_acc: %f ' % (idx, iter, epoch_loss, train_acc))

        return model.state_dict()

    def update_weights_gkd(self, args, idx, model, global_round, avg_teacher):
        '''
        Based on https://github.com/CGCL-codes/FedGKD
        '''
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            correct = 0
            total = 0
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                student_logits, _, _, _ = model(images)

                loss1 = self.gkd_criterion(student_logits, labels)

                if avg_teacher is None:
                    loss = loss1
                else:
                    avg_teacher = self._turn_off_grad(avg_teacher.to(args.device))
                    with torch.no_grad():
                        teacher_logits, _, _, _ = avg_teacher(images)

                    loss2 = self._divergence(args,
                                             student_logits=student_logits / args.gkd_temperature,
                                             teacher_logits=teacher_logits / args.gkd_temperature,
                                             )
                    loss2 = args.distillation_coefficient * loss2

                    loss = loss1 + loss2

                loss.backward()
                optimizer.step()

                _, y_hat = student_logits.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total += labels.size(0)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            acc_last_epoch = correct / total
            print(' User: %d Epoch: %d  Loss: %f ||  train_acc: %f ' % (
            idx, iter, sum(batch_loss) / len(batch_loss), acc_last_epoch))
        epoch_loss = sum(epoch_loss) / len(epoch_loss)

        return model.state_dict(), epoch_loss, acc_val.item(), acc_last_epoch

    def _turn_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        return model

    def _divergence(self, args, student_logits, teacher_logits):
        divergence = args.temperature * args.temperature * F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )  # forward KL
        return divergence

    def update_weights_fedproc(self, args, idx, model, global_protos, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total': [], '1': [], '2': [], '3': []}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        if global_round < 100:
            alpha = global_round / 100
        else:
            alpha = 1

        for iter in range(self.args.train_ep):
            correct = 0
            total = 0
            batch_loss = {'total': [], '1': [], '2': [], '3': []}
            agg_protos_label = {}
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                _, log_probs, protos, _ = model(images)
                loss1 = self.criterion(log_probs, labels)

                loss_mysupcon = MySupConLoss(temperature=0.5)
                if len(global_protos) == 0:
                    loss2 = 0 * loss1
                else:
                    global_h_input, global_h_labels = self.hcall(global_protos)
                    global_h_input = global_h_input.to(self.device)
                    global_h_labels = global_h_labels.to(self.device)
                    local_h_input = protos  # (bs,50)
                    local_h_labels = labels  # (bs,)
                    loss2 = loss_mysupcon.forward(feature_i=local_h_input, feature_j=global_h_input,
                                                  label_i=local_h_labels,
                                                  label_j=global_h_labels)
                if global_round == 0:
                    loss = loss1
                else:
                    loss = alpha * loss1 + (1 - alpha) * loss2

                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if labels[i].item() in agg_protos_label:
                        agg_protos_label[labels[i].item()].append(protos[i, :])
                    else:
                        agg_protos_label[labels[i].item()] = [protos[i, :]]

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total += labels.size(0)
                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        '| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader),
                            loss.item(),
                            acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            acc_last_epoch = correct / total

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label, acc_last_epoch


    def update_weights_fedproto(self, args, idx, global_protos,model, global_round=round):
        '''
        Based on https://github.com/yuetan031/FedProto
        '''
        # Set mode to train model
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}
        epoch_acc=[]

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            correct = 0
            total = 0
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}
            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)

                # loss1: cross-entrophy loss, loss2: proto distance loss
                model.zero_grad()
                _,log_probs, protos,_= model(images)
                loss1 = self.criterion(log_probs, labels)

                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = 0*loss1
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)

                loss = loss1 + loss2 * args.ld
                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        agg_protos_label[label_g[i].item()].append(protos[i,:])
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i,:]]

                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)

                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total += labels.size(0)

                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            acc_last_epoch = correct / total
            epoch_acc.append(acc_last_epoch)

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        return model.state_dict(), agg_protos_label

    def update_weights_fedmps(self, args, idx, global_high_protos, global_low_protos, global_logits, model, global_round=round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total': [], '1': [], '2': [], '3': [],'4':[]}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            correct = 0
            total = 0
            batch_loss = {'total': [], '1': [], '2': [], '3': [],'4':[]}
            agg_high_protos_label = {}
            agg_low_protos_label = {}
            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)

                model.zero_grad()
                probs, log_probs, high_protos, low_protos = model(images)

                # loss1: classification loss
                loss1 = self.criterion(log_probs, labels)# nn.NLLLoss()

                # loss2: high-level contrastive learning loss between local features and global prototypes
                # loss3: low-level contrastive learning loss between local features and global prototypes
                loss_mysupcon = MySupConLoss(temperature=0.5)
                if len(global_high_protos) == 0:
                    loss2 = 0 * loss1
                    loss3 = 0 * loss1
                else:
                    global_h_input, global_h_labels = self.hcfit(global_high_protos, high_protos, labels)
                    global_l_input, global_l_labels = self.hcfit(global_low_protos, low_protos, labels)
                    local_h_input = high_protos
                    local_h_labels = labels
                    local_l_input = low_protos
                    local_l_labels = labels

                    loss3 = loss_mysupcon.forward(feature_i=local_l_input, feature_j=global_l_input,
                                                  label_i=local_l_labels, label_j=global_l_labels)
                    loss2 = loss_mysupcon.forward(feature_i=local_h_input, feature_j=global_h_input,
                                                  label_i=local_h_labels, label_j=global_h_labels)

                # loss4: distillation loss between local soft labels and global soft labels
                soft_loss = nn.KLDivLoss(reduction="batchmean")
                T = args.T
                if len(global_logits) == 0:
                    loss4=0*loss1
                else:
                    global_probs = []
                    for l in labels:
                        class_prob = global_logits[l.item()]
                        global_probs.append(class_prob)
                    global_probs = torch.stack(global_probs)
                    loss4 = soft_loss(F.log_softmax(probs / T, dim=1), F.softmax(global_probs / T, dim=1))

                loss = loss1 + args.alph * loss2 + args.beta * loss3 + args.gama*loss4# Note that the weights of the various losses here differ from those in the article


                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if labels[i].item() in agg_high_protos_label:
                        agg_high_protos_label[labels[i].item()].append(high_protos[i, :])
                        agg_low_protos_label[labels[i].item()].append(low_protos[i, :])
                    else:
                        agg_high_protos_label[labels[i].item()] = [high_protos[i, :]]
                        agg_low_protos_label[labels[i].item()] = [low_protos[i, :]]

                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total += labels.size(0)
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader),
                            loss.item(),
                            acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
                batch_loss['3'].append(loss3.item())
                batch_loss['4'].append(loss4.item())
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            epoch_loss['3'].append(sum(batch_loss['3']) / len(batch_loss['3']))
            epoch_loss['4'].append(sum(batch_loss['3']) / len(batch_loss['4']))
            acc_last_epoch = correct / total

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        epoch_loss['3'] = sum(epoch_loss['3']) / len(epoch_loss['3'])
        epoch_loss['4'] = sum(epoch_loss['4']) / len(epoch_loss['4'])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_high_protos_label, agg_low_protos_label, acc_last_epoch


    def hcall(self,global_protos):
        global_labels = []
        global_labels.extend(global_protos)
        global_labels = torch.tensor(global_labels)
        global_input = torch.ones((global_labels.shape[0], global_protos.get(next(iter(global_protos)))[0].shape[-1]))
        i = 0
        for label in global_labels:
            global_input[i, :] = global_protos[label.item()][0].data
            i += 1
        return global_input, global_labels

    def hcfit(self,global_protos, local_input, local_labels):# Align global_input with local feature labels
        global_input = copy.deepcopy(local_input.data)
        i = 0
        for label in local_labels:
            if label.item() in global_protos.keys():
                global_input[i, :] = global_protos[label.item()][0].data
            i += 1
        global_labels = local_labels
        return global_input, global_labels

    def hcfitnorepeat(self,global_protos, local_input, local_labels):
        labels = torch.unique(local_labels)
        global_input = torch.zeros((len(labels),local_input.size(-1)))
        i = 0
        for label in labels:
            if label.item() in global_protos.keys():
                global_input[i, :] = global_protos[label.item()][0].data
            i += 1
        global_labels = labels
        return global_input, global_labels


    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset,user_groups_gt,idx):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        _,outputs, protos,_ = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss= loss/len(testloader)
    return accuracy, loss

def test_inference_new(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        outputs = torch.zeros(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        cnt = np.zeros(10)
        for i in range(10):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:,i] += prob_list[idx][:,tmp]
                    cnt[i]+=1
        for i in range(10):
            if cnt[i]!=0:
                outputs[:, i] = outputs[:,i]/cnt[i]

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)


    acc = correct/total

    return loss, acc



def test_inference_new_het(args, local_model_list, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        protos_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            _, protos = model(images)
            protos_list.append(protos)

        ensem_proto = torch.zeros(size=(images.shape[0], protos.shape[1])).to(device)
        # protos ensemble
        for protos in protos_list:
            ensem_proto += protos
        ensem_proto /= len(protos_list)

        a_large_num = 100
        outputs = a_large_num * torch.ones(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(10):
                if j in global_protos.keys():
                    dist = loss_mse(ensem_proto[i,:],global_protos[j][0])
                    outputs[i,j] = dist

        # Prediction
        _, pred_labels = torch.min(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    acc = correct/total

    return acc

def test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[]):
    """ Returns the test accuracy and loss.
    """

    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_g = []#w
    acc_list_l = []#wo
    loss_list_l=[]#wo
    loss_list = []#w
    loss_return_list=[]#w
    for idx in range(args.num_users):
        correct_wo,total_wo,loss_wo=0.0,0.0,0.0
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        # test (local model)
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            probs,outputs, protos,_ = model(images)

            batch_loss = criterion(outputs, labels)
            loss_wo += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct_wo += torch.sum(torch.eq(pred_labels, labels)).item()
            total_wo += len(labels)

        acc_wo = correct_wo / total_wo
        loss_re=loss_wo/len(testloader)
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc_wo))
        acc_list_l.append(acc_wo)
        loss_list_l.append(loss_re)

        # test (use global proto)
        correct_w, total_w = 0.0, 0.0
        if global_protos!=[]:
            loss_return=[]
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                probs, outputs, protos ,_= model(images)

                # compute the dist between protos and global_protos
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[idx]:
                            d = loss_mse(protos[i, :], global_protos[j][0])
                            dist[i, j] = d

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct_w += torch.sum(torch.eq(pred_labels, labels)).item()
                total_w += len(labels)

                # compute loss
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                loss2 = loss_mse(proto_new, protos)
                # loss1 = loss_function(probs, labels)
                loss1 = criterion(outputs, labels)
                if args.device == 'cuda':
                    loss2 = loss2.cpu().detach().numpy()
                else:
                    loss2 = loss2.detach().numpy()
                loss_return.append((loss1+loss2*args.ld).item())
            acc_w = correct_w / total_w
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc_w))
            acc_list_g.append(acc_w)
            loss_list.append(loss2)
            loss_return_list.append(sum(loss_return)/len(loss_return))

    return acc_list_l,loss_list_l, acc_list_g, loss_list,loss_return_list

def test_inference_fedproto(args,logger, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_g = []#w
    acc_list_l = []#wo
    loss_list = []
    for idx in range(args.num_users):
        correct_wo,total_wo=0,0
        batch_eval_loss = {'total': [], '1': [], '2': [], '3': []}
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        # test (local model)
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            _,outputs, protos,_ = model(images)

            batch_loss1 = criterion(outputs, labels)

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct_wo += torch.sum(torch.eq(pred_labels, labels)).item()
            total_wo += len(labels)
            batch_eval_loss['1'].append(batch_loss1.item())

        acc_wo = correct_wo / total_wo
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc_wo))
        logger.info('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc_wo))
        acc_list_l.append(acc_wo)

        # test (use global proto)
        correct_w, total_w = 0, 0
        if global_protos!=[]:
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                _,outputs, protos,_ = model(images)

                # compute the dist between protos and global_protos
                a_large_num = 10000
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[idx]:
                            d = loss_mse(protos[i, :], global_protos[j][0])
                            dist[i, j] = d

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct_w += torch.sum(torch.eq(pred_labels, labels)).item()
                total_w += len(labels)

                # compute loss
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                batch_loss2 = loss_mse(proto_new, protos)
                batch_eval_loss['2'].append(batch_loss2.item())
                batch_loss=batch_loss1+batch_loss2
                batch_eval_loss['total'].append(batch_loss.item())

            acc_w = correct_w / total_w
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc_w))
            logger.info('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc_w))
            acc_list_g.append(acc_w)

            loss_list.append(sum(batch_eval_loss['total']) / len(batch_eval_loss['total']))

    return acc_list_l, acc_list_g


def test_inference_fedavg(args,round, local_model_list, test_dataset, user_groups_gt,logger,summary_writer):
    """ Returns the test accuracy and loss.
    """
    acc_list_l = []#wo
    loss_list_l=[]
    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(args.device)
        test_acc, test_loss = test_inference(args, model, test_dataset, user_groups_gt, idx)

        print(' User: %d  Loss: %f ||  test_acc: %f ' % (idx ,test_loss,test_acc))
        logger.info(' User: %d  Loss: %f ||  test_acc: %f ' % (idx ,test_loss,test_acc))
        summary_writer.add_scalar('scalar/net_id%d_Test_Accuracy' % (idx), test_acc, round)
        acc_list_l.append(test_acc)
        loss_list_l.append(test_loss)

    return acc_list_l,loss_list_l




def save_protos(args, local_model_list, test_dataset, user_groups_gt):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)

    agg_protos_label = {}
    for idx in range(args.num_users):
        agg_protos_label[idx] = {}
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            outputs, protos,_ = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label[idx]:
                    agg_protos_label[idx][labels[i].item()].append(protos[i, :])
                else:
                    agg_protos_label[idx][labels[i].item()] = [protos[i, :]]

    x = []
    y = []
    d = []
    for i in range(args.num_users):
        for label in agg_protos_label[i].keys():
            for proto in agg_protos_label[i][label]:
                if args.device == 'cuda':
                    tmp = proto.cpu().detach().numpy()
                else:
                    tmp = proto.detach().numpy()
                x.append(tmp)
                y.append(label)
                d.append(i)

    x = np.array(x)
    y = np.array(y)
    d = np.array(d)
    np.save('./' + args.alg + '_protos.npy', x)
    np.save('./' + args.alg + '_labels.npy', y)
    np.save('./' + args.alg + '_idx.npy', d)

    print("Save protos and labels successfully.")



from torch.autograd import Variable
def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


def train_global_proto_model(global_model,train_dataloder):
    epochs = 6
    loss_function = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(global_model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        global_model.train()
        num_correct = 0
        num_samples = 0
        for image, label in train_dataloder:
            image, label = image.to(device), label.to(device)
            optim.zero_grad()
            out = global_model.forward(image.detach())
            pre = out.max(1).indices
            num_correct += (pre == label).sum()
            num_samples += pre.size(0)

            loss = loss_function(out, label)
            loss.backward()
            optim.step()
        acc = num_correct.item() / num_samples
        print("global epoch:{},train acc={}".format(epoch, acc))

    global_model.eval()
    logits=[]
    la=[]
    with torch.no_grad():
        for im, l in train_dataloder:
            im, l = im.to(device), l.to(device)
            output = global_model(im)
            logits.extend(output)
            la.extend(l)
    logits=torch.stack(logits)
    la=torch.tensor(la)
    class_logits={}
    for i in range(len(la)):
        lo=logits[i]
        lo_label=la[i]
        if lo_label.item() in class_logits:
            class_logits[lo_label.item()].append(lo)
        else:
            class_logits[lo_label.item()] = [lo]
    # global logits
    class_logits=agg_func(class_logits)
    return class_logits


import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
from net_core import MnistCNN
import torch.nn.functional as F
from torch import optim
import copy


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


learning_rate = 0.01
clip = 32      # 裁剪系数
E = 1
q = 0.03
eps = 16
delta = 1e-5
tot_T = 150
batch_size = 128
# DP-SGD with sampling rate = 3% and noise_multiplier = 1.0180295400464534 iterated over 5000 steps satisfies differential privacy with eps = 16 and delta = 1e-05.
sigma = compute_noise(1, q, eps, E*tot_T, delta, 1e-5)      # 高斯分布系数

# 累计训练数据
used_data_map = np.array([[0 for i in range(6000)] for j in range(10)])
ones_map = np.array([1 for i in range(6000)])


class FedClient(nn.Module):
    def __init__(self):
        super(FedClient, self).__init__()
        self.model = MnistCNN().to(dev)
        self.learning_rate = learning_rate
        self.clip = clip      # 裁剪系数
        self.E = E
        self.q = q
        self.eps = eps
        self.delta = delta
        self.tot_T = tot_T
        self.batch_size = batch_size
        # DP-SGD with sampling rate = 3% and noise_multiplier = 1.0180295400464534 iterated over 5000 steps satisfies differential privacy with eps = 16 and delta = 1e-05.
        self.sigma = sigma      # 高斯分布系数
        # self.sigma = 0.9054

    def train(self, client_dataset, epoches, client_id):
        loss_func = F.cross_entropy
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.0)

        self.model.train()
        train_loss = 0
        num = 0
        for epoch in range(epoches):
            idx = np.where(np.random.rand(len(client_dataset[:][0])) < self.q)[0]
            n_data = len(idx)
            # # rand_int = np.random.randint(n_data)
            # # while (rand_int == 0):
            # #     rand_int = np.random.randint(n_data)
            n_data = (int)(n_data / (10 - client_id))
            idx = idx[:n_data]
            print(f"n_data: {n_data}")
            
            # print(f"idx: {idx}")
            used_data_map[client_id][idx] = 1
            # print(f"used_data_map[{client_id}]: {used_data_map[client_id]}")

            sampled_dataset = TensorDataset(client_dataset[idx][0], client_dataset[idx][1])
            train_dl = DataLoader(
                dataset=sampled_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
            optimizer.zero_grad()
            for data, label in train_dl:
                data, label = data.to(dev), label.to(dev)
                preds = self.model(data.float())

                # loss = loss_func(preds, label.long())
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                # for name, param in self.model.named_parameters():
                #     clipped_grads[name] += param.grad / len(idx)
                # self.model.zero_grad()

                loss = criterion(preds, label.long())
                for i in range(loss.size()[0]):
                    loss[i].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                    for name, param in self.model.named_parameters():
                        clipped_grads[name] += param.grad / len(idx)
                    self.model.zero_grad()
            # add gaussian noise
            for name, param in self.model.named_parameters():
                clipped_grads[name] += torch.normal(0, self.sigma*self.clip, clipped_grads[name].shape).to(dev) / len(idx)
            for name, param in self.model.named_parameters():
                param.grad = clipped_grads[name]

            optimizer.step()
        #     if epoch == epoches - 1:
        #         num += 1
        #         train_loss += float(loss.item())
        # return train_loss / num
        # return loss.mean().item()
        # return len(sampled_dataset)
        # accu_num = 0
        accu_num = (used_data_map[client_id] == ones_map).sum()
        print(f"accu_num: {accu_num}")
        return accu_num

    def evaluate(self, test_data, test_labels):
        self.model.eval()
        correct = 0
        tot_sample = 0
        t_pred_y = self.model(test_data)
        _, predicted = torch.max(t_pred_y, 1)
        correct += (predicted == test_labels).sum().item()
        tot_sample += test_labels.size(0)
        acc = correct / tot_sample
        return acc

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path), strict=True)

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)


class FedServer(nn.Module):
    def __init__(self):
        super(FedServer, self).__init__()
        self.model = MnistCNN().to(dev)
            
    def fed_avg(self, model_path_list, accu_data_list, cur_client_list, merge_round_list):
        # FL average
        n_clients = len(model_path_list)
        all_merge_rounds = 0
        for client_id in cur_client_list:
            all_merge_rounds += merge_round_list[client_id]
        print(f"all_merge_rounds: {all_merge_rounds}")
        print(f"merge_round_list: {merge_round_list}")
        print(f"cur_client_list: {cur_client_list}")
        model_par = [torch.load(model_path) for model_path in model_path_list]
        new_par = copy.deepcopy(model_par[0])
        for name in new_par:
            new_par[name] = torch.zeros(new_par[name].shape).to(dev)
        all_data = sum(accu_data_list)
        print(f"accu_data_list: {accu_data_list}")
        # w_set = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.4]
        # w_set = [0.0375, 0.0375, 0.0375, 0.0375, 0.0375, 0.0375, 0.0375, 0.0375, 0.1, 0.6]
        # w_set = [1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 2/14, 4/14]
        
        # w_set1 = [1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 2/14, 4/14]
        # w_set2 = [0 for i in range(n_clients)]
        # for i in range(n_clients):
        #     w_set2[i] = accu_data_list[i] / all_data
        
        # def transform(set1, set2):
        #     n = len(set1)
        #     ret = [0 for i in range(n)]
        #     for i in range(n):
        #         ret[i] = (set1[i] + set2[i]) / 2
        #     print(ret)
        #     add_w = 0
        #     for i in range(0, 9):
        #         tmp = ret[i] - ret[i] * 0.7
        #         ret[i] = ret[i] * 0.7
        #         add_w += tmp
        #     sum_weights = 0
        #     for i in range(9, 10):
        #         sum_weights += ret[i]
        #     for i in range(9, 10):
        #         ret[i] += add_w * (ret[i] / sum_weights)
        #     return ret
        
        # w_set = transform(w_set1, w_set2)
        # print(f"sum(w_set): {sum(w_set)}")
        
        for idx, par in enumerate(model_par):
            # w = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
            # w = accu_data_list[idx] / all_data
            w = 0.1
            # w = w_set[idx]
            # w = 1 / n_clients
            # w = merge_round_list[cur_client_list[idx]] / all_merge_rounds
            print(f"w({idx}): {w}")
            for name in new_par:
                # new_par[name] += par[name] * (self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users]))
                # new_par[name] += par[name] * (w / self.C)
                new_par[name] += par[name] * w
        self.model.load_state_dict(copy.deepcopy(new_par))

    def evaluate(self, test_data, test_labels):
        self.model.eval()
        correct = 0
        tot_sample = 0
        t_pred_y = self.model(test_data)
        _, predicted = torch.max(t_pred_y, 1)
        correct += (predicted == test_labels).sum().item()
        tot_sample += test_labels.size(0)
        acc = correct / tot_sample
        return acc

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)


if __name__ == '__main__':
    pass
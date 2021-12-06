from random import random

from src_torch_distributed.get_data import DataSet
from src_torch_distributed.net_core import Mnist_2NN
from src_torch_distributed.fed_core import FedClient, FedServer
import torch.nn.functional as F
from torch import optim
from src_torch_distributed.utils import *
import numpy as np

# global settings
clients_num = 10
client_id = 1
epoch = 10
batch_size = 64
learning_rate = 0.01
dataset = DataSet(clients_num, True)
x_test, y_test = dataset.get_test_dataset()


def train_one_model(client_id):
    check_and_build_dir("../models/train")
    sub_model_path = f"../models/train/{client_id}.pkl"
    x_train, y_train = dataset.get_train_batch(client_id, batch_size*10)

    model = FedClient(net=Mnist_2NN(), ID=client_id)
    model.setJob(jobAdress="x3tg83jx0m4jf8chyp5djas4jf9")
    model.set_model_settings(loss_func=F.cross_entropy, optimizer=optim.SGD(model.net.parameters(), lr=learning_rate))
    global_model_path = "../models/global/global.pkl"
    if os.path.exists(global_model_path):
        model.load_model(global_model_path, weight=True)
    loss = model.train(x_train, y_train, epoch, batch_size)
    acc = model.evaluate(x_test, y_test, batch_size)

    model.save_model(sub_model_path, weight=True)
    model.upload()
    print(f"Client-ID:{client_id}, loss:{loss}, acc:{acc}")
    print("training done.")
    return loss, acc


def test_one_model():
    sub_model_path = "../models/global/global.pkl"
    client_model = FedClient(net=Mnist_2NN(), ID=client_id)
    client_model.load_model(sub_model_path, weight=True)
    acc = client_model.evaluate(x_test, y_test, batch_size)
    print(f'client({client_id})_acc:{acc}')


def test_federated_model():
    sub_model_paths = []
    sub_model_acc = []
    for i in range(clients_num):
        path = f"../models/train/{i + 1}.pkl"
        # client_model = FedClient(net=Mnist_2NN(), ID=client_id)
        # client_model.load_model(path, weight=True)
        # acc = client_model.evaluate(x_test, y_test, batch_size)
        # sub_model_acc.append(acc)
        sub_model_paths.append(path)
    # print(f"mean of sub_model_acc: {np.mean(sub_model_acc)}")

    global_model = FedServer(net=Mnist_2NN())

    global_model.load_client_weights(sub_model_paths)
    global_model.fed_avg()
    acc = global_model.evaluate(x_test, y_test, batch_size)
    print(f'clients_num:{clients_num}, global_acc:{acc}')

    global_model_dir = "../models/global"
    check_and_build_dir(global_model_dir)
    global_model_path = f"{global_model_dir}/global.pkl"
    global_model.save_model(global_model_path, weight=True)
    return acc


def test_federated():
    # initialization
    federated_rounds = 10
    init_federated_model()

    # federated main
    clients_loss_list = [[] for x in range(clients_num)]
    clients_acc_list = [[] for x in range(clients_num)]
    global_acc_list = []
    clients_avg_loss_list = []
    clients_avg_acc_list = []
    for epoch in range(federated_rounds):
        print(f"Round {epoch + 1}:")
        clients_loss_sum = 0
        clients_acc_sum = 0
        for client_id in range(clients_num):
            client_loss, client_acc = train_one_model(client_id + 1)
            clients_loss_list[client_id].append(round(client_loss, 4))
            clients_acc_list[client_id].append(round(client_acc, 4))
            clients_loss_sum += client_loss
            clients_acc_sum += client_acc
        clients_avg_loss_list.append(clients_loss_sum / clients_num)
        clients_avg_acc_list.append(clients_acc_sum / clients_num)
        global_acc = test_federated_model()
        global_acc_list.append(round(global_acc, 4))
    save_results(clients_loss_list, clients_acc_list, clients_avg_loss_list, clients_avg_acc_list, global_acc_list)
    save_pics(clients_num)


def init_federated_model():
    global_model = FedServer(net=Mnist_2NN())
    global_model_dir = "../models/global"
    check_and_build_dir(global_model_dir)
    global_model_path = f"{global_model_dir}/global.pkl"
    global_model.save_model(global_model_path, weight=True)


if __name__ == '__main__':
    test_federated()
    # train_one_model()
    # test_one_model()
    # test_federated_model(10)

    # batch training
    # for i in range(clients_num):
    #     train_one_model(i + 1)

    # batch federated
    # for i in range(clients_num):
    #     test_federated_model(i + 1)
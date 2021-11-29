from src.fedlib import *
from src.dataset_funcs import load_all_dataset, load_mnist
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
import os
import random
import matplotlib.pyplot as plt
import pandas as pd

def create_model_mnist():
    model = Sequential()
    model.add(Dense(512, activation="relu", input_shape=(28*28,)))
    model.add(Dense(10, activation="softmax"))
    return model


def train_one_model(client_id, epoch, batch_size, train, x_test, y_test):
    # process data
    x_train, y_train = np.hsplit(train, [784,])

    sub_model_path = f"../models/train/{client_id}.npy"

    model = FedClient(model=create_model_mnist(), ID=client_id)
    model.setJob(jobAdress="x3tg83jx0m4jf8chyp5djas4jf9")
    global_model_path = "../models/global/global_model.npy"
    if os.path.exists(global_model_path):
        model.load_model(global_model_path, weight=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train,
              y=y_train,
              batch_size=batch_size,
              epochs=epoch,
              verbose=1)
    loss, acc = model.evaluate(x=x_test,
                               y=y_test,
                               batch_size=128)
    model.save_model(sub_model_path, weight=True)
    model.upload()
    print(f"Client-ID:{client_id} , loss:{loss} , acc:{acc}")
    print("training done.")
    return loss, acc


def test_one_model():
    # x_train, y_train, x_test, y_test = load_all_dataset(dataset_path, features, test_size=0.5)
    x_train, y_train, x_test, y_test = load_mnist()

    sub_model_path = "../models/train/2.npy"
    idx = 1
    client_model = FedClient(model=create_model_mnist(), ID=idx)
    client_model.load_model(sub_model_path, weight=True)
    client_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    loss, acc = client_model.evaluate(x=x_test,
                                      y=y_test, batch_size=128)
    print(f'client({idx})_loss:{loss}, client({idx})_acc:{acc}')


def test_federated_model(client_num, x_test, y_test):
    sub_model_paths = []
    for i in range(client_num):
        sub_model_paths.append(f"../models/train/{i}.npy")

    global_model = FedServer(model=create_model_mnist())
    global_model.load_client_weights(sub_model_paths)
    global_model.fl_average()

    global_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    loss, acc = global_model.evaluate(x=x_test,
                                      y=y_test, batch_size=128)
    global_model_path = "../models/global/global_model.npy"
    global_model.save_model(global_model_path, weight=True)
    print(f'global_loss:{loss}, global_acc:{acc}')
    return loss, acc


def test_federated():
    # load data
    x_train, y_train, x_test, y_test = load_mnist()

    # parameters
    federated_rounds = 10
    train_epoch = 2
    clients_num = 10
    batch_size = 64

    # slice the dataset according to clients_num
    x_train_set = []
    y_train_set = []
    x_train_len = len(x_train)
    y_train_len = len(y_train)
    each_x_train_len = (int)(x_train_len / clients_num)
    each_y_train_len = (int)(y_train_len / clients_num)
    for i in range(clients_num):
        x_train_set.append(x_train[i * each_x_train_len : (i+1) * each_x_train_len])
        y_train_set.append(y_train[i * each_y_train_len : (i+1) * each_y_train_len])

    # zip dataset
    train_set = []
    for i in range(clients_num):
        train_set.append(np.concatenate([x_train_set[i], y_train_set[i]], 1))

    # split dataset
    # data1, label1 = np.hsplit(train_set[0], [784,])

    # federated main
    clients_loss_list = [[] for x in range(clients_num)]
    clients_acc_list = [[] for x in range(clients_num)]
    global_loss_list = []
    global_acc_list = []
    for epoch in range(federated_rounds):
        for client_id in range(clients_num):
            train = random.sample(train_set[client_id].tolist(), batch_size * 10)
            client_loss, client_acc = train_one_model(client_id, train_epoch, batch_size, np.array(train), x_test, y_test)
            clients_loss_list[client_id].append(round(client_loss, 4))
            clients_acc_list[client_id].append(round(client_acc, 4))
        global_loss, global_acc = test_federated_model(clients_num, x_test, y_test)
        global_loss_list.append(round(global_loss, 4))
        global_acc_list.append(round(global_acc, 4))
    save_results(clients_loss_list, clients_acc_list, global_loss_list, global_acc_list)


def check_and_build_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_results(clients_loss_list, clients_acc_list, global_loss_list, global_acc_list):
    clients_num = len(clients_loss_list)
    # 1. save results of each client to a single file
    for client_id in range(clients_num):
        client_dir = f"../results/clients/client_{client_id}"
        check_and_build_dir(client_dir)
        save_to_file(f"{client_dir}/client_{client_id}_loss.txt", clients_loss_list[client_id])
        save_to_file(f"{client_dir}/client_{client_id}_acc.txt", clients_acc_list[client_id])
    # 2. save results of federated model to file
    global_dir = f"../results/global"
    check_and_build_dir(global_dir)
    save_to_file(f"{global_dir}/global_loss.txt", global_loss_list)
    save_to_file(f"{global_dir}/global_acc.txt", global_acc_list)


def save_to_file(file_path, content_list):
    with open(file_path, 'w+') as f:
        for line in content_list:
            f.write(str(line) + '\n')
    f.close()


def save_pics(clients_data_dir, global_data_dir, clients_num):
    # plt config
    plt_config = {
        "title" : "",
        "xlabel" : "federated rounds",
        "ylabel" : "",
    }

    for client_id in range(clients_num):
        # 1. process loss data
        client_loss_file_path = f"{clients_data_dir}/client_{client_id}/client_{client_id}_loss.txt"
        client_pic_dir = f"../pic/clients/client_{client_id}"
        check_and_build_dir(client_pic_dir)
        client_loss_pic_path = f"{client_pic_dir}/client_{client_id}_loss.png"
        plt_config["title"] = f"loss of client-{client_id}"
        plt_config["ylabel"] = "loss"
        save_to_pic(client_loss_file_path, client_loss_pic_path, plt_config)

        # 2. process acc data
        client_acc_file_path = f"{clients_data_dir}/client_{client_id}/client_{client_id}_acc.txt"
        client_acc_pic_path = f"{client_pic_dir}/client_{client_id}_acc.png"
        plt_config["title"] = f"accuracy of client-{client_id}"
        plt_config["ylabel"] = "acc"
        save_to_pic(client_acc_file_path, client_acc_pic_path, plt_config)

    # 3. process global loss data
    global_loss_file_path = f"{global_data_dir}/global_loss.txt"
    global_pic_dir = f"../pic/global"
    check_and_build_dir(global_pic_dir)
    global_loss_pic_path = f"{global_pic_dir}/global_loss.png"
    plt_config["title"] = f"loss of federated model"
    plt_config["ylabel"] = "loss"
    save_to_pic(global_loss_file_path, global_loss_pic_path, plt_config)

    # 4. process global acc data
    global_acc_file_path = f"{global_data_dir}/global_acc.txt"
    global_acc_pic_path = f"{global_pic_dir}/global_acc.png"
    plt_config["title"] = f"accuracy of federated model"
    plt_config["ylabel"] = "acc"
    save_to_pic(global_acc_file_path, global_acc_pic_path, plt_config)


def save_to_pic(data_dir, dest_dir, plt_config):
    # 1. read data
    loss_data = pd.read_csv(data_dir, header=None)

    # 2. plt configure
    plt.figure(figsize=(10, 6))
    plt.title(plt_config["title"])
    plt.xlabel(plt_config["xlabel"])
    plt.ylabel(plt_config["ylabel"])
    y_axis_data = loss_data[0].tolist()
    clients_num = len(y_axis_data)
    x_axis_data = [i for i in range(clients_num)]
    plt.plot(x_axis_data, y_axis_data)
    plt.savefig(dest_dir)
    plt.close()


if __name__ == '__main__':
    # test_federated()
    clients_data_dir = "../results/clients"
    global_data_dir = "../results/global"
    clients_num = 10
    save_pics(clients_data_dir, global_data_dir, clients_num)
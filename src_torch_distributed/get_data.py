from torchvision import datasets, transforms
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 30, 2000
    num_shards = int(num_users * 2)
    num_imgs = int(60000 / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def mnist_noniid_plus(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 30, 2000
    # num_shards = int(num_users * 2)
    # num_imgs = int(60000 / num_shards)
    # idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # idxs = np.arange(num_shards * num_imgs)
    # idxs = np.arange(60000)
    labels = dataset.targets.numpy()
    digits_labels_set = {}
    for i in range(10):
        count = sum(labels == i)
        digits_labels_set[i] = count


    # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs = idxs_labels[0, :]
    labels = labels.argsort()
    
    # divide and assign
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, 2, replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    # return dict_users
    
    # 0 1 2 3 4 5 6 7 8 9
    # 0 0 1 1 2 2 3 3 4,5 6,7,8,9
    # 2000 2000 2000
    start_idx = 0
    for i in range(8):
        digit = i // 2
        bit = i - digit * 2
        dict_users[i] = np.concatenate((dict_users[i], labels[start_idx + bit * 2000 : start_idx + (bit + 1) * 2000]), axis=0)
        if bit == 1:
            start_idx += digits_labels_set[digit]
    
    step = 1000
    for i in range(4, 6):
        dict_users[8] = np.concatenate((dict_users[8], labels[start_idx : start_idx + step]), axis=0)
        start_idx += digits_labels_set[i]

    step = 500
    for i in range(6, 10):
        dict_users[9] = np.concatenate((dict_users[9], labels[start_idx : start_idx + step]), axis=0)
        start_idx += digits_labels_set[i]
    print(f"len(dict_users): {len(dict_users)}")
    
    return dict_users

class DataSet(object):
    def __init__(self, clients_num):
        # whole data
        self.test_data = None
        self.test_label = None
        self.clients_num = clients_num

        # 加载mnist数据集
        data_train = datasets.MNIST(root="../datasets/", download=False, train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

        data_test = datasets.MNIST(root="../datasets/", download=False, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

        # split MNIST (training set) into non-iid data sets
        self.client_non_iid = []
        user_dict = mnist_noniid_plus(data_train, clients_num)
        # user_dict = mnist_noniid(data_train, clients_num)
        for i in range(clients_num):
            idx = user_dict[i]
            d = data_train.data[idx].float().unsqueeze(1)
            targets = data_train.targets[idx].float()
            self.client_non_iid.append((d, targets))
        self.test_data = torch.tensor(data_test.data.float().unsqueeze(1)).to(dev)
        self.test_label = torch.tensor(data_test.targets.float()).to(dev)

    def get_test_dataset(self):
        return self.test_data, self.test_label

    def get_train_batch(self, client_id):
        return TensorDataset(torch.tensor(self.client_non_iid[client_id][0]),
                             torch.tensor(self.client_non_iid[client_id][1]))


if __name__ == "__main__":
    dataset = DataSet(10)
    # test_data, test_labels = client_dataset.get_test_dataset()
    client_id = 9
    client_dataset = dataset.get_train_batch(client_id)
    print(f"client_id: {client_id}")
    print("type(client_dataset): ", type(client_dataset))
    print("type(client_dataset[:][0]): ", type(client_dataset[:][0]))
    print("type(client_dataset[:][0]): ", type(client_dataset[:][1]))
    print("len(client_dataset[:][0]): ", len(client_dataset[:][0]))
    print("len(client_dataset[:][1]): ", len(client_dataset[:][1]))
    print("client_dataset[0:10][1]: ", client_dataset[0:10][1])
    print("client_dataset[500:510][1]: ", client_dataset[500:510][1])
    print("client_dataset[1000:1010][1]: ", client_dataset[1000:1010][1])
    print("client_dataset[1500:1510][1]: ", client_dataset[1500:1510][1])
    # for i in range(10):
    #     client_dataset = dataset.get_train_batch(i)
    #     print(f"client_id: {i}")
    #     print("type(client_dataset): ", type(client_dataset))
    #     print("type(client_dataset[:][0]): ", type(client_dataset[:][0]))
    #     print("type(client_dataset[:][0]): ", type(client_dataset[:][1]))
    #     print("len(client_dataset[:][0]): ", len(client_dataset[:][0]))
    #     print("len(client_dataset[:][1]): ", len(client_dataset[:][1]))
    #     print("client_dataset[:10][1]: ", client_dataset[:10][1])
    #     print("client_dataset[-10:][1]: ", client_dataset[-10:][1])

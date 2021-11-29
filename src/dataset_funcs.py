import pandas as pd
from keras.utils import np_utils
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np


def load_all_dataset(filename, features, test_size=0.1):
    df = pd.read_csv(filename, names=features)
    df.sample(frac=1)
    n_records = len(df)
    data = df.iloc[:, 0:-1]
    labels = df.iloc[:, -1]
    x_train = data.iloc[:int(n_records * (1 - test_size))]
    y_train = labels.iloc[:int(n_records * (1 - test_size))]
    x_test = data.iloc[int(n_records * (1 - test_size)):]
    y_test = labels.iloc[int(n_records * (1 - test_size)):]

    n_features = 12
    n_classes = 20
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    ranges = [100, 100, 100, 20, 2000, 8000, 5, 128, 2000, 4000, 10000, 5]
    for i in range(len(ranges)):
        x_train.iloc[:, i] /= ranges[i]
        x_test.iloc[:, i] /= ranges[i]
    x_train = x_train.values
    x_test = x_test.values
    # x_train = x_train.reshape(len(x_train), 1, n_features, 1)
    # x_test = x_test.reshape(len(x_test), 1, n_features, 1)
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)
    return x_train, y_train, x_test, y_test


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((60000, 28*28))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.reshape((10000, 28*28))
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    dataset_path = "../datasets/computer_status_dataset.csv"
    features = ['cpu_usage', 'memory_usage', 'disk_usage', 'num_tasks',
                'bandwidth', 'mips', 'cpu_freq', 'cpu_cache', 'ram',
                'ram_freq', 'disk', 'pes_num', 'priority']
    x_train, y_train, x_test, y_test = load_all_dataset(dataset_path, features)
    # print(len(x_train))
    # print(len(y_train))
    # print(len(x_test))
    # print(len(y_test))
    # print("x_train")
    # print(x_train[:10])
    # print("x_test")
    # print(x_test[:10])
    print(np.shape(x_train))

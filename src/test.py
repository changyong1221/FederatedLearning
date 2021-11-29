from src.fedlib import *
from src.dataset_funcs import load_all_dataset, load_mnist
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

# settings
dataset_path = "../datasets/computer_status_dataset.csv"
features = ['cpu_usage', 'memory_usage', 'disk_usage', 'num_tasks',
            'bandwidth', 'mips', 'cpu_freq', 'cpu_cache', 'ram',
            'ram_freq', 'disk', 'pes_num', 'priority']

def create_model():
    n_features = 12
    n_classes = 20
    model = Sequential()
    input_shape = (1, n_features, 1)
    model.add(Conv2D(filters=32, kernel_size=(1, 3), activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(1, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    return model


def create_model_update():
    n_features = 12
    n_classes = 20
    model = Sequential()
    input_shape = (n_features, )
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    return model


def create_model_mnist():
    model = Sequential()
    model.add(Dense(512, activation="relu", input_shape=(28*28,)))
    model.add(Dense(10, activation="softmax"))
    return model

def train_one_model(i):
    sub_model_path = f"../models/train/{i}.npy"
    epoch = 2000
    x_train, y_train, x_test, y_test = load_all_dataset(dataset_path, features, test_size=0.1)
    # x_train, y_train, x_test, y_test = load_mnist()

    client_id = 1
    model = FedClient(model=create_model_update(), ID=client_id)
    model.setJob(jobAdress="x3tg83jx0m4jf8chyp5djas4jf9")

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train,
              y=y_train,
              batch_size=128,
              epochs=epoch,
              verbose=1)
    loss, acc = model.evaluate(x=x_test,
                               y=y_test,
                               batch_size=128)
    model.save_model(sub_model_path, weight=True)
    model.upload()
    print(f"Client-ID:{client_id} , loss:{loss} , acc:{acc}")
    print("training done.")


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


def test_federated_model():
    x_train, y_train, x_test, y_test = load_all_dataset(dataset_path, features, test_size=0.5)
    # x_train, y_train, x_test, y_test = load_mnist()

    client_num = 2
    sub_model_paths = []
    for i in range(client_num):
        sub_model_paths.append(f"../models/train/{i + 1}.npy")


    global_model = FedServer(model=create_model_update())

    global_model.load_client_weights(sub_model_paths)
    global_model.fl_average()

    global_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    loss, acc = global_model.evaluate(x=x_test,
                                      y=y_test, batch_size=128)
    print(f'global_loss:{loss}, global_acc:{acc}')


if __name__ == '__main__':
    # train_one_model(2)
    # for i in range(10):
    #     train_one_model(i+1)
    # test_one_model()
    test_federated_model()

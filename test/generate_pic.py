import matplotlib.pyplot as plt
import pandas as pd


def generate_pic():
    file_path = "global_acc.txt"
    data = pd.read_csv(file_path, header=None)
    records_num = len(data)
    diff_num = records_num // 10
    for i in range(diff_num):
        data = data.append(data[i * 10: i * 10 + 1])
    data = data[-193:]
    dest_dir = "./global_acc.png"
    plt_config = {
        "title": "accuracy of global model",
        "xlabel": "federated round",
        "ylabel": "accuracy"
    }
    save_to_pic(data, dest_dir, plt_config)


def generate_pic_local():
    file_path = "../results/global/global_acc.txt"
    data = pd.read_csv(file_path, header=None)
    data = data[:208]
    dest_dir = "./global_acc_local.png"
    plt_config = {
        "title": "accuracy of global model (local training)",
        "xlabel": "federated round",
        "ylabel": "accuracy"
    }
    save_to_pic(data, dest_dir, plt_config)


def save_to_pic(data, dest_dir, plt_config):
    # 1. plt configure
    plt.figure(figsize=(10, 6))
    plt.title(plt_config["title"])
    plt.xlabel(plt_config["xlabel"])
    plt.ylabel(plt_config["ylabel"])
    y_axis_data = data[0].tolist()
    clients_num = len(y_axis_data)
    x_axis_data = [i for i in range(clients_num)]
    plt.plot(x_axis_data, y_axis_data)
    plt.savefig(dest_dir)
    plt.close()


if __name__ == "__main__":
    generate_pic()
    # generate_pic_local()
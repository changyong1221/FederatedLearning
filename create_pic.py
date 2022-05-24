#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

# 设置汉字格式
plt.rcParams['font.sans-serif'] = ['Times new roman']  # 指定默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号


def plot_data():
    data_1 = pd.read_csv("backup/fed_test_400/w=0.6/global_acc.txt", header=None)
    data_2 = pd.read_csv("backup/fed_test_400/avg/global_acc.txt", header=None)
    data_3 = pd.read_csv("backup/fed_test_400/wl/global_acc.txt", header=None)
    data_4 = pd.read_csv("backup/fed_test_400/ad/global_acc.txt", header=None)
    # data_1 = pd.read_csv("backup/fed_ad/ad_data/100rounds/global_acc.txt", header=None)
    # data_2 = pd.read_csv("backup/fed_avg_no_avg_data/100rounds/global_acc.txt", header=None)
    data_1.columns = ['processing_time']
    data_2.columns = ['processing_time']
    data_3.columns = ['processing_time']
    data_4.columns = ['processing_time']
    # data = data.sort_values(by=['execute_time'], ascending=False)
    data_1_list = data_1['processing_time'].tolist()
    data_2_list = data_2['processing_time'].tolist()
    data_3_list = data_3['processing_time'].tolist()
    data_4_list = data_4['processing_time'].tolist()
    print(data_1_list)
    print(data_2_list)
    print(data_3_list)
    print(data_4_list)
    # 指定画布大小
    plt.figure(figsize=(10, 4))

    # 设置图标标题
    # plt.title(u"Mnist fed_ad_pa vs fed_avg")
    # plt.title(u"Mnist fed_wl vs fed_avg")
    plt.title(u"Mnist comparison")

    # 设置图表x轴名称和y轴名称
    plt.xlabel(u"epoch")
    plt.ylabel("accuracy")

    # 将数据进行处理，以便于画图（需要将DataFrame转为List）
    # x轴数据，只需一份即可
    linewidth = 2.5
    plt.plot(data_1_list, label="fed_optimized", linewidth=linewidth)
    plt.plot(data_2_list, label="fed_avg", linewidth=linewidth)
    plt.plot(data_3_list, label="fed_wl", linewidth=linewidth)
    plt.plot(data_4_list, label="fed_ad", linewidth=linewidth)
    # plt.plot(data_1_list, label="fed_ad_pa", linewidth=linewidth)
    # plt.plot(data_2_list, label="fed_avg", linewidth=linewidth)

    # 设置图例
    plt.legend(loc='best')

    # 保存图片
    # plt.savefig(f"backup/pic/results.png")
    plt.savefig(f"backup/fed_test_400/results_comparison_400rounds.png")

    # 展示折线图
    # plt.show()


if __name__ == '__main__':
    plot_data()

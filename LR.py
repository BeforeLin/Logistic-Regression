import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tqdm import trange
import time
import random

# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 初始化权值
def init_w(n):
    w = np.ones((n, 1))
    return w

# 传统梯度下降
def gradient_descent(x_train, y_train, w, alpha):
    y_hat = sigmoid(np.dot(x_train, w))
    # print(np.shape(y_hat))
#         print(np.shape(y_train))
    error = y_train.T - y_hat
#         print(error)
#         print(np.shape(error))
    w = w + alpha * np.dot(x_train.T, error)
    return w

# 普通SGD
def SGD(x_train, y_train, w, alpha):
    NumSample,NumFeatures=np.shape(x_train)

    example = random.randint(0, NumSample-1)
    x1 = np.array([x_train[example, :]])

    y_hat = sigmoid(np.dot(x1, w))
    error = y_train[:, example].T - y_hat
    w = w + alpha * np.dot(x1.T, error)
    return w

# 平滑SGD
def smoothSGD(x_train, y_train, w, iter):
    NumSample,NumFeatures=np.shape(x_train)

    example = random.randint(0, NumSample-1)
    x1 = np.array([x_train[example, :]])

    y_hat = sigmoid(np.dot(x1, w))
    error = y_train[:, example].T - y_hat
    alpha = float(10/(iter + 1))
    w = w + alpha * np.dot(x1.T, error)
    return w

# 模型训练
def log_regress_train(x_train, y_train, alpha, imax, gd_type):

    y_train = np.array([y_train])
    w = init_w(x_train.shape[1])
    for i in trange(imax):
        if gd_type == 0:
            w = gradient_descent(x_train, y_train, w, alpha)
        elif gd_type == 1:
            w = SGD(x_train, y_train, w, alpha)
        elif gd_type == 2:
            w = smoothSGD(x_train, y_train, w, i)
    return w

# 预测
def log_regress_predict(x_test, w):
    y_hat = []
    n = x_test.shape[0]

    acc_count = 0
    for i in range(n):
        p = sigmoid(np.dot(x_test[i, :], w))

        if p > 0.5:
            class_predict = 1
        else:
            class_predict = 0
        y_hat.append(class_predict)
    return y_hat

# 测试，计算准确率
def log_regress_test(x_test, y_test, w):
    n = x_test.shape[0]
    acc_count = 0
    for i in range(n):

        p = sigmoid(np.dot(x_test[i, :], w))

        if p > 0.5:
            class_predict = 1
        else:
            class_predict = 0
        if class_predict == y_test[i]:
            acc_count = acc_count + 1
    acc = acc_count/n
    return acc

# 绘制分类图的函数
def draw(w):
    N, M = 500, 500     
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)
    x_test = np.stack((x1.flat, x2.flat), axis=1)

    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_hat = np.array([log_regress_predict(x_test, w)])
    y_hat = y_hat.reshape(x1.shape)
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=50, cmap=cm_dark)
    plt.xlabel('PetalLength')
    plt.ylabel('PetalWidth')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()

iris = datasets.load_iris()
# 选取前两种花
y = iris.target[0:100]
# 选取花瓣长宽作为特征
x = iris.data[0:100,2:4]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.3)

alpha = 0.01
iteration = 4000

start = time.time()
w0 = log_regress_train(x_train, y_train, alpha, iteration, 0)
acc0 = str(log_regress_test(x_test, y_test, w0))
end = time.time()
process_t0 = str(end - start)
print("【批量梯度下降法】weight: " + str(w0[0]) + str(w0[1]) + "; acc: " + acc0 + "; process time: " + process_t0 + "\n")

start = time.time()
w1 = log_regress_train(x_train, y_train, alpha, iteration, 1)
acc1 = str(log_regress_test(x_test, y_test, w1))
end = time.time()
process_t1 = str(end - start)
print("【随机梯度下降法】weight: " + str(w1[0]) + str(w1[1]) + "; acc: " + acc1 + "; process time: " + process_t1 + "\n")

start = time.time()
w2 = log_regress_train(x_train, y_train, alpha, iteration, 2)
acc2 = str(log_regress_test(x_test, y_test, w2))
end = time.time()
process_t2 = str(end - start)
print("【改良随机梯度下降法】weight: " + str(w2[0]) + str(w2[1]) + "; acc: " + acc2 + "; process time: " + process_t2 + "\n")

# 分别绘图
draw(w0)
draw(w1)
draw(w2)
plt.show()
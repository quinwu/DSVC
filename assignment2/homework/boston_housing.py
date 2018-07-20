# coding=UTF-8
# 载入此项目所需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from sklearn.model_selection import train_test_split
# 检查你的Python版本
from sys import version_info

if version_info.major != 2 and version_info.minor != 7:
    raise Exception('请使用Python 2.7来完成此项目')

# 让可视化的结果在notebook中显示
# % matplotlib inline

# 载入波士顿房屋的数据集
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)

# 完成
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plt.title('RM')
plt.scatter(features.get_values()[:, 0], prices.get_values())
plt.subplot(2, 2, 2)
plt.title('LSTAT')
plt.scatter(features.get_values()[:, 1], prices.get_values())
plt.subplot(2, 2, 3)
plt.title('PTRATIO')
plt.scatter(features.get_values()[:, 2], prices.get_values())

# TODO 1

# 目标：计算价值的最小值
minimum_price = None

# 目标：计算价值的最大值
maximum_price = None

# 目标：计算价值的平均值
mean_price = None

# 目标：计算价值的中值
median_price = None

# 目标：计算价值的标准差
std_price = None

# change to array
# prices = np.array(prices)

maximum_price = np.max(prices)
minimum_price = np.min(prices)
mean_price = np.mean(prices)
median_price = np.median(prices)
std_price = np.std(prices)

# 目标：输出计算的结果
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))

LEN, seed = 489, None

# def generate_train_and_test(X, y):
#     """打乱并分割数据为训练集和测试集"""
#     train_test_split(X,y,test_size=0.2,random_state=0)
#
# #TODO
#     X = np.random.shuffle(X)#打乱特征数据
#     y = np.random.shuffle(y)#打乱价格
# seed = LEN * 0.8
# print(np.random.seed(seed))
# X_train = X[[np.random.seed(seed)]]
# X_test = X[[np.random.seed(LEN*0.2)]]
# y_train = y[[np.random.seed(seed)]]
# y_test = y[[np.random.seed(LEN*0.2)]]

# X_train, X_test, y_train, y_test = generate_train_and_test(features.values, prices.values)
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=0)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

# TODO 3

# 提示： 导入r2_score
from sklearn.metrics import r2_score


def performance_metric(y_true, y_predict):
    """计算并返回预测值相比于预测值的分数"""

    score = None

    score = r2_score(y_true, y_predict, sample_weight=None, multioutput=None)

    return score


# 计算这个模型的预测结果的决定系数
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))

data = np.array(data)
# TODO4
def compute_error(a, b, c, d, data):
    totalError = 0
    for i in range(0, len(data)):
        x1 = data[i, 0]
        x2 = data[i, 1]
        x3 = data[i, 2]

        y = data[i, 3]

    totalError += (y - (a * x1 + b * x2 + c * x3 + d)) ** 2


    return totalError / 2 * float(len(data))


# init a,b,c
def optimizer(data, init_a, init_b, init_c, init_d, learning_rate, num_iter):
    a = init_a
    b = init_b
    c = init_c
    d = init_d

    for i in range(num_iter):
        a, b, c, d = compute_gradient(a, b, c, d, data, learning_rate)

    if i % 100 == 0:
        print('iter {0}:error={1}'.format(i, compute_error(a, b, c, d, data)))

    return [a, b, c, d]


# compute gradient
def compute_gradient(a, b, c, d, data, learning_rate):
    a_gradient = 0
    b_gradient = 0
    c_gradient = 0
    d_gradient = 0

    # print(data)

    N = float(len(data))

    for i in range(0, len(data)):
        x1 = data[i, 0]
        x2 = data[i, 1]
        x3 = data[i, 2]
        y = data[i, 3]

        a_gradient = a_gradient - (1 / N) * learning_rate * ((a * x1 + b * x2 + c * x3 + d) - y) * x1
        b_gradient = b_gradient - (1 / N) * learning_rate * ((a * x1 + b * x2 + c * x3 + d) - y) * x2
        c_gradient = c_gradient - (1 / N) * learning_rate * ((a * x1 + b * x2 + c * x3 + d) - y) * x3
        d_gradient = d_gradient - (1 / N) * learning_rate * ((a * x1 + b * x2 + c * x3 + d) - y)

    new_a = a_gradient
    new_b = b_gradient
    new_c = c_gradient
    new_d = d_gradient
    return [new_a, new_b, new_c, new_d]


def train():
    learning_rate = 0.001
    init_a = 0
    init_b = 0
    init_c = 0
    init_d = 0

    num_iter = 10000

    print(
        'initial variables:\n initial_a = {0}\ninitial_b = {1}\ninitial_c = {2}\ninitial_d = {3}\n error of begin = {4} \n' \
        .format(init_a, init_b, init_c, init_d, compute_error(init_a, init_b, init_c, init_d, data)))

    [a, b, c, d] = optimizer(data, init_a, init_b, init_c, init_d, learning_rate, num_iter)

    print('final formula parmaters:\n a = {1}\n b={2}\n c={3}\n d={4}\n error of end = {5} \n'.format(num_iter, a, b,c,d,
                                                                                compute_error(a, b,c,d, data)))


if __name__ == '__main__':
    train()

# def predict(data):
# train(data)

'''
# 生成三个客户的数据
client_data = [[5, 17, 15], # 客户 1
               [4, 32, 22], # 客户 2
               [8, 3, 12]]  # 客户 3

# 进行预测
predicted_price = train(data)
for i, price in enumerate(predicted_price):
   print ("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))



#TODO 5

# 提示：你可能需要用到 X_test, y_test, performance_metric
# 提示：你需要使用编程练习 4 中得到参数数值进行预测
# 提示：你可能需要参考问题3的代码来计算R^2的值

r2 = 1

print ("Optimal model has R^2 score {:,.2f} on test data".format(r2))



'''

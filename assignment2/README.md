## assignment2

#### 准备工作

这个项目需要安装**Python 2.7**和以下的Python函数库

- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

其中，**Numpy**、**Matplotlib**是必须的安装的（一般在第一次作业中安装了Anaconda2.7后这两个包都是自动安装的），**scikit-learn**不是必须的。

#### 内容

- 通过学习线性回归算法来了解机器学习的一些基本的过程。
- 利用线性回归来完成作业中波士顿房价数据集的拟合与预测。

以下的内容可以帮助你更好的完成这个作业：

- [机器学习入门：线性回归及梯度下降](http://blog.csdn.net/xiazdong/article/details/7950084) 

- [小记Linear Regression](http://quinwu.org/2017/05/03/ML-Linear-Regression/)

- 我自己用Python可视化实现的一个Linear Regression的效果[Linear Regression Python implement](https://github.com/quinwu/ml_implementation/tree/master/Linear-Regression)

  给出这部分的代码仅在下面作业的时候作为一个参考，但是直接套用不一定会有很好的效果。

#### 作业

- Linear Regression 可视化（optional）

  独立使用Numpy、Matplotlib 来完成Linear Regression的可视化任务，就像给出的我的实现效果一样。数据集可以参考我上面的可视化效果的数据集，也可以选择Boston House Price的数据集或者其他的。帮助理解Linear Regression的过程。

- [波士顿房价预测](https://github.com/quinwu/DSVC/tree/master/assignment2/homework)

  在此项目中，我们将对为马萨诸塞州波士顿地区的房屋价格收集的数据应用本周学到的几个机器学习概念，以预测新房屋的销售价格。你首先将探索这些数据以获取数据集的重要特征和描述性统计信息。接下来，你要正确地将数据拆分为测试数据集和训练数据集，并确定适用于此问题的性能指标。然后，你将自己编写一个线性回归的模型，并使用不同的参数和训练集大小分析学习算法的性能图表。最后，你将根据一个新样本测试此模型并将预测的销售价格与你的统计数据进行比较。

  作业文件：`homework/boston_housing.ipynb`
## 作业说明

#### 数据集：CIFAR-10

[The CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html) consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
[download](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

如果上面的链接下载速度太慢（或者无法访问），可以再QQ群里下在数据。

下载到`cifar-10-python.tar.gz`文件后，解压。将解压后的文件夹拷贝到`/homework/DSVC/datasets/`路径下即可。

#### k-Nearest Neighbor classifier

Jupyter Notebook **knn.ipynb**将引导你实现k-NN分类器。

数据集: CIFAR-10

- 实现k-NN算法。
- 通过使用部分向量化加快k-NN中的距离矩阵计算。
- 通过交叉验证确定超参`k`的最佳值.

为了在本练习中更有效的代码执行，我们使用5000张图像进行训练和500张图像进行测试。 你可以自己改变它。

*暂时不考虑图像数据的特征提取以及其他更好的分类算法。*

## Optional 
This is your chance to show off! Try to get higher accuracy.

Design and implement a new type of feature and select an algorithm, and use them for image classification on CIFAR-10. 





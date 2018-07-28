## 作业

#### 数据集：CIFAR-10

[The CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html) consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
[download](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

如果上面的链接下载速度太慢（或者无法访问），可以再QQ群里下在数据。

下载到`cifar-10-python.tar.gz`文件后，解压。将解压后的文件夹拷贝到`/classwork/DSVC/datasets/`路径下即可。

#### k-Nearest Neighbor classifier

Jupyter Notebook **knn.ipynb**将引导你实现k-NN分类器。

数据集: CIFAR-10

##### 任务

- 实现k-NN算法。
- 通过使用部分向量化完成KNN算法。
- 通过使用全部矢量化来完成KNN算法。
- 通过交叉验证确定超参`k`的最佳值。

*暂时不考虑图像数据的特征提取以及其他更好的分类算法。*

## Optional 
This is your chance to show off! Try to get higher accuracy.

Design and implement a new type of feature and select an algorithm, and use them for image classification on CIFAR-10. 

## 说明

上面的optional为选做题，可以选择其他的分类算法来训练CIFAR-10数据集。

所有的文件，真正需要去完成只有两个文件。

-  `knn-en.ipynb`（英文版）或者`knn-zh.ipynb`（中文版）
-  `classwork/DSVC/classifiers/k_nearest_neighbor.py`

这两个文件，其他的都是一些辅助验证的程序，不需要理会。

- `.ipynb`文件只能通过`jupyter notebook `启动本地的`jupyter`服务器。默认端口是`8888`，在浏览器输入`localhost:8888`就可以。需要注意的是在`jupyter`的目录结构里找到你的`DSVC`项目文件的路径。
- `k_nearest_neighbor.py`文件在本地可以选择任意的编辑器去填写相关的程序。

#### 作业提交方式

- 提交方式：将完成的作业文件提交到[wukuan@qq.com](mailto:wukuan@qq.com)，邮件的标题为`姓名-assignment1`。
- 格式说明：
  - 需要提交的文件：`knn.ipynb`文件以及生成的`.html`文件和相关的资源文件、`k_nearest_neighbor.py`代码文件。
  - 将这些文件压缩成一个`.zip`文件， 命名与邮件标题相同。
  - 你提交的作业版本应与Github上的最新版本一致。
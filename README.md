[![image](https://img.shields.io/badge/Author-ZeroFreeze-brightgreen)](https://github.com/)

# TensorFlow 2 Examples
This is a repo about TensorFlow tutorial python files translated from official TensorFlow 2.x tutorial site. 
You can run those file directly by python interpreter or any Python IDE with tensorflow.
If you find this repo useful，please give me a star ^_^ <br/>

>这个仓库是我个人整理的一系列 TensorFlow 示例代码，一部分翻译自谷歌的 TensorFlow 2.x 官方文档。你可以直接运行这些文件或者用任何支持Python的IDE来运行。你也可以在编写自己的训练过程时，用这些代码来快速调用TensorFlow API.
>如果你喜欢或觉得有用，请赏我一个star ^_^

## Requirments 环境需求

The first step is to install python3. You'd better have version 3.7 installed to fit TensorFlow 2. If you don't have a right version, we recommend to use conda.
Remember, always update your pip (the package) to the latest version to meet the requirement for TensorFlow 2.x.<br/>

>首先是要配置好Python3环境，最好是3.7。如果你版本不对，我们建议使用conda来配置一个新的虚拟环境。记住始终更新pip到最新以配合TensorFlow的正常安装。

```shell
pip install --upgrade pip
```

Using those pip command to install missing packages in your python enviroment. We recommened to use ver2.1 or higher because ver2.0 has many bugs.<br/>
You can also use conda enviroment to install the following packages. <br/>

>用下面的命令去安装你的环境中没有的 pip 包，Anaconda的小伙伴可以使用 conda install 来给虚拟环境配置好。推荐使用Tensorflow 2.1或更高版本，因为2.0存在不少bug。

```shell
pip install tensorflow>=2.0.0
pip install -q git+https://github.com/tensorflow/docs
pip install matplotlib pycocotools opencv-python
```

```shell
conda install tensorflow
conda install matplotlib pycocotools opencv-python
```

## Files 文件说明

`tensorflowexample.sln` Those python files were written in Visual Studio 2019 with python support, so you can open it directly with VS2019.<br/>
>整个项目是用VS2019写的，所以你可以用Visual Studio导入sln文件直接打开它。

`tensorflowexample.py` This file is a quick example with comments for TensorFlow ver 1.x compact.<br/>
>这个文件是一个学习线性回归的入门文档，使用兼容TensorFlow 1.x的API编写的。

`envcheck.py` Run this file to checkout your TensorFlow enviroment and supports such as CUDA or MKL support.<br/>
>运行此文件可以检查你的TensorFlow环境和其他支持，比如是否支持GPU或是mkl库。

`mnistexample.py` Reorganized from Google official TensorFlow 2 tutorial, use MNIST dataset to train a simple model and evaluate.<br/>
>从谷歌TensorFlow文档编排的，快速训练一个MNIST手写数据集并计算准确度。

`mnistlayers.py` Compare the effects of different layers in one MNIST dataset.<br/>
>比较不同大小层级的模型对于MNIST数据集和训练结果的影响。

`cifarwithvgg.py` Build a VGG network manually and train with CIFAR10/100 datasets, you can save checkpoint while training.<br/>
>用 `tf.keras` API手动搭建一个VGG网络，并使用CIFAR10/100数据集进行训练，同时保存训练进度。

`coco_init.py` Prepare MS COCO datasets using `tf.data.TFRecordDataset` API.<br/>
>使用 `tf.data.TFRecordDataset` API初始化MS COCO数据集。

`cocoexample.py` Training with MS COCO datasets. You must run `coco_init.py`correctly once before using this example.<br/>
>使用MS COCO进行训练。运行之前，需要先保证初始化了COCO数据集。

## Contact

My Userpage in Zhihu.com : [https://www.zhihu.com/people/ren-sheng-ru-lu](https://www.zhihu.com/people/ren-sheng-ru-lu)<br/>
>我的知乎主页： [https://www.zhihu.com/people/ren-sheng-ru-lu](https://www.zhihu.com/people/ren-sheng-ru-lu)

Gitee Repository : [https://gitee.com/zerofreeze/tensorflow-2-example](https://gitee.com/zerofreeze/tensorflow-2-example)<br/>
>本仓库的码云镜像：[https://gitee.com/zerofreeze/tensorflow-2-example](https://gitee.com/zerofreeze/tensorflow-2-example)

If you have idea about any improvment in this repository or other TensorFlow questions, welcome to contact me at any time.
>如果你对改进这些代码有任何意见，随时欢迎来找我交流~~

## Attention
This repository can be used only for experimental or personal studies.


---
layout: post
title: "Tensorflow 初探"
tags: tensorflow
categories: 深度学习
---
### 引言
> 机器学习领域目前最火的非深度学习莫属。自从Alphago战胜了李世乭，深度学习以势不可挡之势的进入了大众视野。深度学习极大的提高了图片识别、语音识别和自然语言处理的准确率。Google作为这方面研究的开拓者之一，开源了其深度学习包tensorflow。本文就按照tensorflow官方文档里的mnist tutorial，学习一下如何使用tensorflow训练一个简单的神经网络。
ps：本文不涉及神经网络的[基础知识](http://neuralnetworksanddeeplearning.com/index.html)

### TensorFlow安装  

为了避免跟本地的`python`包混淆，先使用`virtualenv`创建一个虚拟环境。 

    mkvirtualenv tensorflow --python=python3

要执行上面的命令需要两个`packages`：`virtualenv`和`virtualenvwrapper`。 

`pip`安装。先找到相应的[binary](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#pip-installation)。
例如我是Mac OS X，CPU only, Python 3:  

    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0rc0-py3-none-any.whl
    pip install $TF_BINARY_URL

如果要安装GPU支持，首先安装[Cuda](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#optional-install-cuda-gpus-on-linux)。 

### 基本概念
为了对tensorflow的工作原理有一个整体认识，先总结几个常见概念。

* Graph. 描绘了整个神经网络的结构，代表计算过程 
* Ops. 即Operations
* Tensor. 张量矩阵，是Op中的输入输出数据
* Session. Graph只有在一个session的环境下才能被执行
* Variable. 一种Op, 常用来存储需要优化的参数
* PlaceHolder. 一种Op, 在训练时需要传入数据，比如输入的图片数据

### MNIST Tutorial
对于了解机器学习的童鞋，MNIST数据应该是比较熟悉了，里面是一些手写的阿拉伯数字的图片，通过机器学习的算法，来识别图片上对应的数字。
下面利用tensorflow来训练一个非常简单的Neural Network来完成这一任务。

#### 加载数据
首先利用tensorflow提供的函数来下载MNIST数据集

~~~ python
from tensorflow.examples.tutorial.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
~~~~

数据由三部分training、validation和testing组成。

#### 构造Graph
首先创建两个`PlaceHolder`接受图片输入和标签

~~~ python
import tensorflow as tf
x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])
~~~
x的形状第一个参数为None，指batch大小为任意值。第二个参数784是图片像素的个数28*28。
同理，y_有10个分类0～9。两者的数据类型都是浮点型。

首先定义几个Helper函数，来辅助后面的神经网络构造。

~~~ python
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.con2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
~~~
前两个函数用于初始化参数，后面两个分别是构造卷积层和池化层。有关参数设定的意义参见[这里](https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html#convolution)。

**两层卷积**
一个卷积层其实包括三层：卷积层、激活层和池化层。这里简称为一个卷积层。
输入为28x28的图片，经过两层卷积，最后输出为7x7x64的feature map。

~~~ python
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
~~~

**全连接层**
用一层含有1024个神经元的全连接层来整合卷积层得到的所有features。

~~~  python
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flattened = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flattened, w_fc1) + b_fc1)
~~~

**Dropout Layer**
为了避免过拟合，加一层dropout。dropout概率通过placeholder在训练时传入。

~~~ python
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)
~~~~

**Output Layer**
最后用softmax输出。

~~~ python
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y = tf.nn.sofmax(tf.matmul(h_fc1_dropout, w_fc2) + b_fc2)
~~~

**训练及评估模型**
损失函数使用交叉熵损失，用adam算法来优化。
*注意点*：在训练前，需要先初始化一个Session。（变量初始化和训练都是在Session里面运行）
可以这样

    sess = tf.Session()
    sess.run()

也可以使用`with`

    with tf.Session() as sess:
        sess.run()

附上训练的代码。最后测试的准确率大概在99.2%。

~~~ python
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
~~~











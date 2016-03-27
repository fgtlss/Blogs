# 决策森林和卷积神经网络二道归一

现在有很多人认为神经网络可以和人脑中的机制相似。我却认为，或许人脑中有的机制与此类似，但一定是一个复杂的系统。因为人脑的运行没有那么快，却能识别大千世界。所以直观上看人脑应该是知识库加快速索引加级联识别算法，之所以用级联是因为要保证速度。

但我们其实可以完全不必模仿人脑的构造，因为人工建立的智能一定比人脑在各个方面都要强上百倍，也正如学飞行不能看雄鹰振翅而是空气动力学一样。

我认为人脑最重要的机制是元推理能力，所谓的元推理能力是推理能力的最小集合，基于此可以衍生出更精确的更强大的推理。当然，记忆存储和感知识别系统算是外设。比如福尔摩斯为何会那么聪明？其实是三点皆备，第一，经验丰富，数据库存储的东西多，很多事情是知道就知道不知道就不知道的东西。第二，数据库索引快且完备，根据一个事物能很快联想到它的发生原理。第三，感官能力强，望闻问切无不敏锐。这才造就了福尔摩斯。

因为我的如此认识，所以当我看到有一片论文是将决策森林和卷积神经网络糅合到一起的时候，我感觉到`something is more close.`

本博客是论文笔记，该论文是MSA的工作，引用见最后，特此声明。

# 两个完全不一样的模型

决策森林和卷积神经网络其实是完全不同类型的模型，卷积神经网络是一层到一层的密集运算，是层级结构，可以看做是有向图；而决策森林则是通过一个节点来决定将数据发送到哪个子节点再作运算，即数据路由的效果，是树级结构。

但是，也有一些神经网络可以认为与树沾边，比如GoogLeNet，还有AlexNet的双GPU版，甚至于DeepID中多层共同与隐含层相连。

# CNN的变异

得益于ReLU，CNN中隐含层有相当一部分的数值会为0。从而，在衡量相关性的时候，发现是这样子的。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/1.png)

左边是原图，中间是layer1与layer2之间的相关性，注意，相关性同权值参数是不同的东西。越白色的点越趋近于0。那么将这些相关性重新排列之后，如右图所示。可以发现，只有一些比较规整的矩形部分是强相关的。

那么可以通过删除不相关的网络连接，达到删减网络的目的。删减之后的网络结构如下图。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/2.png)

可以看到，对于稀疏连接的CNN，也有一种数据路由的效果，即数据经过筛选之后发送给不同的节点。


# 表示方法的统一

正其名且得其实，想要把两种模型统一起来，还是得先把他们的表示方法统一起来。

首先看CNN网络的表示方法。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/3.png)

其中P是线性变换，P后的竖着的波浪线是非线性变换，非线性变换包括sigmoid、ReLU、Dropout等。

然后是树的表示方法。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/4.png)

在该图中，I是单位矩阵，S是选择子集使用的矩阵，I和S都可以看做是线性变换的一种特例。P<sup>R</sup>是路由节点，它输出一系列概率来决定数据该被发往那个子节点。可以是Single Best模式，也可以是Multi-Way模式，还可以是Soft-Routing模式（发送到所有子节点）。

# 计算量的节省

## 直接数据路由

直接数据路由包括两种模式，一种就是决策树中的数据传送，还有一种就是上述所说的选择子集。当下一步的计算在原有数据的子集上的时候，自然计算量要有相应的降低。

## 隐式数据路由

如下图，将过滤器分成两组，可以减少过滤器之间的联系，原来是100%×100%，现在是2×50%×50%，计算量少了一半。而且更容易并行。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/5.png)

# 后向传播

那么，如果将CNN和决策树放到一起，对于隐式数据路由，可以直接进行反向传播，对于子集选择，需要添加一个额外的参数S，也可以直接进行反向传播。对于single best树（只传到某个子节点）来说，反向传播是个大问题。所以不得不将决策树在soft-routing（将数据传到所有子节点）下来做反向传播，然后在测试时用single best树。

例如，对于如下神经网络来说：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/6.png)

使用平方差损失函数，V<sup>*</sup>是真实标记。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/7.png)


![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/8.png)

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/9.png)

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/10.png)

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/11.png)

# 实验

## VGG最后一层上的toy实验

使用ImageNet1000类数据集。

VGG前面不变，将最后一层全连接和softmax层变成树状结构。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/12.png)

- 路由函数使用感知机模型来做，输出值使用softmax进行归一化作为概率
- 由于图中将数据发送给四个子节点，是使用的单位矩阵，所以，我猜测应该是把输出向量进行了切分。这样才能达到，如果只传给一个子节点，大大减少计算量，如果传给所有节点，计算量同全连接相似。这一点论文仍然没有说明。

效果如下：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/13.png)

- 效果图中是测试时间与准确率的权衡。
- 每条曲线的变化的参数是multi-routes数目，即将数据发送给几个子节点（我自己猜的）。
- 可以看到，效果的提升和时间的增加是次线性的，所以随着树的分叉的增多，可以在更多的时间节省下同时保证效果的降低在一定的范围之内。

## 树状卷积神经网络

使用ImageNet1000类数据集。

上述实验只是在最后一层上验证了树与CNN结合的好处。在卷积层使用分叉结构的效果并未被测验。此次使用的网络结构如下：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/14.png)

该网络基于这样一个假设，即：每个filter应该只需要在一些而不是全部的输入feature map上做卷积即可。

所以，网络把filter分成组，每个组的filter处理上一层中特定组的feature map。由上图也可以看到，上面的从3以上的奇数层，其filter分组的个数为2<sup>n-2</sup>个，偶数层的filter分组个数与奇数层一致。

训练参数：

- 由于在最后一层卷积层之后做全局pooling可以在微量降低效果的情况下大大减少参数的数目，所以在最后一层卷积层之后全连接层之前进行全局pooling
- 参数初始化方法与论文参考文献[9]相同
- 学习速率衰减

	![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/15.png)
- 当验证集上的准确率上升一个层次的时候，学习率10倍衰减，这样进行两次。
- 该模型训练迭代次数相对于VGG11来说是两倍。
- 只使用mirroring和random crops来做数据预处理。

效果如下：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/16.png)

由图中可见，本网络除了逊于GoogLeNet之外，要强于其他的网络。

## Cifar10上的实验

- 使用NIN作为对照模型，为了简化，将NIN模型第一层的192filter变为64filter。
- 直接通过Bayesian Search来进行模型的自学习。自学习过程中优化
	- alpha=![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/17.png)
	- 学习到的网络如下：
	![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/18.png)
- 为了对比，同时对NIN网络进行Bayesian优化。

> Bayesian Search还不了解，需要再看看原始论文。

效果如下，其中：

- 菱形代表没有数据路由的CNN
	- 原始NIN使用红色表示
	- 进行过删减优化的NIN使用粉色表示
- 圆形代表有数据路由的CNN
	- 300个带数据路由的网络使用灰色表示
	- 绿色的则表示最优解

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/19.png)

## CNN组合提升

还有一种数据路由的方法就是CNN的组合，即将两个CNN组合到一起，然后使用路由感知器进行组合。如下图：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/20.png)

两条分支都是使用GoogLeNet，但在测试时，上面的路径不使用oversampling，下面的路径使用10倍oversampling。

> Oversampling我觉得应该是图像的crop，四个角加中心和对称的四个角加中心。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_cnn_df/21.png)

可以看到，通过这种方式，可以在保证效果的前提下，把计算量降低一半。


# Tips

在GPU上，由于有数据传输的时间，所以总体计算操作的减少并不能线性的导致运算时间的降低，所以，将filter进行分组可以导致数据传输量的减少，从而进一步的提升速度。

现在有两种基础的并行化：

- 矩阵运算的并行化（BLAS）
- 数据并行化，（mini-batch）

# 总结

探索了树与CNN之间的各种组合形式。主要包括：

- 将隐含层和输出层变成树状结构
- 卷积filter分组，然后通过假设下一层的filter应该只在一部分channels上做来把卷积层变成树状结构。
- 通过Bayesian Search来自学习网络结构。（这个感觉很有意思）
- 把CNN当做黑盒子，将不同的CNN组合形成树状结构。


# 参考文献

[1]. Ioannou Y, Robertson D, Zikic D, et al. Decision Forests, Convolutional Networks and the Models in-Between[J]. arXiv preprint arXiv:1603.01250, 2016.
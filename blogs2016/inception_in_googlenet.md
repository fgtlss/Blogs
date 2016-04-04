# Inception

之前也写过GoogLeNet的笔记，但那个时候对Inception有些似懂非懂，这周重新看了一遍，觉得有了新的体会，特地重新写一篇博客与它再续前缘。

# Network in Network

GoogLeNet提出之时，说到其实idea是来自NIN，NIN就是Network in Network了。

NIN有两个特性，是它对CNN的贡献：

- MLP代替GLM
- Global Average Pooling

## mlpconv

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_inception/1.png)

普通的卷积可以看做是比较特殊的GLM，GLM就是广义线性模型。那么MLP是指，在做卷积操作的时候，把线性操作变为多层感知机。

这个idea的理论基础是多层感知机的抽象能力更强。假如我们把从图像中抽取出来的特征称作是这个图像的隐含概念（只是一个名称罢了，不要过度追究），那么如果隐含概念是线性可分的，那么，GLM抽取出来的特征没有问题，抽象表达能力刚刚好。但是假如隐含概念并不是线性可分的，那么就悲剧了，在只使用GLM的情况下，不得不过度的使用filter来表现这个隐含概念的各个方面，然后在下一层卷积的时候重新将这些概念组合，形成更加抽象的概念。

所以，基于如上，可以认为，在抽特征的时候直接做了非线性变换，可以有效的对图像特征进行更好的抽象。

从而，Linear convolution layer就变成了Mlpconv layer。

值得一提的是，Mlpconv相当于在正常的卷积层后面，再添加一个1×1的卷积层。

## Global Average Pooling

Global Average Pooling的做法是将全连接层去掉。

全连接层的存在有两个缺点：

- 全连接层是传统的神经网络形式，使用了全连接层以为着卷积层只是作为特征提取器来提取图像的特征，而全连接层是不可解释的，从而CNN也不可解释了
- 全连接层中的参数往往占据CNN整个网络参数的一大部分，从而使用全连接层容易导致过拟合。

而Global Average Pooling则是在最后一层，将卷积层设为与类别数目一致，然后全局pooling，从而输出类别个数个结果。

使用了mlpconv和Global Average Pooling之后，网络结构如下：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_inception/2.png)

# Inception

读google的论文，你立马会感到一股工程的气息扑面而来。像此时的春风一样，凌厉中透着暖意。所谓凌厉，就是它能把一个idea给用到节省内存和计算量上来，太偏实现了，所谓暖意，就是真的灰常有效果。

自2012年AlexNet做出突破以来，直到GoogLeNet出来之前，大家的主流的效果突破大致是网络更深，网络更宽。但是纯粹的增大网络有两个缺点——过拟合和计算量的增加。

解决这两个问题的方法当然就是增加网络深度和宽度的同时减少参数，为了减少参数，那么自然全连接就需要变成稀疏连接，但是在实现上，全连接变成稀疏连接后实际计算量并不会有质的提升，因为大部分硬件是针对密集矩阵计算优化的，稀疏矩阵虽然数据量少，但是所耗的时间却是很难缺少。

所以需要一种方法，既能达到稀疏的减少参数的效果，又能利用硬件中密集矩阵优化的东风。Inception就是在这样的情况下应运而生。

第一步，将卷积分块，所谓的分块就是其实就是将卷积核分组，既然是分组索性就让卷积和不一样吧，索性使用了1×1，3×3，5×5的卷积核，又因为pooling也是CNN成功的原因之一，所以把pooling也算到了里面，然后将结果在拼起来。这就是最naive版本的Inception。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_inception/3.png)

对于这个Inception，有两点需要注意：

- 层级越高，所对应的原始图片的视野就越大，同样大小的卷积核就越难捕捉到特征，因而层级越高，卷积核的数目就应该增加。
- 1×1，3×3，5×5 只是随意想出来的，不是必须这样。

这个naive版的Inception，还有一个问题，因为所有的卷积核都在上一层的所有输出上来做，那5×5的卷积核所需的计算量就太大了。因而，可以采用NIN中的方法对上一层的输出进行Merge。这样就衍生出了真正可用的Inception。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_inception/4.png)

这个结构利用了NIN结构中非线性变换的强大表达能力。

同时，正如上一篇博客[决策森林和卷积神经网络二道归一](http://blog.csdn.net/stdcoutzyx/article/details/50993124)中的隐式数据路由，计算量也大大减少，因为四个分支之间是不需要做计算的。

再同时，还具有不同的视野尺度，因为不同尺寸的卷积核和pooling是在一起使用的。

> 旁白赞曰：其谋略不可为不远，其心机不可谓不深啊。

> 之前心中一直有个疑问，那就是max-pooling之后的feature_map不是应该长宽都减半了么，那怎么与conv的输出拼接。后来才想到，是自己被theano中的实现误导了，theano的实现是自动就缩小了，但stride为1的时候，max_pooling的输出还可以是长宽不变。

GoogLeNet的模型参数详细如下：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_inception/5.png)

结构如下：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_inception/6.png)

需要注意的是，为了避免梯度消失，网络额外增加了2个辅助的softmax用于向前传导梯度。文章中说这两个辅助的分类器的loss应该加一个衰减系数，实际测试的时候，这两个额外的softmax会被去掉。


# Inception-V2

Google的论文还有一个特点，那就是把一个idea发挥到极致，不挖干净绝不罢手。所以第二版的更接近实现的Inception又出现了。Inception-V2这就是文献[3]的主要内容。

Rethinking这篇论文中提出了一些CNN调参的经验型规则，暂列如下：

- 避免特征表征的瓶颈。特征表征就是指图像在CNN某层的激活值，特征表征的大小在CNN中应该是缓慢的减小的。
- 高维的特征更容易处理，在高维特征上训练更快，更容易收敛
- 低维嵌入空间上进行空间汇聚，损失并不是很大。这个的解释是相邻的神经单元之间具有很强的相关性，信息具有冗余。
- 平衡的网络的深度和宽度。宽度和深度适宜的话可以让网络应用到分布式上时具有比较平衡的computational budget。

## Smaller convolutions

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_inception/7.png)

简而言之，就是将尺寸比较大的卷积，变成一系列3×3的卷积的叠加，这样既具有相同的视野，还具有更少的参数。

这样可能会有两个问题，
- 会不会降低表达能力？
- 3×3的卷积做了之后还需要再加激活函数么？（使用ReLU总是比没有要好）

实验表明，这样做不会导致性能的损失。

> 个人觉得，用大视野一定会比小视野要好么？ 叠加的小视野还具有NIN的效果。所以，平分秋色我觉得还不能说是因为某个原因。

于是Inception就可以进化了，变成了

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_inception/9.png)

## Asymmetric Convoluitons

使用3×3的已经很小了，那么更小的2×2呢？2×2虽然能使得参数进一步降低，但是不如另一种方式更加有效，那就是Asymmetric方式，即使用1×3和3×1两种来代替3×3. 如下图所示：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_inception/10.png)

使用2个2×2的话能节省11%的计算量，而使用这种方式则可以节省33%。

于是，Inception再次进化。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_inception/11.png)

> 注意：实践证明，这种模式的Inception在前几层使用并不会导致好的效果，在feature_map的大小比较中等的时候使用会比较好



## Auxiliary Classifiers

在GoogLeNet中，使用了多余的在底层的分类器，直觉上可以认为这样做可以使底层能够在梯度下降中学的比较充分，但在实践中发现两条：

- 多余的分类器在训练开始的时候并不能起到作用，在训练快结束的时候，使用它可以有所提升
- 最底层的那个多余的分类器去掉以后也不会有损失。
- 以为多余的分类器起到的是梯度传播下去的重要作用，但通过实验认为实际上起到的是regularizer的作用，因为在多余的分类器前添加dropout或者batch normalization后效果更佳。


## Grid Size Reduction

Grid就是图像在某一层的激活值，即feature_map，一般情况下，如果想让图像缩小，可以有如下两种方式：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_inception/12.png)

右图是正常的缩小，但计算量很大。左图先pooling会导致特征表征遇到瓶颈，违反上面所说的第一个规则，为了同时达到不违反规则且降低计算量的作用，将网络改为下图：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_inception/13.png)

使用两个并行化的模块可以降低计算量。

## V2-Inception

经过上述各种Inception的进化，从而得到改进版的GoogLeNet，如下：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_inception/14.png)

图中的Figure 4是指没有进化的Inception，Figure 5是指smaller conv版的Inception，Figure 6是指Asymmetric版的Inception。

## Label Smoothing

除了上述的模型结构的改进以外，Rethinking那篇论文还改进了目标函数。

原来的目标函数，在单类情况下，如果某一类概率接近1，其他的概率接近0，那么会导致交叉熵取log后变得很大很大。从而导致两个问题：

- 过拟合
- 导致样本属于某个类别的概率非常的大，模型太过于自信自己的判断。

所以，使用了一种平滑方法，可以使得类别概率之间的差别没有那么大，

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_inception/15.png)

用一个均匀分布做平滑，从而导致目标函数变为：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_inception/16.png)

该项改动可以提升0.2%。

> Rethinking 那篇论文里还有关于低分辨率的输入的图像的处理，在此不赘述了。



















# 参考文献

[1]. Lin M, Chen Q, Yan S. Network in network[J]. arXiv preprint arXiv:1312.4400, 2013.

[2]. Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015: 1-9.

[3]. Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the Inception Architecture for Computer Vision[J]. arXiv preprint arXiv:1512.00567, 2015.
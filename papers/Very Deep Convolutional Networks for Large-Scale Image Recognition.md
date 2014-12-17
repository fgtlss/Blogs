# Very Deep Convolutional Networks for Large-Scale Image Recognition
> 这篇论文是今年9月份的论文[1]，比较新，其中的观点感觉对卷积神经网络的参数调整大有指导作用，特总结之。关于卷积神经网络(Convolutional Neural Network, CNN)，笔者后会作文阐述之，读者若心急则或可用谷歌百度一下。
> 本文以下内容即是论文的笔记，笔者初次尝试对一篇论文提取重点做笔记，若有不足之处请阅读原文者指出。
## 1. Main Contribution+ 考察在参数总数基本不变的情况下，CNN随着层数的增加，其效果的变化。+ 论文中的方法在ILSVRC-2014比赛中获得第二名。
 	+ ILSVRC——ImageNet Large-Scale Visual Recongnition Challenge
## 2. CNN improvement
> 在论文[2]出现以后，有很多对其提出的CNN结构进行改进的方法。例如：
+ Use smaller receptive window size and smaller stride of the first convolutional layer.+ Training and testing the networks densely over the whole image and over multiple scales.## 3. CNN Configuration Principals
+ CNN的输入都是224×224×3的图片。+ 输入前唯一的预处理是减去均值。+ 1×1的核可以被看成是输入通道的线性变换。+ 使用较多的卷积核大小为3×3。+ Max-Pooling 一般在2×2的像素窗口上做，with stride 2。+ 除了最后一层全连接的分类层外，其他层都需要使用rectification non-linearity(RELU)。+ 不需要添加Local Response Normalization(LRN)，因为它不提升效果反而会带来计算花费和内存花费，增加计算时间。
## 4. CNN Configuration+ 卷积层的通道数目（宽度）从64，每过一个max-pooling层翻倍，到512为止。+ Use filters with 3×3 size throughout the whole net, because a stack of two 3×3 conv layers (without spatial pooling in between) has an effective receptive of 5×5, and three a stack of 3×3 conv layers has a receptive of 7×7, and so on.+ 为甚么使用三层3×3代替一层7×7？ 	+ 第一，三层比一层更具有判别性；	+ 第二，假设同样的通道数C，那么三层3×3的参数数目为3×(3×3)C×C=27C×C，一层7×7参数数目为7×7×C×C=49C×C。大大减少了参数数目。+ 使用1*1的卷积核可以在不影响视野域的情况下增加判别函数的非线性。该核可以用于“Network in Network”网络结构，可以参考论文的参考文献12。
+ 图1是论文中实验使用的神经网络结构，可以看到，CNN的层数从11层到19层，结构符合上面的总结的点。图2则是各个CNN的参数总数，可以看到，虽然深度变化了，但是参数数目变化不大。 
![image1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/papers/imgs/1-1.png)
<center>Figure1 Convnet Configuration</center>
![image2](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/papers/imgs/1-2.png)
<center>Figure2 Parameter Num</center>
	## 5. Training+ 除了使用multiple scale之外，论文[1]实验基本都follow论文[2]的设置。batch size是256，momentum是0.9，正则化系数是5×10e-4，前两层全连接的dropout参数设置为0.5，学习步长初始化为10e-2，且当验证集结果不再上升时步长除以10，除三次为止。学习了370K迭代(74 epochs)时停止。+ 论文推测，本文的网络比原来的网络要更容易收敛，原因有二：	+ Implicit regularization imposed by greater depth and smaller conv filter sizes		+ Pre-initialisation of certain layers. 先训练浅层网络，如图中的A网络，得到参数后，当训练更深的网如E时，使用A中得到的参数初始化对应的层，新层的参数则随机初始化。需要注意的是，使用这样的方式进行初始化，不改变步长。	+ 224×224输入的获得，将原始图片等比例缩放，保证短边大于224，然后随机选择224×224的窗口，为了进一步data augment，还要考虑随机的水平仿射和RGB通道切换。+ Multi-scale Training， 多尺度的意义在于图片中的物体的尺度有变化，多尺度可以更好的识别物体。有两种方法进行多尺度训练。	+ 在不同的尺度下，训练多个分类器，参数为S，参数的意义就是在做原始图片上的缩放时的短边长度。论文中训练了S=256和S=384两个分类器，其中S=384的分类器的参数使用S=256的参数进行初始化，且将步长调为10e-3。	+ 另一种方法是直接训练一个分类器，每次数据输入时，每张图片被重新缩放，缩放的短边S随机从[min, max]中选择，本文中使用区间[256,384]，网络参数初始化时使用S=384时的参数。
	## 6. Testing> 测试使用如下步骤：
+ 首先进行等比例缩放，短边长度Q大于224，Q的意义与S相同，不过S是训练集中的，Q是测试集中的参数。Q不必等于S，相反的，对于一个S，使用多个Q值进行测试，然后去平均会使效果变好。+ 然后，按照本文参考文献16的方式对测试数据进行测试。	+ 将全连接层转换为卷积层，第一个全连接转换为7×7的卷积，第二个转换为1×1的卷积。	+ Resulting net is applied to the whole image by convolving the filters in each layer with the full-size input. The resulting output feature map is a class score map with the number channels equal to the number of classes, and the variable spatial resolution, dependent on the input image size.	+ Finally, class score map is spatially averaged(sum-pooled) to obtain a fixed-size vector of class scores of the image.
	## 7. Implementation+ 使用C++ Caffe toolbox实现	+ 支持单系统多GPU	+ 多GPU把batch分为多个GPU-batch，在每个GPU上进行计算，得到子batch的梯度后，以平均值作为整个batch的梯度。	+ 论文的参考文献[9]中提出了很多加速训练的方法。论文实验表明，在4-GPU的系统上，可以加速3.75倍。
## 8. Experiments> 共进行三组实验：
### 8.1	Configuration Comparison> 使用图1中的CNN结构进行实验，在C/D/E网络结构上进行多尺度的训练，注意的是，该组实验的测试集只有一个尺度。如下图所示：
![image3](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/papers/imgs/1-3.png) <center>Figure3 Performance at a single test scale</center>
### 8.2	Multi-Scale Comparison> 测试集多尺度，且考虑到尺度差异过大会导致性能的下降，所以测试集的尺度Q在S的上下32内浮动。对于训练集是区间尺度的，测试集尺度为区间的最小值、最大值、中值。![image4](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/papers/imgs/1-4.png)
<center>Figure4 Convnet performance at multiple test scales</center>
### 8.3	Convnet Fusion> 模型融合，方法是取其后验概率估计的均值。
> 融合图3和图4中两个最好的model可以达到更好的值，融合七个model会变差。
![image5](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/papers/imgs/1-5.png)
<center>Figure5 Convnet Fusion</center> ## 9. Reference
> [1]. Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. arXiv preprint arXiv:1409.1556, 2014.
> [2]. Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[C]//Advances in neural information processing systems. 2012: 1097-1105.
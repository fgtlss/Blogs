# 局部敏感哈希之KSH在[局部敏感哈希](http://blog.csdn.net/stdcoutzyx/article/details/44456679)文中，分析了局部敏感哈希方法是如何应用在检索过程中的，以及原始的哈希方法和基于p-stable分布的哈希方法。
原始的哈希方法和基于p-stable分布的哈希方法都是随机产生的，其效果受随机函数的限制并会产生动荡。本文中描述一种有监督学习的哈希方法，根据不同的数据学习到不同的哈希方法，相对于随机产生的方法具有较大的优势。本文介绍的方法的原始论文在[1]，名为KSH，即Kernel-Based Supervised Hashing。KSH方法要点如下：
- Kernel Function- Supervised Information- Code Inner Product
- Objective Function- Greedy Optimization- Spectral Relaxation- Sigmoid Smoothing这些要点共同组合成了KSH方法一整套的使用与训练流程。下面将为大家一一介绍。# 核函数（Kernel Function）## 流程
核哈希的方法继承于[2]，具体操作如下：首先，从数据中取m个点，称之为锚点（anchor point），m是KSH的重要参数之一。
对一个点x来说，需要先计算x与锚点的核函数值，得到m维向量，如下面公式，其中下标为样本标记。核函数的选择也是KSH需要控制的参数。
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-1.png)
然后，在使用一个m维向量a与上述向量求内积，在减去偏差b，a和b都是KSH的参数。
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-2.png) 上述公式得到一个实数，可以根据该数的符合，将其二值化。
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-3.png) 这样，就将数据转化成了一个汉明码。当有r组(a,b)参数时，就可以将数据转化为r位汉明码。
为了保证学习到的汉明编码中保存的信息量最大，需要保证：
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-4.png) 于是，b应该等于f(x)公式中的第一项的和的中值。
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-5.png) 将b代入f(x)，得到：
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-6.png) ## 分析在上述流程中，a可以用随机的方法产生。但在KSH，要根据标注数据对a进行学习。注意到上述流程中Kernel的作用，对数据进行第一步的处理，这样做的好处是可以降维，比如，原始数据是10000维，但若选择500个锚点，那么，生成的数据就变成了500维，大大降低了需要学习的参数a的数目。# 监督信息（Supervised Information）既然KSH是监督学习算法，那么需要标注信息，在KSH算法中，其标注信息是一个矩阵S。
KSH的标注信息可以如下得到，从样本集中选取L个样本，然后形成一个L*L的矩阵，矩阵中(i,j)处的值表示样本i和样本j是否相似。这种信息实质上是pair信息，即pair对中两个样本是否是相似。
# 内积法计算相似度（Code Inner Product）假设已经学习到了参数a，那么可以得到汉明编码，如果汉明编码一致，说明两个样本是相似样本，否则不是。汉明编码的相似度的计算是异或计算，但是，在学习过程中，我们要以汉明编码相似度去反推参数a，使用异或计算很难求导。于是，需要将异或运算进行转化。转化方法为：> 将汉明编码中的0值换成-1，两个汉明编码的内积与原来的相似度就有了一一对应的关系，推导如下：
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-7.png) 其中， code函数样本转换为（1/-1）的汉明码，D函数求取汉明码的汉明距离。由于内积的范围在[-r,r]，为了将其归一化到[-1,1]，需要再让内积除以r。如下图所示：
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-8.png) # 目标函数（Objective Function）目标函数如下：
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-9.png) 其中，H为L*r的矩阵，L为监督信息中样本的数目，r为汉明编码的位数。S为监督信息矩阵。将目标函数展开，得到：
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-10.png) 其中，K为L×m的矩阵，m为锚点数目，代表着做完核函数处理的样本数据，A为m×r的矩阵，即参数a。# 贪心算法求解（Greedy Optimization）将目标函数再度展开：
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-11.png) 直接求解难度大大滴，因而，论文提出了一种贪心算法，求取较优解。贪心的方式就是逐位求解，首先求a1，然后a2，直到ak。
为了逐位求解，首先需要定义剩余矩阵，即目标函数中的第二项S，再求完一位后会如何变化。如下
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-12.png)显然，R0=rS。其中a*是已经求完的参数。那么，单个求解的目标函数为：
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-13.png) 在第一步等式中，第一项永远等于L的平方，R也不随着a的变化而变化，因而，它们都是常量。所以，单步的目标函数就变为：
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-14.png) # 频谱化宽松（Spectral Relaxation）
为了对上述目标函数求解，对目标函数进行放宽。得到：
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-15.png) 其中，需要满足的条件就是从没有去掉sgn函数的结果中引申出来的。该问题是一个标准的求特征向量问题：
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-16.png) 其中，ak就是该问题的对应最大特征值的特征向量。求出该值后，并不将其当做最后的求出的值，而是作为初始值然后使用下面的方法再进行优化。# Sigmoid平滑（Sigmoid Smoothing）上述频谱化宽松似乎宽松的过了头，此处使用一种更加接近sgn函数的方法对其进行平滑处理。
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-17.png) 其中，phi函数就是对sgn函数的模拟，在[-6,6]外几乎完全接近sgn函数。
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-18.png) 添加了平滑处理后，就可以使用梯度下降进行求解了。梯度如下：
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-19.png) # 最终算法	最终算法流程如图：
![img1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n4-20.png) # 参考文献[1]. Liu W, Wang J, Ji R, et al. Supervised hashing with kernels[C]//Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012: 2074-2081.
[2]. Kulis B, Grauman K. Kernelized locality-sensitive hashing[J]. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 2012, 34(6): 1092-1104.
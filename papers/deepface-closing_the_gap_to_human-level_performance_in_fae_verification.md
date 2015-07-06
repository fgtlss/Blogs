# DeepFace--Facebook的人脸识别

连续看了DeepID和FaceNet后，看了更早期的一篇论文，即FB的DeepFace。这篇论文早于DeepID和FaceNet，但其所使用的方法在后面的论文中都有体现，可谓是早期的奠基之作。因而特写博文以记之。

# DeepFace基本框架

人脸识别的基本流程是：

	detect -> aligh -> represent -> classify

## 人脸对齐流程

![img](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/papers/imgs/n9-1.png)

分为如下几步：

a. 人脸检测，使用6个基点
b. 二维剪切，将人脸部分裁剪出来
c. 67个基点，然后Delaunay三角化，在轮廓处添加三角形来避免不连续 
d. 将三角化后的人脸转换成3D形状
e. 三角化后的人脸变为有深度的3D三角网
f. 将三角网做偏转，使人脸的正面朝前。
g. 最后放正的人脸
h. 一个新角度的人脸（在论文中没有用到）

总体上说，这一步的作用就是使用3D模型来将人脸对齐，从而使CNN发挥最大的效果。

## 人脸表示

![img](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/papers/imgs/n9-2.png)

经过3D对齐以后，形成的图像都是152×152的图像，输入到上述网络结构中，该结构的参数如下：

- Conv：32个11×11×3的卷积核
- max-pooling: 3×3， stride=2
- Conv: 16个9×9的卷积核
- Local-Conv: 16个9×9的卷积核，Local的意思是卷积核的参数不共享
- Local-Conv: 16个7×7的卷积核，参数不共享
- Local-Conv: 16个5×5的卷积核，参数不共享
- Fully-connected: 4096维
- Softmax: 4030维

前三层的目的在于提取低层次的特征，比如简单的边和纹理。其中Max-pooling层使得卷积的输出对微小的偏移情况更加鲁棒。但没有用太多的Max-pooling层，因为太多的Max-pooling层会使得网络损失图像信息。

后面三层都是使用参数不共享的卷积核，之所以使用参数不共享，有如下原因：

- 对齐的人脸图片中，不同的区域会有不同的统计特征，卷积的局部稳定性假设并不存在，所以使用相同的卷积核会导致信息的丢失
- 不共享的卷积核并不增加抽取特征时的计算量，而会增加训练时的计算量
- 使用不共享的卷积核，需要训练的参数量大大增加，因而需要很大的数据量，然而这个条件本文刚好满足。

全连接层将上一层的每个单元和本层的所有单元相连，用来捕捉人脸图像不同位置的特征之间的相关性。其中，第7层（4096-d）被用来表示人脸。

全连接层的输出可以用于Softmax的输入，Softmax层用于分类。

## 人脸表示归一化

对于输出的4096-d向量：

- 先每一维进行归一化，即对于结果向量中的每一维，都要除以该维度在整个训练集上的最大值。
- 每个向量进行L2归一化

## 分类

得到表示后，使用了多种方法进行分类：

- 直接算内积
- 加权的卡方距离
- 使用Siamese网络结构

加权卡方距离计算公式如下：

![img](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/papers/imgs/n9-3.png)

其中，加权参数由线性SVM计算得到。

Siamese网络结构是成对进行训练，得到的特征表示再使用如下公式进行计算距离：

![img](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/papers/imgs/n9-4.png)

其中，参数alpha是训练得到。Siamese网络与FaceNet就很像了。

# 实验评估

## 数据集

- Social Face Classification Dataset(SFC): 4.4M张人脸/4030人
- LFW: 13323张人脸/5749人
	- restricted: 只有是/不是的标记
	- unrestricted：其他的训练对也可以拿到
	- unsupervised：不在LFW上训练
- Youtube Face(YTF): 3425videos/1595人

## Training on SFC

- 训练使用的人数不同(1.5K/3.3K/4.4K)
- 训练使用的照片数目不同(10%/20%/50%)
- 使用的网络不同(去掉第三层/去掉第4、5层/去掉第3、4、5层)

![img](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/papers/imgs/n9-6.png)

## Results on LFW

![img](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/papers/imgs/n9-7.png)

## Results on YTF

![img](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/papers/imgs/n9-8.png)

# 总结

DeepFace与之后的方法的最大的不同点在于，DeepFace在训练神经网络前，使用了对齐方法。论文认为神经网络能够work的原因在于一旦人脸经过对齐后，人脸区域的特征就固定在某些像素上了，此时，可以用卷积神经网络来学习特征。

针对同样的问题，DeepID和FaceNet并没有对齐，DeepID的解决方案是将一个人脸切成很多部分，每个部分都训练一个模型，然后模型聚合。FaceNet则是没有考虑这一点，直接以数据量大和特殊的目标函数取胜。

在DeepFace论文中，只使用CNN提取到的特征，这点倒是开后面之先河，后面的DeepID、FaceNet全都是使用CNN提取特征了，再也不谈LBP了。


# 参考文献

[1]. Taigman Y, Yang M, Ranzato M A, et al. Deepface: Closing the gap to human-level performance in face verification[C]//Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on. IEEE, 2014: 1701-1708.
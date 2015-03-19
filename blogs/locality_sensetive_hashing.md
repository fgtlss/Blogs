# 局部敏感哈希在检索技术中，索引一直需要研究的核心技术。当下，索引技术主要分为三类：基于树的索引技术（tree-based index）、基于哈希的索引技术（hashing-based index）与基于词的倒排索引（visual words based inverted index）[1]。本文主要对哈希索引技术进行介绍。
## 哈希技术概述在检索中，需要解决的问题是给定一个查询样本query，返回与此query相似的样本，线性搜索耗时耗力，不能承担此等重任，要想快速找到结果，必须有一种方法可以将搜索空间控制到一个可以接受的范围，哈希在检索中就是承担这样的任务，因而，这些哈希方法一般都是局部敏感（Locality-sensitive）的，即样本越相似，经过哈希后的值越有可能一样。所以，本文中介绍的技术都是局部敏感哈希（Locality Sensitive Hashing，LSH），与hashmap、hashtable等数据结构中的哈希函数有所不同。## 哈希技术分类
![Application_of_LSH](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n3-1.png) <center style='font-size:16px'>图 1 LSH分层法使用</center>
对于哈希技术，可以按照不同的维度对齐进行划分。
按照其在检索技术中的应用方法来划分，可以分为分层法和哈希码法：	
+ 分层法即为在数据查询过程中使用哈希技术在中间添加一层，将数据划分到桶中；在查询时，先对query计算桶标号，找到与query处于同一个桶的所有样本，然后按照样本之间的相似度计算方法（比如欧氏距离、余弦距离等）使用原始数据计算相似度，按照相似度的顺序返回结果，在该方法中，通常是一组或一个哈希函数形成一个表，表内有若干个桶，可以使用多个表来提高查询的准确率，但通常这是以时间为代价的。分层法的示意图如图1所示。在图1中，H1、H2等代表哈希表，g1、g2等代表哈希映射函数。
  分层法的代表算法为E2LSH[2]。+ 哈希码法则是使用哈希码来代替原始数据进行存储，在分层法中，原始数据仍然需要以在第二层被用来计算相似度，而哈希码法不需要，它使用LSH函数直接将原始数据转换为哈希码，在计算相似度的时候使用hamming距离来衡量。转换为哈希码之后的相似度计算非常之快，比如，可以使用64bit整数来存储哈希码，计算相似度只要使用同或操作就可以得到，唰唰唰，非常之快，忍不住用拟声词来表达我对这种速度的难言之喜，还望各位读者海涵。  
  哈希码法的代表算法有很多，比如KLSH[3]、Semantic Hashing[4]、KSH[5]等。
  以我看来，两者的区别在于如下几点：
+ 在对哈希函数的要求上，哈希码方法对哈希函数的要求更高，因为在分层法中，即便哈希没有计算的精确，后面还有原始数据直接计算相似度来保底，得到的结果总不会太差，而哈希码没有后备保底的，胜则胜败则败。
+ 在查询的时间复杂度上，分层法的时间复杂度主要在找到桶后的样本原始数据之间的相似度计算，而哈希码则主要在query的哈希码与所有样本的哈希码之间的hamming距离的相似计算。哈希码法没有太多其他的需要，但分层法中的各个桶之间相对较均衡方能使复杂度降到最低。按照我的经验，在100W的5000维数据中，KSH比E2LSH要快一个数量级。
+ 在哈希函数的使用上，两者使用的哈希函数往往可以互通，E2LSH使用的p-stable LSH函数可以用到哈希码方法上，而KSH等哈希方法也可以用到分层法上去。上述的区别分析是我自己分析的，如果有其他意见欢迎讨论。
按照哈希函数来划分，可以分为无监督和有监督两种：	
+ 无监督，哈希函数是基于某种概率理论的，可以达到局部敏感效果。如E2LSH等。+ 有监督，哈希函数是从数据中学习出来的，如KSH、Semantic Hashing等。一般来说，有监督算法比无监督算法更加精确，因而也更常用于哈希码法中。本文中，主要对无监督的哈希算法进行介绍。## Origin LSH
最原始的LSH算法是1999年提出来的[6]。在本文中称之为Origin LSH。### Embedding
Origin LSH在哈希之前，首先要先将数据从L1准则下的欧几里得空间嵌入到Hamming空间。在做此embedding时，有一个假设就是原始点在L1准则下的效果与在L2准则下的效果相差不大，即欧氏距离和曼哈顿距离的差别不大，因为L2准则下的欧几里得空间没有直接的方法嵌入到hamming空间。
	Embedding算法如下：
> + 找到所有点的所有坐标值中的最大值C；+ 对于一个点P来说，P=(x<sub>1</sub>,x<sub>2</sub>,…,x<sub>d</sub>)，d是数据的维度；+ 将每一维x<sub>i</sub>转换为一个长度为C的0/1序列，其中序列的前x<sub>i</sub>个值为1，剩余的为0.+ 然后将d个长度为C的序列连接起来，形成一个长度为Cd的序列.
这就是embedding方法。注意，在实际运算过程中，通过一些策略可以无需将embedding值预先计算出来。### Algorithm of Origin LSH
在Origin LSH中，每个哈希函数的定义如下：
![origin_hash](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n3-12.png)

即输入是一个01序列，输出是该序列中的某一位上的值。于是，hash函数簇内就有Cd个函数。在将数据映射到桶中时，选择k个上述哈希函数，组成一个哈希映射，如下：
![k_hash_funcs](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n3-2.png)
	再细化，LSH的算法步骤如下：

> + 从[0,Cd]内取L个数，形成集合G，即组成了一个桶哈希函数g。+ 对于一个向量P，得到一个L维哈希值，即P<sub>|G</sub>，其中L维中的每一维都对应一个哈希函数h。+ 由于直接以上步中得到的L维哈希值做桶标号不方便，因而再进行第二步哈希，第二步哈希就是普通的哈希，将一个向量映射成一个实数。
![hash_to_bucket_id](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n3-3.png)

  其中，a是从[0,M-1]中随机抽选的数字。这样，就把一个向量映射到一个桶中了。## LSH based on p-stable distribution
该方法由[2]提出，E2LSH[7]是它的一种实现。
### p-stable分布
> 定义：对于一个实数集R上的分布D，如果存在P>=0，对任何n个实数v1,…,vn和n个满足D分布的变量X1,…,Xn，随机变量Σ<sub>i</sub>v<sub>i</sub>X<sub>i</sub>和(∑<sub>i</sub>|v<sub>i</sub>|<sup>p</sup>)<sup>(1/p)</sup>X有相同的分布，其中X是服从D分布的一个随机变量，则称D为一个p稳定分布。
对任何p∈(0,2]存在稳定分布。P=1时是柯西分布；p=2时是高斯分布。
当p=2时，两个向量v1和v2的映射距离a·v1-a·v2和||v1-v2||<sub>p</sub>X的分布是一样的，此时对应的距离计算方式为欧式距离。
利用p-stable分布可以有效的近似高维特征向量，并在保证度量距离的同时，对高维特征向量进行降维，其关键思想是，产生一个d维的随机向量a，随机向量a中的每一维随机的、独立的从p-stable分布中产生。对于一个d维的特征向量v，如定义，随机变量a·v具有和(∑<sub>i</sub>|v<sub>i</sub>|<sup>p</sup>)<sup>(1/p)</sup>X一样的分布，因此可以用a·v表示向量v来估算||v||<sub>p</sub>。
### E2LSH
基于p-stable分布，并以‘哈希技术分类’中的分层法为使用方法，就产生了E2LSH算法。E2LSH中的哈希函数定义如下：![k_hash_funcs](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n3-4.png)
其中，v为d维原始数据，a为随机变量，由正态分布产生; w为宽度值，因为a∙v+b得到的是一个实数，如果不加以处理，那么起不到桶的效果，w是E2LSH中最重要的参数，调得过大，数据就被划分到一个桶中去了，过小就起不到局部敏感的效果。b使用均匀分布随机产生，均匀分布的范围在[0,w]。
与Origin LSH类似，选取k个上述的哈希函数，组成一个哈希映射，效果如图2所示：
![k_hash_funcs](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n3-5.png)
 <center style='font-size:16px'>图 2 E2LSH映射</center>
但是这样，得到的结果是(N<sub>1</sub>,N<sub>2</sub>,…,N<sub>k</sub>)，其中N<sub>1</sub>,N<sub>2</sub>,…,N<sub>k</sub>在整数域而不是只有0,1两个值，这样的k元组就代表一个桶。但将k元组直接当做桶标号存入哈希表，占用内存且不便于查找，为了方便存储，设计者又将其分层，使用数组+链表的方式，如图3所示：

![data_structure_of_e2lsh](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n3-6.png)
<center style='font-size:16px'>图 3 E2LSH为存储桶标号而产生的数组+链表二层结构</center>
对每个形式为k元组的桶标号，使用如下h1函数和h2函数计算得到两个值，其中h1的结果是数组中的位置，数组的大小也相当于哈希表的大小，h2的结果值作为k元组的代表，链接到对应数组的h1位置上的链表中。在下面的公式中,r<sup>'</sup>为[0,prime-1]中依据均匀分布随机产生。
![h1](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n3-9.png)
![h2](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n3-10.png) 	经过上述组织后，查询过程如下：> + 对于查询点query，+ 使用k个哈希函数计算桶标号的k元组；+ 对k元组计算h1和h2值，+ 获取哈希表的h1位置的链表，+ 在链表中查找h2值，+ 获取h2值位置上存储的样本+ Query与上述样本计算精确的相似度，并排序+ 按照顺序返回结果。
E2LSH方法存在两方面的不足[8]：首先是典型的基于概率模型生成索引编码的结果并不稳定。虽然编码位数增加，但是查询准确率的提高确十分缓慢；其次是需要大量的存储空间，不适合于大规模数据的索引。E2LSH方法的目标是保证查询结果的准确率和查全率，并不关注索引结构需要的存储空间的大小。E2LSH使用多个索引空间以及多次哈希表查询，生成的索引文件的大小是原始数据大小的数十倍甚至数百倍。
### Hashcode of p-stable distribution
E2LSH可以说是分层法基于p-stable distribution的应用。另一种当然是转换成hashcode，则定义哈希函数如下：
![h2](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n3-11.png)
其中，a和v都是d维向量，a由正态分布产生。同上，选择k个上述的哈希函数，得到一个k位的hamming码，按照"哈希技术分类"中描述的技术即可使用该算法。## Reference[1]. Ai L, Yu J, He Y, et al. High-dimensional indexing technologies for large scale content-based image retrieval: a review[J]. Journal of Zhejiang University SCIENCE C, 2013, 14(7): 505-520.[2]. Datar M, Immorlica N, Indyk P, et al. Locality-sensitive hashing scheme based on p-stable distributions[C]//Proceedings of the twentieth annual symposium on Computational geometry. ACM, 2004: 253-262.	
[3]. Kulis B, Grauman K. Kernelized locality-sensitive hashing[J]. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 2012, 34(6): 1092-1104.	
[4]. Salakhutdinov R, Hinton G. Semantic hashing[J]. International Journal of Approximate Reasoning, 2009, 50(7): 969-978.	
[5]. Liu W, Wang J, Ji R, et al. Supervised hashing with kernels[C]//Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012: 2074-2081.[6]. Gionis A, Indyk P, Motwani R. Similarity search in high dimensions via hashing[C]//VLDB. 1999, 99: 518-529.	
[7]. http://web.mit.edu/andoni/www/LSH/[8]. http://blog.csdn.net/jasonding1354/article/details/38237353
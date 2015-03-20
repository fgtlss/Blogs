# 机器学习算法概览


机器学习（Machine Learning， ML）是什么，作为一个MLer，经常难以向大家解释何为ML。久而久之，发现要理解或解释机器学习是什么，可以从机器学习可以解决的问题这个角度来说。对于MLers，理解ML解决的问题的类型也有助于我们更好的准备数据和选择算法。

# 十个机器学习问题样例

想入门机器学习的同学，经常会去看一些入门书，比如《集体智慧编程》、《机器学习实战》、《数据挖掘》、《推荐系统实践》等。看书的过程中，经常性的会看到如下样例：

+ 垃圾邮件识别
+ 信用卡交易异常检测
+ 手写数字识别
+ 语音识别
+ 人脸检测
+ 商品推荐
+ 疾病检测（根据以往病例记录，确定病人是否患病）
+ 股票预测
+ 用户分类（根据用户行为判断该用户是否会转化为付费用户）
+ 形状检测（根据用户在手写板上上画得形状，确定用户画的到底是什么形状）

因此，当再有人问ML是什么的时候，就可以说这个是ML可以handle的，这个问题ML也可以handle，blahblah。

# 机器学习问题类型

对问题进行分类，好处就在于可以更好的把握问题的本质，更好的知道什么类型的算法需要用到。

一般有四大类型：

+ 分类（classification）：有一些已经标注好类别的数据，在标注好的数据上建模，对于新样本，判断它的类别。如垃圾邮件识别
+ 回归（regression）：有一些已经标注好的数据，标注值与分类问题不同，分类问题的标注是离散值，而回归问题中的标注是实数，在标注好的数据上建模，对于新样本，得到它的标注值。如股票预测。
+ 聚类（clustering）：数据没有被标注，但是给出了一些相似度衡量标准，可以根据这些标准将数据进行划分。如在一堆未给出名字的照片中，自动的将同一个人的照片聚集到一块。
+ 规则抽取（rule extraction）：发现数据中属性之间的统计关系，而不只是预测一些事情。如啤酒和尿布。

# 机器学习算法

知道了机器学习要解决的问题后，就可以思考针对某一个问题，需要采集的数据的类型和可以使用的机器学习算法，机器学习发展到今天，诞生了很多算法，在实际应用中往往问题在于算法的选择，在本文中，使用两种标准对算法进行分类，即学习方式和算法之间的相似性。

## 学习方式（Learning Style）

在ML中，只有几个主流的学习方式，在下面的介绍中，使用一些算法和问题的样例来对这些方式进行解释说明。按照学习方式对机器学习算法进行分类可以使我们更多的思考输入数据在算法中的角色和使用模型前需要的准备工作，对我们选择最适合的模型有很好的指导作用。

+ 监督学习（supervised learning）：输入数据都有一个类别标记或结果标记，被称作训练数据，比如垃圾邮件与非垃圾邮件、某时间点的股票价格。模型由训练过程得到，利用模型，可以对新样本做出推测，并可以计算得到这些预测的精确度等指标。训练过程往往需要在训练集上达到一定程度的精确度，不欠拟合或过拟合。监督学习一般解决的问题是分类和回归，代表算法有逻辑斯底回归（Logistic Regression）和神经网络后向传播算法（Back Propagation Neural Network）。

+ 无监督学习（Unsupervised Learning）：输入数据没有任何标记，通过推理数据中已有的结构来构建模型。一般解决的问题是规则学习和聚类，代表算法有Apriori算法和k-means算法。

+ 半监督学习（Semi-Supervised Learning）：输入数据是标注数据和非标注数据的混合，它也是为了解决预测问题的，但是模型必须同时兼顾学习数据中已经存在的结构和作出预测，即上述监督学习和无监督学习的融合。该方法要解决的问题仍然是分类的回归，代表算法一般是在监督学习的算法上进行扩展，使之可以对未标注数据建模。

+ 增强学习（Reinforcement Learning）：在这种学习方式中，模型先被构建，然后输入数据刺激模型，输入数据往往来自于环境中，模型得到的结果称之为反馈，使用反馈对模型进行调整。它与监督学习的区别在于反馈数据更多的来自于环境的反馈而不是由人指定。该方式解决的问题是系统与机器人控制，代表算法是Q-学习（Q-learning）和时序差分算法（Temporal difference learning）。

在商业决策中，一般会使用的方法是监督学习和无监督学习。当下一个热门的话题是半监督学习，比如在图片分类中，有很多数据集都是有少量的标记数据和大量的非标记数据。增强学习更多的用于机器人控制机其他的控制系统中。

## 算法相似度（Algorithm Similarity）

一般会根据模型的模式或者函数模式的相似度来对算法进行划分。比如基于树的方法（tree-based method）与神经网络算法（neural network）。当然，这种方法并不完美，因为很多算法可以很容易的被划分到多个类别中去，比如学习矢量量化算法（Learning Vector Quantization）既是神经网络算法也是基于样例的算法（Instance-based method）。在本文中，可以看到很多不同的分类方法。

### 回归（Regression）

回归是在自变量和需要预测的变量之间构建一个模型，并使用迭代的方法逐渐降低预测值和真实值之间的误差。回归方法是统计机器学习的一种
常用的回归算法如下：

+ Ordinary Least Squares（最小二乘法）
+ Logistic Regression（逻辑斯底回归）
+ Stepwise Regression（逐步回归）
+ Multivariate Adaptive Regression Splines（多元自适应回归样条法）
+ Locally Estimated Scatterplot Smoothing（局部加权散点平滑法）

### 基于样例的方法（Instance-based Methods）

基于样例的方法需要一个样本库，当新样本出现时，在样本库中找到最佳匹配的若干个样本，然后做出推测。基于样例的方法又被成为胜者为王的方法和基于内存的学习，该算法主要关注样本之间相似度的计算方法和存储数据的表示形式。

+ k-Nearest Neighbour (kNN)
+ Learning Vector Quantization (LVQ)
+ Self-Organizing Map (SOM)

### 正则化方法（Regularization Methods）

这是一个对其他方法的延伸（通常是回归方法），这个延伸就是在模型上加上了一个惩罚项，相当于奥卡姆提到，对越简单的模型越有利，有防止过拟合的作用，并且更擅长归纳。我在这里列出它是因为它的流行和强大。

+ Ridge Regression
+ Least Absolute Shrinkage and Selection Operator (LASSO)
+ Elastic Net

### 决策树模型（Decision Tree Learning）

决策树方法建立了一个根据数据中属性的实际值决策的模型。决策树用来解决归纳和回归问题。

+ Classification and Regression Tree (CART)
+ Iterative Dichotomiser 3 (ID3)
+ C4.5
+ Chi-squared Automatic Interaction Detection (CHAID)
+ Decision Stump
+ Random Forest
+ Multivariate Adaptive Regression Splines (MARS)
+ Gradient Boosting Machines (GBM)

### 贝叶斯（Bayesian）

贝叶斯方法是在解决归类和回归问题中应用了贝叶斯定理的方法。

+ Naive Bayes
+ Averaged One-Dependence Estimators (AODE)
+ Bayesian Belief Network (BBN)

### 核方法（Kernel Methods）

核方法中最有名的是Support Vector Machines(支持向量机)。这种方法把输入数据映射到更高维度上，将其变得可分，使得归类和回归问题更容易建模。

+ Support Vector Machines (SVM)
+ Radial Basis Function (RBF)
+ Linear Discriminate Analysis (LDA)

### 聚类（Clustering Methods）

聚类本身就形容了问题和方法。聚类方法通常是由建模方式分类的比如基于中心的聚类和层次聚类。所有的聚类方法都是利用数据的内在结构来组织数据，使得每组内的点有最大的共同性。

+ K-Means
+ Expectation Maximisation (EM)

### 联合规则学习（Association Rule Learning）

联合规则学习是用来对数据间提取规律的方法，通过这些规律可以发现巨量多维空间数据之间的联系，而这些重要的联系可以被组织拿来使用或者盈利。

+ Apriori algorithm
+ Eclat algorithm

### 人工神经网络（Artificial Neural Networks）

受生物神经网络的结构和功能的启发诞生的人工神经网络属于模式匹配一类，经常被用于回归和分类问题，但是它存在上百个算法和变种组成。其中有一些是经典流行的算法（深度学习拿出来单独讲）：

+ Perceptron
+ Back-Propagation
+ Hopfield Network
+ Self-Organizing Map (SOM)
+ Learning Vector Quantization (LVQ)

### 深度学习（Deep Learning）

Deep Learning(深度学习)方法是人工神经网络在当下的一个变种。相比传统的神经网络，它更关注更加复杂的网络构成，许多方法都是关心半监督学习，就是一个大数据集中只有少量标注数据的那种问题。

+ Restricted Boltzmann Machine (RBM)
+ Deep Belief Networks (DBN)
+ Convolutional Network
+ Stacked Auto-encoders

### 降维（Dimensionality Reduction）

与聚类方法类似，对数据中的固有结构进行利用，使用无监督的方法学习一种方式，该方式用更少的信息来对数据做归纳和描述。这对于对数据进行可视化或者简化数据很有用，也有去除噪声的影响，经常采用这种方法使得算法更加高效。

+ Principal Component Analysis (PCA)
+ Partial Least Squares Regression (PLS)
+ Sammon Mapping
+ Multidimensional Scaling (MDS)
+ Projection Pursuit

### 组合方法（Ensemble Methods）

Ensemble methods(组合方法)由许多小的模型组成，这些模型经过独立训练，做出独立的结论，最后汇总起来形成最后的预测。组合方法的研究点集中在使用什么模型以及这些模型怎么被组合起来。

+ Boosting
+ Bootstrapped Aggregation (Bagging)
+ AdaBoost
+ Stacked Generalization (blending)
+ Gradient Boosting Machines (GBM)
+ Random Forest

#	原文及链接：
	
	A Tour of Machine Learning Algorithms
+ http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/
+ http://machinelearningmastery.com/practical-machine-learning-problems/







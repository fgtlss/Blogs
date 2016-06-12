# TensorFlow

> 又是好久没有写博客了，上班以来，感觉时间过得飞快，每天时间很紧，过得有点累，不知道自己的博客能坚持到何时，且行且珍惜。

本片博文是参考文献[1]的阅读笔记，特此声明

TensorFlow，以下简称TF，是Google去年发布的机器学习平台，发布以后由于其速度快，扩展性好，推广速度还是蛮快的。江湖上流传着Google的大战略，Android占领了移动端，TF占领神经网络提供AI服务，未来的趋势恰好是语音图像以及AI的时代，而Google IO上发布的Gbot似乎正是这一交叉领域的初步尝试。

TF的特点之一就是可以支持很多种设备，大到GPU、CPU，小到手机平板，五花八门的设备都可以跑起来TF。不得不说这一点很有前瞻性，可以预见的是，mobile-end的用户将会享受到越来越多的AI服务。说个极端，说不定以后某天，单机版的AlphaGo会出现也是可以的。

话不多说，开始正文。

# Basic Concepts

## 张量(Tensor)
名字就是TensorFlow，直观来看，就是张量的流动。张量(tensor)，即任意维度的数据，一维、二维、三维、四维等数据统称为张量。而张量的流动则是指保持计算节点不变，让数据进行流动。这样的设计是针对连接式的机器学习算法，比如逻辑斯底回归，神经网络等。连接式的机器学习算法可以把算法表达成一张图，张量从图中从前到后走一遍就完成了前向运算；而残差从后往前走一遍，就完成了后向传播。

## 算子(operation)
在TF的实现中，机器学习算法被表达成图，图中的节点是算子(operation)，节点会有0到多个输出，下图是TF实现的一些算子。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_tensorflow/1.png)

每个算子都会有属性，所有的属性都在建立图的时候被确定下来，比如，最常用的属性是为了支持多态，比如加法算子既能支持float32，又能支持int32计算。

## 核(kernel)
TF中还有一个概念是kernel，kernel是operation在某种设备上的具体实现。TF的库通过注册机制来定义op和kernel，所以可以通过链接一个其他的库来进行kernel和op的扩展。

## 边(edge)
TF的图中的边分为两种：

- 正常边，正常边上可以流动数据，即正常边就是tensor
- 特殊边，又称作控制依赖，(control dependencies)
	- 没有数据从特殊边上流动，但是特殊边却可以控制节点之间的依赖关系，在特殊边的起始节点完成运算之前，特殊边的结束节点不会被执行。
	- 也不仅仅非得有依赖关系才可以用特殊边，还可以有其他用法，比如为了控制内存的时候，可以让两个实际上并没有前后依赖关系的运算分开执行。
	- 特殊边可以在client端被直接使用

## 会话(Session)

客户端使用会话来和TF系统交互，一般的模式是，建立会话，此时会生成一张空图；在会话中添加节点和边，形成一张图，然后执行。	

下图有一个TF的会话样例和所对应的图示。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_tensorflow/2.png)

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_tensorflow/3.png)

## 变量(Variables)

机器学习算法都会有参数，而参数的状态是需要保存的。而参数是在图中有其固定的位置的，不能像普通数据那样正常流动。因而，TF中将Variables实现为一个特殊的算子，该算子会返回它所保存的可变tensor的句柄。

# Implementation

首先是实现中的几个部分：

- TF中最重要的Tensor被支持的非常全面，8bit到64bit， signed和unsigned，IEEE float/double，complex number等等。使用引用计数来保存tensor，当计数到0时，tensor被回收。
- 客户端，用户会使用；与master和一些worker process交流
- master，用来与客户端交互，同时调度任务；
- worker process，工作节点，每个worker process可以访问一到多个device。
- device，TF的计算核心，通过将device的类型、job名称、在worker process中的索引将device命名。可以通过注册机制来添加新的device实现，每个device实现需要负责内存分配和管理调度TF系统所下达的核运算需求。

TF的实现分为了单机实现和分布式实现，在分布式实现中，需要实现的是对client，master，worker process不在同一台机器上时的支持。此时，关于这些进程的调度，使用的是原始论文中参考文献51的调度方式。关于分布式和单机的不同如下图所示：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_tensorflow/4.png)

## Single-Device Execution

构建好图后，使用拓扑算法来决定执行哪一个节点，即对每个节点使用一个计数，值表示所依赖的未完成的节点数目，当一个节点的运算完成时，将依赖该节点的所有节点的计数减一。如果节点的计数为0，将其放入准备队列待执行

## Multi-Device Execution

当系统到了分布式情况下时，事情就变得复杂了很多，还好前述调度用了现有的框架。那么对于TF来说，剩下的事情就是：

- 决定运算在哪个设备上运行
- 管理设备之间的数据传递

### 决定设备

使用一个cost model算法来进行预估时间，计算后使用贪心算法来分配设备。在决定设备的时候，也可以预先设置一些约束，比如，某个op只能在GPU上执行等。 

预估时间有两种方法：

- 使用启发式的算法，通过把输入和输出的类型以及tensor的大小输入进去，得到时间的预估
- 使用模拟的方法，对图的计算进行一个模拟，得到各个计算在其可用的设备上的时间。

> 寻找合适设备是TF系统区分与之前很多系统的地方，之前的系统比如Parameter Server，是参数分离出来，运算在一起，同时使用数据切分来达到分布式。而TF是把每个op都映射到某个机器上，意味着每个op可能在不同的机器上，这是对系统的进一步剖离，因而可以达到更高的可扩展性。


### 跨设备通信

当两个需要通信的op在不同的机器上时，就需要跨设备通信，当它们需要通信时，TF会在它们之间的联系中添加Send和Recv节点，通过Send和Recv之间进行通信来达到op之间通信的效果。如下所示：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_tensorflow/5.png)

为了优化网络通信，TF会将相同的数据传送合并，如a->b和a->c的传送合并，这一点可以通过Send和Recv很方便的实现。而通过实现Send和Recv，将master节点的通信调度任务解放出来，master就只需要向图中的各个节点发出运行命令就够了，增加了系统的可扩展性。

Send和Recv通过TCP或RDMA来传输数据

### 错误处理

在分布式系统中，常见的错误来自于两个方面：

- Send/Recv的网络传输
- master和worker process之间的心跳同步

当错误发生的时候，TF会将整个图的计算停止，并从上一次保存的状态重新执行。为了保存状态，每个Variable的节点都去连接一个Save的节点。这些save节点会每隔一段时间或每隔几次迭代运行一次。

> 自从TF将op剖离之后，所有的策略都依赖于节点来实现，Variable利用节点实现，状态保存也用节点实现。感觉还是很不一样的。
> 一个节点出了错误，要停掉整个图的计算，我觉得这样的恢复模式会不会代价太大？


# Extensions

## Gradient Computation(梯度计算)

连接式的机器学习算法往往需要使用梯度下降法来求取参数，TF通过扩展图的方式实现了自动求导，TF做法如下：

对于每张计算图，得到从输入I到输出C的路径，并从C到I回溯，回溯过程中对于路径上的每个节点A，添加另一个节点来计算A'来计算偏导，在计算偏导的过程中，A'不仅仅将上一层传下来的反向导数作为输入，还可能将A的输入和输出也作为其输入。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_tensorflow/6.png)

在执行前向计算的时候，启发式的优化算法通过观察图中的节点的计算顺序，来决定哪种操作放在哪个节点上，从而帮助用户来内存重用；当启发式的算法无效的时候，用户还可以通过添加控制依赖来自行实现内存上的优化。

而当反向传播加入的时候，事情变得有点复杂，在正向计算中较前位置的计算数据在反向传播的后期会被经常用到。这就需要把这些数据存在内存中，从而整个图的内存都将被占用，使得本来就少的GPU内存更加的捉襟见肘。

有三种方法来对其进行优化：

- 更加复杂的启发式算法来决定图的计算顺序
- 重新计算这些向量而不是保存下来
- 将长期在GPU内存中的tensor转移到CPU内存中

## Partial Execution(局部执行)

TF支持部分执行，对于多输出的TF图，可能用户只想获取一个输出，此时可以指定需要的输入(feed)和输出(fetch)值。然后TF会自动计算该输出的依赖，然后只计算必要的部分。

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_tensorflow/7.png)

如上图所示，指定b为输入，f为输出，那么此时d、e是不会被计算的。


## Control Flow(控制流)

虽然TF的图已经有了很强的表达能力，但还不够，还需要控制流的表达，比如已经实现的Switch、Merge、Enter、Leave和NextIteration等一系列控制算子。

TF使用和MIT Token-Tagged machine相似的表示系统，将循环的每次迭代标记为一个tag，迭代的执行状态标记为一个frame，但迭代所需的数据准备好的时候，就可以开始计算，从而多个迭代可以同时执行。

对于分布式系统来说，控制循环的状态是一个大问题。而TF使用图重写的方式来实现它，在图切分的时候，添加一个小的状态机来监控迭代的开始和结束，

而对于有梯度计算的图来说，在有控制流的情况下，需要记录各种状态，比如对于if算子，需要记录哪个分支被运行了；而对于循环，需要记录循环了几次。TF仍然使用图重写来实现记录状态的功能，细节不赘述了。

## Input Operations(输入操作)

为Input也构建了一个Node，来管理数据从硬盘到内存的过程。往往需要提前将数据读入进来以减少内存瓶颈。

## Queues(队列)

TF实现了一个队列来支持异步操作，EnQueue可以阻塞直到队列中的空间足够；DeQueue也可以阻塞直到队列中一系列的要求得到满足。

队列有两个典型应用：

- 读入数据，数据在队列中，这样可以达到数据处理和数据载入的并行
- 梯度的累加，让梯度存储在队列中，直到队列中的梯度积累到一定的数值，这样可以达到多个mini-batch组成一个大的batch
- 句子的聚合，对LSTM中的输入句子按长度来进行聚合，统一计算以提高效率。

除了FIFO队列外，TF还实现了一个shuffle队列。

## Containers(容器)

普通的Cotainer可以长期的存储可变状态，但Container不止于此，用Container，甚至不同的会话中的图之间也可以通过Container来共享状态。

# Optimizaiton

TF给了用户以极其易用的接口，这就需要底层来自动的做很多优化。

## Common Subexpression Elimination

用户给出的图定义中可能会存在重复的计算操作，TF使用Click（原始论文参考文献12）中的算法来进行图的重复表达式的删减

## Controlling Data Combination and Memory Usage

对于复杂的网络模型，GPU是必须的；而对于GPU来说，它的内存是不足的，因而要用良好的管理来提高GPU内存的使用效率。

在这一点上，TF主要关注数据的网络传输，这主要集中在Recv节点何时去远程读取数据，TF会自动分析图上的关键路径，通过设置依赖的方式来使得非关键路径上的数据传输如何不影响关键路径。

## Asynchronous Kernels

异步核在执行后立即返回，同时会执行一个回调函数。这样，可以防止等待计算完成的同时眼看着没有做的IO任务也不做。即异步核也可以提升并行能力。异步核的典型样例就是Recv节点和Enqueue和Dequeue操作。

## Optimized Libraries for Kernel Implementations

对于已经存在的线性代数库自然是要利用的，但TF团队对一些库还扩展了其对任意维度的tensor的支持。

常见的线性计算库包括：

- BLAS、cuBLAS，在很多设备上都优化了矩阵乘法
- cuda-convnet、CuDNN，在GPU上优化

## Lossy Compression

在数据传输过程中，为了加快传输效率，往往需要对精度进行压缩。在TF中，传输之前将32bit的float转变为16float，在传输完之后再转回来，转回来时用0补齐。

# Common Programming Idioms

上面讲的都是一些系统级别的优化，还有一些机器学习算法所用到的技巧。这里假定用户都用SGD来求解机器学习算法。

## Data Parallel Training

通过数据并行的方式来提升SGD的效率，比如，假如每次SGD的mini-batch是1000个样本，那么，切成10份，每份100个，然后将模型复制10份，每份都将梯度传到参数服务器上。

数据切分也分为同步和异步两种方式，同步的切分是等待每个独立的model传上来的梯度都到齐后再进行更新，这样和一个大的batch没有区别。异步的切分则是不用等待，每个独立的模型的参数更新直接更新。

如下图所示：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_tensorflow/8.png)

## Model Parallel Training

还可以对模型进行切分，让模型的不同部分执行在不同的设备上，这样可以一个迭代的样本可以在不同的设备上同时执行。如下图所示的LSTM模型：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_tensorflow/9.png)


## Concurrent Steps for Model Computation PipeLine

为了充分利用同一台设备的计算能力，TF会尽量让相邻的计算在同一台设备上，这样可以节省网络开销，比如对于模型并行来说，如果放在同一个设备上，如下图所示：

![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_tensorflow/10.png)

> 我个人觉得这是TF区分与Parameter Server的一个大区别，对于TF来说，计算节点是分离的，参数也是分离的，因而其PS也是分离的。每个设备上可能分配了计算节点，然后其对应的ps也在该设备上。因而，传统的PS中，计算和参数是分开的，但计算和参数他们分别是在一起的；而TF中，计算本身是分离的，参数也是分离的，而计算和参数是在一起的。

# Related Work

TF出现之前，已经有很多的类似的平台了。

- Theano
- Torch
- Caffe
- Chainer
- Computational Network

可以这么说，TF从他们每一个中都吸取了一些feature和设计思想。根据进化论的观点，我不能说TF要优于他们，而要说，TF更能适应当前的需求。在之前写Parameter Server的时候我就隐隐的感觉到，一种设计的产生是和它的需求紧紧相关的，TF的设计可能会有很多人想到，但TF却只能由google设计和实现，因为需求。而TF的产生也是google大一统移动和PC和Server的战略需求。

> TF的易用性、跨平台能力是其功能亮点，而其可扩展性和高效性则是其根基。不知TF一出，下一代的平台会是什么样子？ 还是说平台的演化是否到此就像android和iOS那样已经比较成熟了？

> 而我更感兴趣的其实是，对于这样一个大的平台，大哥是怎么调试的呢？ 恐怕这是程序员的能力所在。扯远了。

TF与其他平台的区别于联系：

- 支持符号推导，如Theano和Chainer
- 使用C++写内核，从而跨平台部署很方便，如Caffe，
- 支持跨设备计算，让用户高层表达模型，如Adam和Distbelief，但比Adam和DistBelief优越的是，更有弹性，支持更多的模型。
- 相对于Adam、DistBelief和Parameter Server，TF把参数节点化，更新操作也是图中的一个节点。而Adam等三个会有一个独立的Parameter Server。
- Halide拥有和TF相似的中间表达但却有更高级的语义表示，在并行方面优化的更多，但却是单机的，TF有意向向此方向发展，将Halide的东西添加到TF中来。

其他还有很多像TF那样支持数据流图的分布式平台，比如：

- Dryad、Flume、CIEL（数据流调度算法从此借鉴而来）
- Naind（分布式架构非常像）
- Spark、Naind（在内存足够的情况下，TF和他们一样运行的很好）
- Dandelion（跨设备）
- TF的迭代是混合的方法：多个数据流但却一次执行，同时共享状态。多个数据流通过变量来共享，或使用图中的同步机制，比如队列。TF还支持图内的迭代，这是CIEL和Naiad的混合，简单来说，像CIEL一样每个节点只有当准备好之后才会启动；但为了效率整张图则表示为一个静态的循环的图，像Naiad一样。

> TF内的迭代似乎是很重要的一个点，但论文中含糊不清，想必机器学习算法中的控制流都是很纠结的。








	









# Reference

[1]. Abadi M, Agarwal A, Barham P, et al. Tensorflow: Large-scale machine learning on heterogeneous distributed systems[J]. arXiv preprint arXiv:1603.04467, 2016.
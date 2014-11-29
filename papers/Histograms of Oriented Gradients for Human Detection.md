# Histograms of Oriented Gradients for Human Detection
> Hog算子进行行人检测，是05年在CVPR上发表的经典文章。使用Hog算子提取特征，然后使用SVM来进行分类。## 1. Definition> Hog Descriptor：locally normalized histogram of gradient orientation in dense overlapping grids, 即局部归一化梯度方向直方图。
## 2. Important Properties> 1. Fine-scale gradients，较好尺度的梯度计算> 2. Fine orientation binning，较好的方向分区> 3. Relatively coarse spatial binning，相对粗粒度的空间分区> 4. High-quality local contrast normalization in overlapping descriptor blocks，在重叠块中的高质量局部对比度归一化
## 3. Algorithms	Default detector properties：
+ RGB colour space with no gamma correction.+ [-1,0,1] gradient filter with no smoothing.+ Linear gradient voting into 9 orientation bins in 0-180+ 16×16 pixel blocks of four 8×8 pixel cells.+ Gaussian spatial window with variance=8.+ L2-Hys(Lowe-style clipped L2 norm) block normalization.+ Block spacing stride of 8 pixels(hence 4-fold coverage of each cell).+ 64×128 detection window+ Linear SVM classifier
### 3.1 Process
![process](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/2-1.png) > 基本概念：
+ Cell：统计梯度直方图的最小单元，论文中为8×8。+ Blocks：做直方图归一化的单元，论文中为2×2个cell，即block大小为16×16.> Hog+SVM算法的过程分为如下几个步骤：
+ 使用gamma变换对图片进行归一化。+ 计算每个像素点的梯度方向。+ 在Cell中对梯度方向进行统计，得到直方图。+ 在Block中对Cell的梯度直方图进行归一化，Block以窗口滑过每个Cell，Block可重叠。+ 将每个block的直方图串联起来，形成整幅图片的特征向量。+ 这里需要注意，每个Cell可能作为多个Block的子部分被归一化放到整幅图片的特征向量中。+ 使用Linear SVM算法对特征向量进行分类，得到最终模型。
### 3.2 Gamma / Colour Normalization
+ Pixel representation: grayscale, RGB, LAB.+ Optionally with power law (gamma) equalization or log compression+ Best: LAB and RGB with square root gamma compression
### 3.3 Gradient Computation
	Compare on schemes below:+ Various 1-D point derivatives	+ uncentred [-1,1]	+ centred [-1,0,1]	+ cubic-corrected [1,-8,0,8,-1]).	+ Those with Gaussian derivatives.+ 3×3 sobel masks.+ 2×2 diagonal ones (the most compact centred 2-D derivative masks)
> Simplest scheme turns out to be the best. Uncentred [-1,0,1] without Gaussian smooth work best. Larger mask and smoothing damages the performance significantly.
> For colour images, calculate separate gradients for each channel, take the one with largest norm as the pixel’s gradient vector.
### 3.4 Spatial / Orientation Binning
	对每个Cell统计得到直方图，步骤如下：
+ Calculate a weighted vote for each pixel based on the orientation of the gradient element centred on it.	+ Vote weight is function of the gradient magnitude at the pixel		+ Magnitude itself		+ Its square		+ Its square root	+ Magnitude itself gives the best result.+ Accumulated into orientation bins over local spatial regions that called cells	+ Orientation bins have two kinds:		+ 0-180: unsigned gradient		+ 0-360: signed gradient	+ Unsigned gradient is better, for human’s wide range of clothing and background colour make the the signs of contrasts uninformative.		+ Include the signs information maybe helpful in some other object recognition task like cars, motobikes.	+ Number of orientation bins get to the best at 9 using unsigned gradient.	+ To reduce aliasing, votes are interpolated bilinearly between the neighbouring bin centers in both orientation and position.
	### 3.5 Normalization and Descriptor Blocks
+ Gradient strengths vary over a wide range owing to local variations in illumination and fore-background contrast, so local contrast normalization is essential.+ Grouping cells into larger spatial blocks and contrast normalization each block separately.	+ An alternative center-surround style cell normalization scheme is also investigated.		+ 以某cell为中心，使用高斯为周围的cell加权，用该cell和周边cell得到一个总值，以此总值归一化。这样，在最终的结果中，一个cell的权重只出现一次。也因此效果下降。+ Overlapping of the blocks seems redundant but improves the performance significantly.+ 论文中使用了两种算子：R-HOG和C-HOG。+ Vertical cell (2×1) and horizontal cell (1×2) are also considered.+ It’s useful to down-weight pixels near the edges of the blocks by applying a Gaussian spatial window to each pixel before accumulating orientation votes into cells.
### 3.6 Block Normalization schemes	把多个Cell组合成一个Block后，就形成一个Block向量v，对于v，有如下几种方法做归一化：+ L2-norm+ L2-Hys+ L1-norm+ L1-sqrt
## 4. Experiment	共进行如下几组实验:
+ Compare with previous algorithm	+ Generalized Haar Wavelets	+ PCA-Sift	+ Shape Contexts+ Effect of gradient scale	+ 在计算梯度时高斯平滑带来的效果测试+ Effect of orientation bins’ number	+ 考察角度分区带来的效果变化+ Effect of normalization method	+ 在block归一化时，考察不同归一化方法的效果+ Effect of overlap	+ 在block与cell进行组合时，不同的overlap带来的效果。此时cell的大小为8×8.	+ 注意：这里有一个参数stride，表示block每次滑动间隔的像素数。+ Block大小与Cell大小不同带来的效果变化+ 检测窗口大小不同带来的效果变化+ SVM参数带来的效果变化
> 检验指标：	+ Miss Rate：错检率，所有判为有行人的sample中，被错判（没有行人被判为有行人）的样本比例。+ FPPW：False Positives Per Window，平均每个窗口的漏检率，漏检率为所有有行人的sample中，被判为没有行人的样本比例。平均到每个检测窗口

## 5. Hog算子深入理解
> Hog算子最重要的思想是，在一副图像中，局部目标的appearance和shape能够被梯度或边缘的方向密度分布很好的描述。	Hog算子有很多优点：
+ 由于其在图像的局部细胞单元上操作，所以对图像的几何（geometric）和光学（photometric）形变都能保持很好的不变形，因为这两种形变只会出现在更大的空间区域上。
+ 实验表明，粗的空域抽样（coarse spatial sampling）、精细的方向抽样（fine orientation sampling）以及较强的局部光学归一化（strong local photometric normalization）等条件下，行人只要大体能够保持直立的姿势，容许有一些细微的肢体动作而不影响检测效果。
> R-HOG跟SIFT描述器看起来很相似，但他们的不同之处是：R-HOG是在单一尺度下、密集的网格内、没有对方向排序的情况下被计算出来（are computed in dense grids at some single scale without orientation alignment）；而SIFT描述器是在多尺度下、稀疏的图像关键点上、对方向排序的情况下被计算出来（are computed at sparse, scale-invariant key image points and are rotated to align orientation）。补充一点，R-HOG是各区间被组合起来用于对空域信息进行编码（are used in conjunction to encode spatial form information），而SIFT的各描述器是单独使用的（are used singly）。## 6. Reference+ [1]. Dalal N, Triggs B. Histograms of oriented gradients for human detection[C]//Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on. IEEE, 2005, 1: 886-893.
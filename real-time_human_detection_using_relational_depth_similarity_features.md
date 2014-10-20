# Real-Time Human Detection using Relational Depth Similarity Features
> 本文使用TOF摄像头采集到的图像里有深度信息的特点，提出了一种新的深度相关的相似度特征（RDSF）来检测行人。同时，利用深度信息，可以判断人员遮挡的情况，将这种情况考虑进算法，可以同时提升Hog与RDSF的效果。
## 1  Ideas
> 论文使用可以采集深度信息的摄像头可以轻松的发现边缘与重叠部分，能否应用到人脸检测上？
> 使用深度信息只有当物体厚度较小时（如行人）才能轻松发现边缘，如果物体厚度较大，深度信息是否还有利于发现边缘，有什么算子可以帮助它？
## 2  Problems of Conventional methods> Conventional methods include features based on gradients like Hog, EOH, edgelets.
	Problems:
+ Human occlusions
+ Complex backgrounds
+ Real-time processing because of the use of the raster scanning while varying the window scale
## 3  Extraction
### 3.1  Local features based on depth information+ Divide depth image into local regions with cell of size 8*8 pixels+ Compute depth histograms from the depth information+ Normalize the histograms so that the total value of each depth histogram is 1

![equation1](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/3-1.png)+ Compute the similarity of two depth histograms using equation below
![equation1](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/3-2.png) 
	### 3.2  Varied rectangular region sizes
a. Control the window size to 64*128 so it can be divided into 8*16 cells.
b. There are 492 rectangular regions obtained by varying the cell units of the rectangular region from 1*1 to 8*8.
c. Calculate the RDSF from combinations of the 492 rectangular regions, 492*491/2=120786 features. 
> Note：492=8×16+7×15+6×14+5×13+4×12+3×11+2×10+1×9
### 3.3  Faster depth histogram calculations by integral histograms
a. 使用积分图来加快深度直方图的计算
b. 深度以0.3米为单位，而TOF摄像头则是深度范围为0.3-7.5m，所以，共分为25个bin，所以共需25个积分图。
## 4  Human Detection
a.	Perform a raster scan of the detectin window in a 3D space
b.	Computes the RDSFs from the detection windows
c.	Judge whether there are occlusions in the calculated features
d.	Use Real AdaBoost to classify each detection window is of human or not human.
	流程如下图：

![equation1](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/3-3.png) ### 4.1  Raster scaning in 3D space> Convolutional human detection methods involve repeated raster scans while the scale of the detection window varied, so there are many windows do not match the dimisions of humans.
> With the depth information, we can use fixed window size with different depth to detect humans with different scales. Process can be seen below:
![equation1](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/3-4.png) 
> Window with different depth can projected to the 2d image using a projection matrix which is the equation 7 in the paper.
### 4.2  Classification> Using Real Adaboost algorithm to classify the extracted features.
> Adaboost algorithm can ensemble a number of weak classifiers to build a strong classifer.
	
![equation1](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/3-5.png) 
> H(x) is the final strong classifier, and h(x) is a weak classifier.
### 4.3  Classification adjusted for occlusions> Depth information is useful in a confusing scene with a number of people overlapping. Combine the overlapping information into the classifier in a simple way. Process is below:
a. Define occlusion: any object region that is closer to the camera than the detection window
b. Extraction of occlusion regions:
	![equation1](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/3-6.png) 
c. Calculate the proportion of occlusion regions:
	![equation1](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/3-7.png)
 d. Combine the rate OR into the Adaboost algorithm:
	![equation1](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/3-8.png)
	 ## 5  Mean-shift clustering> Mean-shift clustering is a method that clustering the detect windows which detect a same object into one window. In image space, detection window could be erroneously integrated if humans overlap in them. But in 3D space, this problem can be solved easily.
![equation1](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/3-9.png) 
## 6  Experiments	There are two expriments:
	+ Comparison of three feature extraction methods:

	+ HOG
		+ RDFS
		+ HOG+RDFS
	+ Comparison of occlusion and non-occlusion adjustment feature extraction methods:
	+ HOG without occlusion adjustment
		+ HOG with occlusion adjustment
		+ RDFS without occlusion adjustment
		+ RDFS with occlusion adjustment
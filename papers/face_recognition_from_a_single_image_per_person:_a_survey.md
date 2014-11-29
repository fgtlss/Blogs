# Face recognition from a single image per person: A Survey
## 1	Problem Definition
Given a stored database of faces, the goal is to identify a person from the database later in time in any different and unpredictable poses, lighting, etc. from just one image.
## 2	Background
> Face Recognition techniques can divided into two periods:
+ Geometric-based method period, which is early. For these methods, one sample problem is not a problem.+ Apperence-based method period, which is now popular. Which is armed with intelligent tools such as computer vision, pattern recognition, machine learning and neural network, making the feature extraction and face recognition processes more effective and efficiency. 
However, the Apperence-based method suffers the one sample problem.
## 3	Challenges and Significance of One Sample Problem
> Many existing face recognition suffers the one sample problem. These techniques includes:
+ Standard eigenface (PCA).+ LDA based subspace algorithms.+ Probabilistic-based eigenface.+ SVM-based method.+ Feature line method.+ Evolution pursuit.+ Laplacian faces.
To see the paper for detailed explanation.
> Significance of the one sample problem:
+ Easy to collect samples.+ Save storage cost.+ Save computational cost.
## 4	Recognizing from one sample per person
![equation1](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/7-1.png)
 + Holistic method: identify a face using the whole face as input.+ Local method: use local facial features.+ Hybrid method: use both local and holistic features to recognize a face.
在下面的内容中，我选择了上表中跟现在研究的算法相关的技术进行说明。
# 5	Enlarge the training Set
+ Construct new representations	+ Mining more information from the face at hand
	![equation1](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/7-2.png) + Create novel visual images	+ Visual sample generation method concentrates on learning extra information from the 	domain besides the training set.
	![equation1](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/7-3.png) ## 6	Local apperence-based method
![equation1](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/7-4.png)
![equation1](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/7-5.png)  ## 7	Reported results
![equation1](https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/imgs/7-6.png) ## 8	Reference
[1]. Tan X, Chen S, Zhou Z H, et al. Face recognition from a single image per person: A survey[J]. Pattern Recognition, 2006, 39(9): 1725-1745.
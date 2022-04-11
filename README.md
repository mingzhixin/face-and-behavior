# face-and-behavior
		
本系统包含人脸识别、人脸细粒度表情识别、异常行为检测和识别三个功能模块。
	
## 安装
实验环境需要1张显存容量为11GB的Nvidia GeForce RTX 2080Ti显卡，并安装Python 3.7、CUDA 11。依赖的主要开源库包括深度学习框架PyTorch、计算机视觉库OpenCV，以及matplotlib、pillow等工具包。
	
## 功能模块
face_recognition文件夹中是人脸识别的代码，是基于Light CNN框架搭建。
	
facial expression文件夹中式人脸细粒度表情识别的代码，利用JAA-Net提取人脸动作单元特征，捕捉面部局部表情细节；然后训练从动作单元特征回归到考生情绪状态类别的识别模型。
	
yolov5-deepsort文件夹是将YOLOv5算法和deepsort算法进行结合，先进行YOLOv5算法检测，然后将YOLOv5算法的输出作为DeepSort算法的输入。在和YOLOv5的结果进行比较的同时，进行DeepSort算法中的预测、观测和更新。

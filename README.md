	face-facial expression-behavior
	人脸检测和人脸表情识别和异常行为检测
	本系统包含人脸检测、人脸识别、人脸表情识别、异常动作检测和识别五个技术模块。
	实验环境需要1张显存容量为11GB的Nvidia GeForce RTX 2080Ti显卡，并安装Python 3.7、CUDA 11。
	依赖的主要开源库包括深度学习框架PyTorch、计算机视觉库OpenCV，以及matplotlib、pillow等工具包。
	face_recognition文件夹中是人脸识别的代码，是基于Light CNN框架搭建
	facial expression文件夹中式人脸表情识别的代码，利用JAANet提取人脸动作单元特征 捕捉表情细节，然后训练动作单元特征到考场情绪状态类别的模型。
	yolov5-deepsort文件夹是将YOLOv5算法和deepsort算法进行结合，是先进行YOLOv5算法检测，将YOLOv5算法的输出作为DeepSort算法的输入。
	在和YOLOv5的结果进行比较的同时，进行DeepSort算法中的预测、观测和更新

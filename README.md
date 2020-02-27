# Mask
Test face for mask

处理数据集以及构建模型用的是Jupyter Notebook,数据集下载链接：https://pan.baidu.com/s/18rq-NiU4nXH3xoFNpHtANg 提取码：0pz1

项目流程：
1、使用tensorflow框架对人图片数据集预处理，剪裁出人脸。搭建卷积神经网络模型，预测人脸是否有戴口罩
2、使用OpenCV 中的Haar-cascade检测器检测出眼睛区域，再计算出人脸区域
3、用cv2读入图片帧，调用预测接口，实现通过视频轴帧实时预测

from running import running
import cv2
import tensorflow as tf




if __name__=='__main__':
    # 加载模型文件
    model = tf.keras.models.load_model('M4.h5')
    # 创建一个级联分类器
    eye_cascade = cv2.CascadeClassifier('D:\PycharmProjects\haarcascade_eye.xml')

    test = running()
    test.detect_face_mask(model,eye_cascade)


import cv2
import numpy as np



class running:


    def detect_face_mask(self,model,eye_cascade):


        # 打开摄像头
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('Dynamic')


        while (True):

            # 读取一帧图像
            ret, frame = camera.read()
            # 判断图片读取成功？

            if ret:
                gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 眼睛检测
                eyes = eye_cascade.detectMultiScale(gray_img, 1.3, 5)

                # 处理眼睛坐标。
                # 如果获取的眼睛数量是双数
                if len(eyes) % 2 == 0 and len(eyes) > 0:
                    # 根据眼睛坐标计算出人脸坐标
                    face_r = self.eyes_for_face(eyes)

                    for x, y, w, h in face_r:
                        # 截取出人脸区域
                        img = frame[y:y+h, x:x+w]
                        #对人脸区域进行预测戴没戴口罩
                        prd = self.make_prediction(img, model)
                        if prd > 0.5:

                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, "hello!", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
                        else :

                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(frame, "Subversive!Please wear a mask!", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
                else:
                    cv2.putText(frame, "Look at the camera!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 1)
                cv2.putText(frame, "Press Q to exit", (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # cv2.namedWindow('Dynamic', 0)
                # cv2.resizeWindow("Dynamic", 1000, 1000)
                cv2.imshow('Dynamic', frame)
                # 如果按下q键则退出

                if cv2.waitKey(100) & 0xff == ord('q'):
                    break

        camera.release()
        cv2.destroyAllWindows()

    def make_prediction(self, image, model):
        # 预处理
        img = cv2.resize(image, (48, 48))
        img = img / 255
        img = np.expand_dims(img, axis=0)

        # 用训练好的模型进行训练
        prd = model.predict(img)

        return prd

    def eyes_for_face(self,eyes):

        eyes = eyes.tolist()
        es = []
        for a in eyes:
            es.append(a)
            eyes.remove(a)
            b = sorted(eyes, key=lambda x: abs(x[1] - a[1]))[0]
            es.append(b)
            eyes.remove(b)
            if abs(a[0] - b[0]) > 130:
                es.remove(a)
                es.remove(b)

        i = 0
        face_r=[]
        while i < len(es):
            if es[i][0] > es[i+1][0]:
                es[i], es[i+1] = es[i+1], es[i]

            x = int((es[i][0] * 0.85))
            y = int((es[i][1] * 0.65))
            w = int((es[i][2] * 5))
            h = int((es[i][3] * 6))

            i = i + 2
            face_r.append([x, y, w, h])
        return face_r
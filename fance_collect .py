import numpy as np
import face
import sys
import cv2 ##opencv
import msvcrt
import threading
import time

n = 1
def catch(name):
    classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    color = (0, 255, 0)
    recognition = face.Recognition(1)
    cap=cv2.VideoCapture(0)#开启摄像头
    while cap.isOpened():
    	
        ret,frame=cap.read()#读一帧
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))#人脸的框
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)


        cv2.imshow("detect", frame)
        c = cv2.waitKey(30)
        if c & 0xFF == ord('c'):  # 收集
            save_face = recognition.add_identity(frame, name)
            if save_face:

                cv2.imwrite("C:\\Users\\xiaomiao\\Desktop\\facenet\\face_lib\\%s.jpg"%name, frame)
                np.save("C:\\Users\\xiaomiao\\Desktop\\facenet\\face_lib\\%s.npy" % name+str(n), save_face.embedding)
               
                print(save_face.embedding,save_face.embedding.shape)
                print("采集成功!")
            else:
                print("采集失败，再来一次")
            break
        if c & 0xFF == ord('q'):  # 释放
             
            break
    cap.release()
    cv2.destroyAllWindows()
def test():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PR OP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('frame', frame)  # 一个窗口用以显示原视频
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    catch(sys.argv[1])

#def test():
#   test_img = imread("/home/btows/Downloads/data/liu/liu0005.jpg")
#   recognition = face.Recognition()
#   test_face = recognition.identify(test_img)
#   print(test_face[0].name)



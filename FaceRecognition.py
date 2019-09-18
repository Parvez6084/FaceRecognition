import cv2
import numpy as np

face_Classifier = cv2.CascadeClassifier('C:/Users/PARVE/PycharmProjects/FaceRecognition/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')

def face_extroctor(img):

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_Classifier.detectMultiScale(gray, 1.3, 6)

        if faces is ():
            return None

        for (x,y,w,h) in faces :
            crop_faces = img [y:y+h, x:x+w]

        return crop_faces


cap = cv2.VideoCapture(0)
count = 0

while True :
    ret, frame = cap.read()
    if face_extroctor(frame) is not None :
        count += 1
        face = cv2.resize(face_extroctor(frame), (200, 200))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)


        file_name_path = 'D:/WorkShop/Python/faces/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not found!")
        pass

    if cv2.waitKey(1)==13 or count == 100:
        break


cap.release()
cv2.destroyAllWindows()
print("Collection sample completed !!!")


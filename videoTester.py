import os
import cv2
import numpy as np
import checkFace as fr


#This module captures images via webcam and performs face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.xml')#Load saved training data

# faces,faceID=fr.labels_for_training_data('C:\\Users\\giang\\Desktop\\TestVScode\\Images')
# face_recognizer=fr.train_classifier(faces,faceID)
# face_recognizer.write('trainingData.xml')


name = {0 : "Noo",1 : "Giang"}


cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    faces_detected,gray_img=fr.faceDetection(test_img)


    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",confidence)
        print("label:",label)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        if confidence < 50:
            fr.put_text(test_img,predicted_name,x,y-30)


    resized_img = cv2.resize(test_img, (700, 500))
    cv2.imshow('face recognition tutorial ',resized_img)
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break


cap.release()
cv2.destroyAllWindows


import checkFace as fr

faces,faceID=fr.labels_for_training_data('C:\\Users\\giang\\Desktop\\TestVScode\\Images')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.write('trainingData.xml')
import cv2, os
import numpy as np
from PIL import Image
import timeit

start = timeit.default_timer()
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("./fileXML/haarcascade_frontalface_default.xml");
print("Data sedang ditrainingkan")

def getImagesWithLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)

    return faceSamples, Ids
    
faces, Ids = getImagesWithLabels('dataWajah')
recognizer.train(faces, np.array(Ids))
recognizer.save('./latihWajah/training.xml')

print("Data telah ditrainingkan")
stop = timeit.default_timer()
lama_eksekusi = stop - start 

print(lama_eksekusi)

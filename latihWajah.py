from asyncore import read
import cv2, os, numpy as np
from PIL import Image

wajahDir = 'dataWajah' # Folder asal (src)
latihDir = 'latihWajah' # Folder tujuan disimpannya training

file = open('identitas/namaWong.txt', 'r')
nama = file.read()
file.close()

def gambar(path): 
    pathsGambar = [os.path.join(path, f) for f in os.listdir(path)]
    samples = []
    idMuka_ = []
    for pathGambar in pathsGambar:
        PILImg = Image.open(pathGambar).convert('L') # Convert ke dalam grey
        imgNum = np.array(PILImg, 'uint8')
        idMuka = int(os.path.split(pathGambar)[-1].split('.')[1])
        faces = deteksiMuka.detectMultiScale(imgNum)
        for (x, y, w, h) in faces:
            samples.append(imgNum[y:y+h, x:x+w])
            idMuka_.append(idMuka)
            return samples, idMuka_

faceRecognizer = cv2.face.LBPHFaceRecognizer_create() # algoritma LBPH
deteksiMuka = cv2.CascadeClassifier('./fileXML/pendeteksiWajah.xml')

print("Sedang melakukan training wajah")
faces, IDs = gambar(wajahDir + '/' + nama)
faceRecognizer.train(faces, np.array(IDs))

faceRecognizer.write(latihDir + '/training.xml')
print('Data wajah telah ditrainingkan neehhhh')
from tkinter import font
from tensorflow.keras.models import model_from_json
from keras_preprocessing import image
import cv2, os, numpy as np, speech_recognition as sr

# Untuk memanggil kamera
kamera = cv2.VideoCapture(0) 
# Baca file dari gambar maka masukan src file gambar, jika dari webcam maka masukan index webcam yang kita miliki
kamera.set(3, 640) # Untuk mengubah lebar kamera
kamera.set(4, 480) # Untuk mengubah tinggi kamera

wajahDir = 'dataWajah' # Folder asal (src)
latihDir = 'latihWajah' # Folder tujuan disimpannya training

deteksiWajah = cv2.CascadeClassifier('./fileXML/haarcascade_frontalface_default.xml') # src xml mengenai deteksi wajah
faceRecognizer = cv2.face.LBPHFaceRecognizer_create() # algoritma LBPH

model = model_from_json(open('./fileXML/facial_expression_model_structure.json').read())
model.load_weights('./fileXML/facial_expression_model_weights.h5') 

emosi = ('Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut','Biasa')

file = open('identitas/namaWong.txt', 'r')
nama = file.read()
file.close()

faceRecognizer.read(latihDir + '/training.xml') # Untuk read hasil gambar yang sudah ditraining
font = cv2.FONT_HERSHEY_DUPLEX # Memilih font untuk menampilkan nama sipemilik wajah

id = 0
names = ['Unknown', 'Fadhil', 'Zaki', 'Rhezi']

minWidth = 0.1*kamera.get(3)
minHeight = 0.1*kamera.get(4)

mesinSuara = sr.Recognizer()
mic = sr.Microphone() 
hasil = ""

while True: # Perulangan yang berguna untuk menangkap frame per secon
    
    retV, frame = kamera.read() # Disini kamera membaca/merekam
    frame = cv2.flip(frame, 1) # Vertical flip
    warna = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Mengganti warna pada kamera
    muka = deteksiWajah.detectMultiScale(warna, 1.2, 5, minSize=(round(minWidth), round(minHeight))) # frame, scalefactor, min
        
    for (x, y, w, h) in muka :
        id, cocokGatu = faceRecognizer.predict(warna[y:y+h, x:x+w]) # Kecocokan = 0 berarti gambarnya cocok
        if cocokGatu <= 45:
            nameID = names[id] # Identifikasi nama pemilik wajah
            cocokGatuTxt = " {0}%" . format(round(100-cocokGatu))
            deteksi = (20, 255, 0)
        else:
            nameID = names[0]
            cocokGatuTxt = " {0}%" . format(round(100-cocokGatu))
            deteksi = (255, 0, 0)
        
        detected_face = frame[int(y):int(y + h), int(x):int(x + w)]
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
        detected_face = cv2.resize(detected_face, (48, 48))
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotion = emosi[max_index]

        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), deteksi, 2) # Running retangle untuk mendeteksi wajah
        cv2.putText(frame, hasil, (0, 25), font, 0.5, deteksi)
        cv2.putText(frame, emotion, (int(x + 200), int(y - 5)), font, 0.5, deteksi, 2)
        cv2.putText(frame, str(nameID), (x+5, y-5), font, 1, deteksi) # Meletakkan text nama
        cv2.putText(frame, str(cocokGatuTxt), (x+5, y-5+h), font, 1, deteksi) # Meletakkan text persentase kecocokan
    
    cv2.imshow('Recognisi wajah', frame) # Memanggil kamera untuk menampilkan output
    filter = cv2.waitKey(1) & 0xFF
    if filter == 27 or filter == ord('q'): # Pada bagian ini untuk memberi keterangan button kamera untuk di stop
        break

print("Selesai yaa :)")
kamera.release() # Release cache kamera ketika digunakan agar tidak memakan source pada komputer
cv2.destroyAllWindows() # Menyelesaikan session
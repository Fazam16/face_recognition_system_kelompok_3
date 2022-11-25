from tkinter import font
from tensorflow.keras.models import model_from_json
from keras_preprocessing import image
from imutils.video import VideoStream
from numpy import imag
from PIL import Image, ImageTk
import tkinter as tk, numpy as np, imutils, os, speech_recognition as sr, cv2, imutils

kamera = cv2.VideoCapture(1)

wajahDir = 'dataWajah'
latihDir = 'latihWajah'

deteksiWajah = cv2.CascadeClassifier('./fileXML/haarcascade_frontalface_default.xml') # src xml mengenai deteksi wajah
faceRecognizer = cv2.face.LBPHFaceRecognizer_create() # algoritma LBPH

model = model_from_json(open('./fileXML/facial_expression_model_structure.json').read())
model.load_weights('./fileXML/facial_expression_model_weights.h5') 

emosi = ('Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut','Biasa')

file = open('identitas/namaPengguna.txt', 'r')
nama = file.read()
namaWong = nama.split(',')
file.close()

names = ['unknown']

if len(namaWong) >= 1:
    for i in range(len(namaWong)):
        names.append(namaWong[i])

faceRecognizer.read(latihDir + '/training.xml') # Untuk read hasil gambar yang sudah ditraining
font = cv2.FONT_HERSHEY_DUPLEX # Memilih font untuk menampilkan nama sipemilik wajah

minWidth = 0.1*kamera.get(3)
minHeight = 0.1*kamera.get(4)

hasil = ""
id = 0

def deteksi(frame):
    while stat:
        frame = cv2.flip(frame, 1) # Vertical flip
        warna = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Mengganti warna pada kamera
        muka = deteksiWajah.detectMultiScale(warna, 1.2, 5, minSize=(round(minWidth), round(minHeight))) # frame, scalefactor, min
            
        for (x, y, w, h) in muka :
            id, cocokGatu = faceRecognizer.predict(warna[y:y+h, x:x+w]) # Kecocokan = 0 berarti gambarnya cocok
            if cocokGatu <= 50:
                nameID = names[id] # Identifikasi nama pemilik wajah
                cocokGatuTxt = " {0}%" . format(round(120-cocokGatu))
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
            frame = cv2.putText(frame, hasil, (0, 25), font, 0.5, deteksi)
            frame = cv2.putText(frame, emotion, (int(x + 200), int(y - 5)), font, 0.5, deteksi, 2)
            frame = cv2.putText(frame, str(nameID), (x+5, y-5), font, 1, deteksi) # Meletakkan text nama
            frame = cv2.putText(frame, str(cocokGatuTxt), (x+5, y-5+h), font, 1, deteksi) # Meletakkan text persentase kecocokan
    
        filter = cv2.waitKey(1) & 0xFF
        if filter == 27 or filter == ord('q'): # Pada bagian ini untuk memberi keterangan button kamera untuk di stop
            break

        return frame

# GUI
frame = tk.Tk()
frame.geometry("1200x680+200+10")
frame.title("Face and Emotion Recognition by Kelompok 3 KB-L2")

bgImage = tk.PhotoImage(file="D:\\Tugas Kuliah\\Tugas kuliah semester 5\\KB\\face_recognition_system_kelompok_3\\anotherFile\\BGKB() (1).png")
bg = tk.Label(frame, image=bgImage).place(x=0, y=0, relwidth=1, relheight=1)

vd = tk.Label(frame, bg="black")

koorx = 530
koory = 70

def videostreamRecord():
    import recordWajah
    import latihWajah

def videostream():
    vd.place(x=koorx, y=koory)   
    global vs, stat
    # Inisiasi Videostream 
    print("[INFO] starting video stream...")
    stat = True
    vs = VideoStream(src=0).start()
    framevideo()

def framevideo():
    global stat
    frame = vs.read()
    frame = imutils.resize(frame, width = 620)
    frame = deteksi(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img  = Image.fromarray(frame)
    image = ImageTk.PhotoImage(image=img)
    vd.configure(image=image)
    vd.image = image
    vd.after(10, framevideo)

def stop():
    global video, stat
    stat = False
    vd.place_forget()
    cv2.destroyAllWindows()
    vs.stop()

BStart = tk.Button(frame, text ="Rekognisi Wajah", command=videostream, width=20, height=1, bg='#2870ff', fg='#ffffff')
BStart.place(x=770, y = 570)

BRecord = tk.Button(frame, text ="Record Wajah", command=videostreamRecord, width=20, height=1, bg='#2870ff', fg='#ffffff')
BRecord.place(x=570, y=570)

BQuit = tk.Button(frame, text ="Stop", command = stop, width=20, height=1, bg='#2870ff', fg='#ffffff')
BQuit.place(x=970, y=570)

copyRight = tk.Label(frame, text="@Copyright by Kelompok 3 KB L2", font=("Times New Roman", 10), underline=0, bg='#000000', fg='#ffffff')
copyRight.place(x=1000, y=650)

frame.mainloop()
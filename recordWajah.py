import cv2, os, speech_recognition as sr

# Untuk memanggil kamera
kamera = cv2.VideoCapture(0) 
# Baca file dari gambar maka masukan src file gambar, jika dari webcam maka masukan index webcam yang kita miliki
kamera.set(3, 640) # Untuk mengubah lebar kamera
kamera.set(4, 480) # Untuk mengubah tinggi kamera

deteksiWajah = cv2.CascadeClassifier('./fileXML/haarcascade_frontalface_default.xml')
ambilData = 1

mesinSuara = sr.Recognizer()
mic = sr.Microphone() 
hasil = ""

with mic as source:
    print("Sebut nama Anda")
    rekaman = mesinSuara.listen(source) # Mesin melakukan rekaman suara dan memasukan hasil kedalam variabel rekaman
    try:
        hasil = mesinSuara.recognize_google(rekaman, language="id-ID") # Memakai API google untuk melakukan rekaman suara, dan set bahasa indonesia
        print("Oke, terima kasih " + hasil)
    except mesinSuara.UnkownValueError: # Jika suara tidak dapat dikenali oleh mesin maka mesin akan melakukan : 
        print("Tidak dapat dideteksi")
    except Exception as e:
        print(e)

namaWong = open('identitas/namaWong.txt', 'a')
namaWong.write(hasil+',')
namaWong.close

nama = open('identitas/namaWong.txt', 'r')
ambilNama = nama.read()
splitAmbilNama = ambilNama.split(',')
idUser = len(splitAmbilNama)
nama.close

while True: # Perulangan yang berguna untuk menangkap frame per secon
    check, frame = kamera.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = deteksiWajah.detectMultiScale(abu, 1.3, 5)
   
    for(x,y,w,h) in wajah : 
        cv2.imwrite('./dataWajah/' + str(hasil) + '.' + str(idUser) + '.' + str(ambilData) + '.jpg', frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        ambilData+=1

    cv2.imshow('Face Recognition', frame)
    filter = cv2.waitKey(1) & 0xFF 
    if filter == 27 or filter == ord('q'): # Pada bagian ini untuk memberi keterangan button kamera untuk di stop
        break
    elif ambilData == 30:
        break

print("Program Selesai")
kamera.release() # Release cache kamera ketika digunakan agar tidak memakan source pada komputer
cv2.destroyAllWindows() # Menyelesaikan session
#Closing pemanggil kamera
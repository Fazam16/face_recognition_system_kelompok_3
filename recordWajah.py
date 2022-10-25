import cv2, os, speech_recognition as sr

# Untuk memanggil kamera
kamera = cv2.VideoCapture(0) 
# Baca file dari gambar maka masukan src file gambar, jika dari webcam maka masukan index webcam yang kita miliki
kamera.set(3, 640) # Untuk mengubah lebar kamera
kamera.set(4, 480) # Untuk mengubah tinggi kamera

deteksiWajah = cv2.CascadeClassifier('./fileXML/pendeteksiWajah.xml') # src xml mengenai deteksi wajah
ambilData = 1
wajahDir = 'dataWajah' #namaFolder

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

namaWong = open('identitas/namaWong.txt', 'w')
namaWong.write(hasil)
namaWong.close

buatFolder = r'./dataWajah/' + hasil 
if not os.path.exists(buatFolder):
    os.makedirs(buatFolder)

while True: # Perulangan yang berguna untuk menangkap frame per secon
    retV, frame = kamera.read() # Disini kamera membaca/merekam
    warna = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Mengganti warna pada kamera
    muka = deteksiWajah.detectMultiScale(warna,1.3, 5) # frame, scalefactor, min

    for (x, y, w, h) in muka:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2) # Running retangle untuk mendeteksi wajah
        namaFile = str(hasil) + '.' + str(ambilData) + '.jpg'
        cv2.imwrite(wajahDir + '/' + hasil + '/' + namaFile, frame) # wajahDir = namaFolder yang dituju
        ambilData+=1

    cv2.imshow('webcam', frame) # Memanggil kamera untuk menampilkan output

    filter = cv2.waitKey(1) & 0xFF 
    if filter == 27 or filter == ord('q'): # Pada bagian ini untuk memberi keterangan button kamera untuk di stop
        break
    elif ambilData == 100:
        break

print("Program Selesai")
kamera.release() # Release cache kamera ketika digunakan agar tidak memakan source pada komputer
cv2.destroyAllWindows() # Menyelesaikan session
#Closing pemanggil kamera
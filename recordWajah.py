import cv2, os, tkinter as tk, timeit
from tkinter import ttk

def submitNama():
    return namaUser.get()

window = tk.Tk()
window.configure(bg="white")
window.geometry("300x200")
window.resizable(False, False)
window.title("Input Nama")

input_frame = ttk.Frame(window)
input_frame.pack(padx=10, pady=10, fill='x', expand=True)

labelNama = ttk.Label(input_frame, text="Masukan Nama")
labelNama.pack(padx=10, pady=5, fill='x', expand=True)

namaUser = tk.StringVar()
entryNama = ttk.Entry(input_frame, textvariable=namaUser)
entryNama.pack(padx=10, fill='x', expand=True)

submit = ttk.Button(input_frame, text="Submit", command=submitNama)
submit.pack(fill='x', expand=True, padx=10)

tutup = ttk.Button(input_frame, text="Close", command=window.destroy)
tutup.pack(fill='x', expand=True, padx=10)

window.mainloop()

# Untuk memanggil kamera
kamera = cv2.VideoCapture(0) 
# Baca file dari gambar maka masukan src file gambar, jika dari webcam maka masukan index webcam yang kita miliki
kamera.set(3, 640) # Untuk mengubah lebar kamera
kamera.set(4, 480) # Untuk mengubah tinggi kamera

deteksiWajah = cv2.CascadeClassifier('./fileXML/haarcascade_frontalface_default.xml')
ambilData = 1

hasil = submitNama()

namaWong = open('identitas/namaPengguna.txt', 'a')
namaWong.write(hasil+',')
namaWong.close

nama = open('identitas/namaPengguna.txt', 'r')
ambilNama = nama.read()
splitAmbilNama = ambilNama.split(',')
idUser = len(splitAmbilNama)
nama.close

start = timeit.default_timer()

while True: # Perulangan yang berguna untuk menangkap frame per secon
    check, frame = kamera.read()
    frame = cv2.flip(frame, 1) # Vertical flip
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
    elif ambilData == 100:
        break

print("Program Selesai")
stop = timeit.default_timer()
lama_eksekusi = stop - start 

print("Waktu : " , round(lama_eksekusi, 2))
kamera.release() # Release cache kamera ketika digunakan agar tidak memakan source pada komputer
cv2.destroyAllWindows() # Menyelesaikan session
#Closing pemanggil kamera
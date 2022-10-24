from asyncore import read
import cv2, os, speech_recognition as sr

mesinSuara = sr.Recognizer()
mic = sr.Microphone() 
hasil = ""

with mic as source:
    print("Berbicara !")
    rekaman = mesinSuara.listen(source) # Mesin melakukan rekaman suara dan memasukan hasil kedalam variabel rekaman
    try:
        hasil = mesinSuara.recognize_google(rekaman, language="id-ID") # Memakai API google untuk melakukan rekaman suara, dan set bahasa indonesia
        print(hasil)
    except mesinSuara.UnkownValueError: # Jika suara tidak dapat dikenali oleh mesin maka mesin akan melakukan : 
        print("Tidak dapat dideteksi")
    except Exception as e:
        print(e)
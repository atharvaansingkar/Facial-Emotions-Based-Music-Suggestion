import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import pywhatkit
import random
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


def emotions():

    print("Stay natural and press space when you're ready.")
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k % 256 == 32:
            img_name = "opencv_frame_{}.png"
            cv2.imwrite(img_name, frame)
            break

    cam.release()

    cv2.destroyAllWindows()

    model = load_model("D:\\Python\\facial_emotions_model.h5")

    img_path = 'D:\\Python\\opencv_frame_{}.png'
    test_image = image.load_img(
        img_path, target_size=(48, 48), color_mode='grayscale')
    test_image = image.img_to_array(test_image)
    plt.imshow(test_image)
    plt.show()

    happy_songs = ["Aaj Ki Party", "Chogada", "Ainvayi Ainvayi", "London Thumakda", "Gallan Goodiyaan", "Badtameez Dil", "Dil Dhadakne Do", "Kar Gayi Chull", "Tamma Tamma Again",
                   "Kala Chashma", "Chaiyya Chaiyya", "Mere Khwabon Mein", "Tune Maari Entriyaan", "Jai Ho", "Hauli Hauli", "Kudi Nu Nachne De", "Lut Gaye", "Sadi Gali", "Sauda Khara Khara", "Dheeme Dheeme"]
    sad_songs = ["Zinda", "Kal Ho Na Ho", "Hum Hain Rahi Pyar Ke", "Chak De India", "Phir Se Udd Chala", "Lakshya", "Jeet", "Ruk Jaana Nahin", "Kandho Se Milte Hain Kandhe",
                 "Yun Hi Chala Chal", "Roobaroo", "love you zindagi", "Kar Har Maidaan Fateh", "Aashayein", "Atrangi Yaari", "Aazaadiyan", "Ae Watan", "Jai Ho", "Bharat Humko Jaan Se Pyara Hai", "Manzil"]
    angry_songs = ["Tum Se Hi", "Tum Ho", "Tujh Mein Rab Dikhta Hai", "Kabira", "Agar Tum Saath Ho", "Jab Tak", "Tere Bina", "Pee Loon", "Teri Ore", "Jab Koi Baat",
                   "Phir Le Aya Dil", "Kuch Kuch Hota Hai", "Muskurane", "Tera Ban Jaunga", "Judaai", "Tum Hi Ho", "Bekhayali", "Dil Diyan Gallan", "Tu Hi Hai", "Nazm Nazm"]
    fear_songs = ["Allah Ke Bande", "Abhi Mujh Mein Kahin", "Zindagi Kaisi Hai Paheli", "Ek Pyar Ka Nagma Hai", "Hum Honge Kamyaab", "Kabhi Alvida Naa Kehna", "Jeena Yahan Marna Yahan", "Kisi Ki Muskurahaton Pe", "Hum Hain Is Pal Yahan",
                  "Aye Zindagi Gale Lagaa Le", "Tum Aa Gaye Ho Noor Aa Gaya", "Muskurata Hua Mera Yaar", "hanuman chalisa", "gayatri mantra", "hare ram hare ram", "Aal Izz Well", "Roobaroo", "Aashayein", "Iktara", "Manzar Naya"]

    test_image = test_image.reshape(1, 48, 48, 1)
    classes = ['Angry', 'Fear', 'Fear', 'Happy', 'Sad', 'Sad', 'Fear']
    result = model.predict(test_image)
    y_pred = np.argmax(result[0])
    print('Emotion successfully detected, facial emotion is:', classes[y_pred])

    if (classes[y_pred] == 'Angry'):
        print("Please calm down and take deep breaths.\nListening to calm songs might help calm you down :)")
    elif (classes[y_pred] == 'Sad'):
        print("I'm sorry you're feeling sad, but know that you're not alone and things will get better.\nListening to some motivating and energetic songs might help uplift your mood :)")
    elif (classes[y_pred] == 'Happy'):
        print("Keep spreading your positive energy and joy, it's infectious and makes the world a brighter place.\nHere are some songs you can enjoy :)")
    elif (classes[y_pred] == 'Fear'):
        print("Remember that God is always with you, and trust in His love and protection to guide you through this difficult time.\nHere are a few songs that may help to calm your nerves and ease your fear")

    print("press 1 for songs to start playing, 2 for suggestions of songs from which you can choose from")
    ch = int(input())

    if (ch == 1):
        if (classes[y_pred] == 'Angry'):
            pywhatkit.playonyt(angry_songs[random.randint(0, 19)])
        elif (classes[y_pred] == 'Sad'):
            pywhatkit.playonyt(sad_songs[random.randint(0, 19)])
        elif (classes[y_pred] == 'Happy'):
            pywhatkit.playonyt(happy_songs[random.randint(0, 19)])
        elif (classes[y_pred] == 'Fear'):
            pywhatkit.playonyt(fear_songs[random.randint(0, 19)])

    elif (ch == 2):
        if (classes[y_pred] == 'Sad'):
            for k in range(20):
                print("press", k+1, "for", sad_songs[k])
            ch = int(input())
            pywhatkit.playonyt(sad_songs[ch-1])

        elif (classes[y_pred] == 'Angry'):
            for k in range(20):
                print("press", k+1, "for", angry_songs[k])
            ch = int(input())
            pywhatkit.playonyt(angry_songs[ch-1])

        elif (classes[y_pred] == 'Happy'):
            for k in range(20):
                print("press", k+1, "for", happy_songs[k])
            ch = int(input())
            pywhatkit.playonyt(happy_songs[ch-1])

        elif (classes[y_pred] == 'Fear'):
            for k in range(20):
                print("press", k+1, "for", fear_songs[k])
            ch = int(input())
            pywhatkit.playonyt(fear_songs[ch-1])


emotions()

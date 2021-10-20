import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random


img_array = cv2.imread("train/0/Training_63181.jpg")

Datadirectory = "train/" #training dataset
Classes = ["0", "1", "2", "3", "4", "5", "6"] #lista de clases

#for category in Classes:
    #path = os.path.join(Datadirectory, category)
    #for img in os.listdir(path):
        #img_array = cv2.imread(os.path.join(path,img))
        #plt.imshow(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB))
        #plt.show()
        #break
   # break

img_size = 224
new_array = cv2.resize(img_array, (img_size, img_size)) #pasa la imagen de 48x48 a 224x224
#plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
#plt.show()

#leemos todas las imagenes y las convertimos en arreglos

#-----------------------------------------------------------------------------------------------------
training_Data = [] #arreglo de datos

def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(img_size,img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                pass


create_training_Data()  #ejecuta la funcion anterior y carga las imagenes en el arreglo
#print(len(training_Data))
random.shuffle(training_Data)

#Se guardan los features
X = []
Y = []
for features,label in training_Data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1,img_size,img_size,3) #pasando a 4 dimensiones
X = X/255.0 #se normalizan los features
print(X)

#Deep learning model

model = tf.keras.applications.MobileNetV2()
#transfer learning
base_input = model.layers[0].input
base_output = model.layers[-2]
final_output = layers.Dense(128)(base_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_output)

new_model = keras.Model(inputs = base_input, outputs = final_output)

new_model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
#Se entrena el modelo con 25 epocas y se guarda en una archivo
new_model.fit(X,Y, epochs= 25)
new_model.save('my_model_93p33.h5')
new_model = tf.keras.models.load_model('my_model_93p33.h5')

#prueba final con imagenes y Preprocesamiento de imagen

frame = cv2.imread("happyboy.jpg") #imagen a reconocer
#se necesita un algoritmo de deteccion facial

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #reconocimiento de caras
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #pasa a gris la imagen deseada

faces = faceCascade.detectMultiScale(gray,1.1,4)
for x,y,w,h in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    facess = faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
        print("Rostro no detectado")
    else:
        for (ex, ey, ew, eh) in facess:
            face_roi = roi_color[ey:ey+eh, ex: ex+ew]

final_image = cv2.resize(face_roi, (224,224))
final_image = np.expand_dims(final_image, axis=0)
final_image = final_image/255.0

#predicciones

Predictions = new_model.predict(final_image)
np.argmax(Predictions) #este muesta el numero de la carpeta de la emocion


#falta descargar el haar, poner la imagen y hacer la parte de reconocimiento con la camara, el resto ya esta













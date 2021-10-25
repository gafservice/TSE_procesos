import cv2
import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.image import imread
from os import remove
import time

count=0

log=open('logfile.txt','w')
while (count!=6):
	
	cap = cv2.VideoCapture(0)
	flag = cap.isOpened()
	
	index = 1
	while(flag):

		ret, frame = cap.read()
		cv2.imshow("Captura de la imagen",frame)
		time.sleep(7)
		k = cv2.waitKey(1) & 0xFF
		
		cv2.imwrite("/home/saulep98/Downloads/" + str(index) + ".jpg", frame)
 		
		print("save" + str(index) + ".jpg successfuly!")
		print("-------------------------")

		index += 1
 		
		break

	cap.release()
	cv2.destroyAllWindows()

	img2 = cv2.imread('1.jpg')
	img2= cv2.resize(img2,dsize=(48,48))

	x = img2.astype('float32')
	y = np.expand_dims(x, axis=0)

	tflite_interpreter = tflite.Interpreter(model_path='model.tflite')

	input_details = tflite_interpreter.get_input_details()
	output_details = tflite_interpreter.get_output_details()

	tflite_interpreter.resize_tensor_input(input_details[0]['index'], (1, 48, 48, 3))
	tflite_interpreter.resize_tensor_input(output_details[0]['index'], (1, 6))


	tflite_interpreter.allocate_tensors()

	input_details = tflite_interpreter.get_input_details()
	output_details = tflite_interpreter.get_output_details()

	tflite_interpreter.set_tensor(input_details[0]['index'], y)

	tflite_interpreter.invoke()

	output_details = tflite_interpreter.get_output_details()[0]
	scores = tflite_interpreter.get_tensor(output_details['index'])[0]
	print("Predicted class label score      =", np.max(np.unique(scores)))
	k=np.max(np.unique(scores))
	count+=1

	if (k>1.9):
		log.write('surprise\n')
	if (k>0 and k<=0.6):
		log.write('fear\n')
	if (k>0.6 and k<=1):
		log.write('angry\n')
	if (k>1.3 and k<=1.6):
		log.write('sad\n')
	if (k>1 and k<=1.3):
		log.write('disgust\n')
	if (k>1.6 and k<=1.9):
		log.write('happy\n')
	remove('/home/saulep98/Downloads/1.jpg')

log.close()

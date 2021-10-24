import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
count=0
cam=cv2.VideoCapture(0)
log=open('logfile.txt','w')
while count!=32:
	check, frame = cam.read()
	img = cv2.imread(frame)
	res = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
	tflite_interpreter = tf.lite.Interpreter(model_path=model)

	input_details = tflite_interpreter.get_input_details()
	output_details = tflite_interpreter.get_output_details()

	tflite_interpreter.resize_tensor_input(input_details[0]['index'], (1, 224, 224, 3))
	tflite_interpreter.resize_tensor_input(output_details[0]['index'], (1, 5))
	tflite_interpreter.allocate_tensors()

	input_details = tflite_interpreter.get_input_details()
	output_details = tflite_interpreter.get_output_details()

	tflite_interpreter.set_tensor(input_details[0]['index'], res)

	tflite_interpreter.invoke()

	tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
	print("Prediction results shape:", tflite_model_predictions.shape)
	count+=1
	prediction_val=model.predict(X_test,1)
	if (prediction_val<=0.5):
		log.write('surprise\n')
	if (prediction_val>0.5 & prediction_val<=1.5):
		log.write('fear\n')
	if (prediction_val>1.5 & prediction_val<=2.5):
		log.write('angry\n')
	if (prediction_val>2.5 & prediction_val<=3.5):
		log.write('sad\n')
	if (prediction_val>3.5 & prediction_val<=4.5):
		log.write('disgust\n')
	if (prediction_val>4.5 & prediction_val<=5.5):
		log.write('happy\n')
	else:
		log.write('no se detecto emocion\n')

log.close()
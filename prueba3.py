import cv2
from os import remove
a=0
while(a!=5):

	cap = cv2.VideoCapture(0)
	flag = cap.isOpened()
 
	index = 1
	while(flag):
		ret, frame = cap.read()
		cv2.imshow("Capture_Paizhao",frame)
		k = cv2.waitKey(1) & 0xFF
		cv2.imwrite("/home/saulep98/Downloads/" + str(index) + ".jpg", frame)
 		
		print(cap.get(3))
		print(cap.get(4))
		print("save" + str(index) + ".jpg successfuly!")
		print("-------------------------")
		index += 1
 		
		break
	cap.release()
	cv2.destroyAllWindows()
	a+=1
	remove('/home/saulep98/Downloads/1.jpg')

# import numpy as np
import cv2
import threading as T
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

# import serial
# ser = serial.Serial('COM5', 9600, timeout=2)

recommend = 200
results = [0 for i in range(50)]
max_faces_num = 1; max_eyes_num = 0; faces_num = 0; eyes_num = 0
frames = 0
start_time = str(int(time.clock()*10e+6) % 10000) + '_'

time.sleep(2)
#multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

def analyze_eye(img, show_result):
	img = cv2.GaussianBlur(img,(3,3),1)
	left = 0; right = len(img[0])-1
	for i in range(len(img[0])//2, 1, -1):
		#print(min(img[:,i]),max(img[:,i]), min(img[:,i-1]),max(img[:,i-1]))
		if min(img[:,i]) > 95 and max(img[:,i]) < 200 and \
		   min(img[:,i-1]) > 95 and max(img[:,i-1]) < 200:
			left = i
			break
	for i in range(len(img[0]) // 2, len(img[0])-2):
		if min(img[:,i]) > 95 and max(img[:,i]) < 200 and \
		   min(img[:,i+1]) > 95 and max(img[:,i+1]) < 200:
			right = i
			break

	if show_result:
		cv2.imshow('img', cv2.resize(img, (100,100), interpolation=cv2.INTERSECT_NONE))
		cv2.waitKey(0)
	if left >= right:
		#print("no black area")
		return
	img = img[len(img)//3:len(img),left:right]
	#print(left, right)
	if show_result:
		cv2.imshow('img', cv2.resize(img, (100,100), interpolation=cv2.INTERSECT_NONE))
		cv2.waitKey(0)

	if len(img[0]) <= 1:
		#print("too small")
		return

	arg = np.argmin(list(map(lambda x: min(x),img.T)))
	return (arg - (len(img[0])-1)/2)/((len(img[0])-1)/2)

def main():
	global eyes_num, faces_num, frames
	# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


	cap = cv2.VideoCapture("00001.MTS")
	cap.set(3,1920)
	cap.set(4,1080)

	it = 0	
	while(True):

		_, img = cap.read()
		#cv2.imwrite("pictures/" + start_time + str(frames)+'.jpg', img)
		frames += 1
		eyes_count = 0

		it += 1
		if it % 50 != 0:
			continue
		#img = cv2.imread("people3.jpg")
		print(img.shape)
		if cv2.waitKey(1) & 0xFF == ord('t'):
			cv2.imwrite("people33.jpg", img)
			print("saving")
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.1, 10)
		faces = list(filter(lambda x: x[2] > 35, faces))
		faces_num = len(faces)

		biases = []
		for i,(x, y, w, h) in enumerate(faces):
			cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
			roi_gray = gray[y:y + h, x:x + w]
			roi_color = img[y:y + h, x:x + w]

			eyes = eye_cascade.detectMultiScale(cv2.resize(roi_gray,(recommend, recommend)), 1.2, 10)
			eyes = sorted(eyes, key=lambda x: x[1])[:2]
			eyes_count += len(eyes)

			common_bias = 0
			one_eye = False
			for j,(ex, ey, ew, eh) in enumerate(eyes):
				ex = int(round(ex*(w/recommend)))
				ey = int(round(ey*(h/recommend)))
				eh = int(round(eh*(h/recommend)))
				ew = int(round(ew*(w/recommend)))
				ex -= ew//4
				ew = ew*6//4

				search = i == 660 and j == 1
				if search:
					cv2.imshow('img', cv2.resize(roi_color, (100,100), interpolation=cv2.INTERSECT_NONE))
					cv2.waitKey(0)
					cv2.imshow('img', cv2.resize(roi_color[ey:ey+eh,ex:ex+ew], (100,100), interpolation=cv2.INTERSECT_NONE))
					cv2.waitKey(0)
					pass
					#cv2.imshow('img', cv2.resize(roi_gray, (recommend, recommend)))
					#cv2.waitKey(0)
				bias = analyze_eye(gray[y+ey:y+ey+eh, x+ex:x+ex+ew],search)
				#cv2.putText(img,str(i) + " " + str(j),(x+ex,y+ey-5),2,1.5,(255,0,0),5)
				cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
				if bias is None:
					one_eye = True
					continue
				common_bias += bias
				cv2.line(img, (x+ex+ew//2,y+ey+eh), (x+ex+ew//2+int(160*bias),y+ey+eh+80),(255,0,0),3)
			if not one_eye:
				common_bias /= 2
			biases.append(common_bias)

		eyes_num = eyes_count

		'''if len(biases) == 0:
			results.append(0)
		else:
			results.append(len(list(filter(lambda x: abs(x) < 0.1, biases)))/len(biases))
		if len(results) > 30:
			results.pop(0)'''

		cv2.imshow('img', cv2.resize(img,(1000,600)))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

th = T.Thread(target=main, args=())
th.start();


style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def animate(i):
	global max_eyes_num, max_faces_num, eyes_num, faces_num

	if max_faces_num < faces_num:
		max_faces_num = faces_num
	if max_eyes_num < eyes_num:
		max_eyes_num = eyes_num

	print(faces_num, eyes_num, max_faces_num, max_eyes_num)
	score = (faces_num + eyes_num/2) / (max_faces_num + max_eyes_num/2)
	results.append(score)
	if len(results) > 50:
		results.pop(0)
	score = int(round(score))
	
	averaged = moving_average(results,6)

	global ser
	ser.reset_input_buffer()
	ser.write(bytearray(str(int(averaged[len(averaged)-1]*47)), 'utf-8'))
	
	ax1.clear()
	ax1.set_ylim([0,1])
	ax1.plot(range(0,len(averaged)),averaged)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()

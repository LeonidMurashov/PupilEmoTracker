import numpy as np
import cv2

recommend = 200
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

def analyze_eye(img):
	img = cv2.GaussianBlur(img,(3,3),1)
	left = 0; right = len(img[0])-1
	for i in range(len(img[0])//2, 0, -1):
		if min(img[:][i]) > 130:
			left = i+1
			break
	for i in range(len(img[0]) // 2, len(img[0])-1):
		if min(img[:][i]) > 130:
			right = i-1
			break

	print(left, right)
	cv2.imshow('img', cv2.resize(img, (100,100), interpolation=cv2.INTERSECT_NONE))
	cv2.waitKey(0)
	img = img[:,left:right]
	cv2.imshow('img', cv2.resize(img, (100,100), interpolation=cv2.INTERSECT_NONE))
	cv2.waitKey(0)

	arg = np.argmin(list(map(lambda x: sum(x),img.T)))
	return (arg - (len(img[0])-1)/2)/((len(img[0])-1)/2)

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread("people2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
faces = list(filter(lambda x: x[2] > 35, faces))

for i,(x, y, w, h) in enumerate(faces):
	cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
	roi_gray = gray[y:y + h, x:x + w]
	roi_color = img[y:y + h, x:x + w]

	eyes = eye_cascade.detectMultiScale(cv2.resize(roi_gray,(recommend, recommend)), 1.3, 10)
	eyes = sorted(eyes, key=lambda x: x[1])[:2]
	for (ex, ey, ew, eh) in eyes:
		ex = int(round(ex*(w/recommend)))
		ey = int(round(ey*(h/recommend)))
		eh = int(round(eh*(h/recommend)))
		ew = int(round(ew*(w/recommend)))

		if i == 3:
			cv2.imshow('img', cv2.resize(roi_gray, (recommend, recommend)))
			cv2.waitKey(0)
		bias = analyze_eye(gray[y+ey:y+ey+eh, x+ex:x+ex+ew])
		print(bias)
		cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
		cv2.line(img, (x+ex+ew//2,y+ey+eh), (x+ex+ew//2+int(160*bias),y+ey+eh+80),(255,0,0),3)




cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

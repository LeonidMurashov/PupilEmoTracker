import numpy as np
import cv2

recommend = 200
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

def analyze_eye(img, show_result):
	img = cv2.GaussianBlur(img,(3,3),1)
	left = 0; right = len(img[0])-1
	for i in range(len(img[0])//2, 1, -1):
		print(min(img[:,i]),max(img[:,i]), min(img[:,i-1]),max(img[:,i-1]))
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
		print("no black area")
		return 0
	img = img[len(img)//3:len(img),left:right]
	print(left, right)
	if show_result:
		cv2.imshow('img', cv2.resize(img, (100,100), interpolation=cv2.INTERSECT_NONE))
		cv2.waitKey(0)

	if len(img[0]) <= 1:
		print("too small")
		return 0

	arg = np.argmin(list(map(lambda x: sum(x),img.T)))
	return (arg - (len(img[0])-1)/2)/((len(img[0])-1)/2)

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


cap = cv2.VideoCapture(0)
while(True):

	#_, img = cap.read()
	img = cv2.imread("people4.jpg")
	if cv2.waitKey(1) & 0xFF == ord('t'):
		cv2.imwrite("people6.jpg", img)
		print("saving")
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 5)
	faces = list(filter(lambda x: x[2] > 35, faces))

	for i,(x, y, w, h) in enumerate(faces):
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
		roi_gray = gray[y:y + h, x:x + w]
		roi_color = img[y:y + h, x:x + w]

		eyes = eye_cascade.detectMultiScale(cv2.resize(roi_gray,(recommend, recommend)), 1.2, 10)
		eyes = sorted(eyes, key=lambda x: x[1])[:2]
		for j,(ex, ey, ew, eh) in enumerate(eyes):
			ex = int(round(ex*(w/recommend)))
			ey = int(round(ey*(h/recommend)))
			eh = int(round(eh*(h/recommend)))
			ew = int(round(ew*(w/recommend)))
			ex -= ew//4
			ew = ew*6//4

			search = i == 60 and j == 0 and False
			if search:
				a = 6
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
			cv2.line(img, (x+ex+ew//2,y+ey+eh), (x+ex+ew//2+int(160*bias),y+ey+eh+80),(255,0,0),3)

	cv2.imshow('img', cv2.resize(img,(1000,600)))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()

from __future__ import print_function
import time
import requests
import cv2
import operator
import numpy as np
import json

# Import library to display results
import matplotlib.pyplot as plt

_url = 'https://westus.api.cognitive.microsoft.com/emotion/v1.0/recognize?'
_key = '45e50b7a90e54c3d8dcdfba98bf8fd3f' #Here you have to paste your primary key
_maxNumRetries = 10

def processRequest(json, data, headers, params):
	"""
	Helper function to process the request to Project Oxford

	Parameters:
	json: Used when processing images from its URL. See API Documentation
	data: Used when processing image read from disk. See API Documentation
	headers: Used to pass the key information and the data type request
	"""

	retries = 0
	result = None

	while True:

		response = requests.request('post', _url, json=json, data=data, headers=headers, params=params)

		if response.status_code == 429:

			print("Message: %s" % (response.json()['error']['message']))

			if retries <= _maxNumRetries:
				time.sleep(1)
				retries += 1
				continue
			else:
				print('Error: failed after retrying!')
				break

		elif response.status_code == 200 or response.status_code == 201:

			if 'content-length' in response.headers and int(response.headers['content-length']) == 0:
				result = None
			elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str):
				if 'application/json' in response.headers['content-type'].lower():
					result = response.json() if response.content else None
				elif 'image' in response.headers['content-type'].lower():
					result = response.content
		else:
			print("Error code: %d" % (response.status_code))
			print("Message: %s" % (response.json()['error']['message']))

		break

	return result


def renderResultOnImage(arr, img):
	"""Display the obtained results onto the input image"""
	for face in arr:
		rect = face['faceRectangle']
		cv2.rectangle(img,(rect['left'], rect['top']),(rect['left']+rect['width'], rect['top']+rect['height']),(255,0,0), 5)
		scores = face['scores']
		s = sorted()[0]
		print(np.argmax(scores))
		cv2.putText(img, s,(rect['left'], rect['top']), 5, 5, (255,255,255))

with open("people3.jpg",'rb') as f:
	data = f.read()

# Computer Vision parameters
params = { 'visualFeatures' : 'Color,Categories'}

headers = dict()
headers['Ocp-Apim-Subscription-Key'] = _key
headers['Content-Type'] = 'application/octet-stream'

result = processRequest( None, data, headers, params )

if result is not None:
	print(result)
	# Load the original image, fetched from the URL
	data8uint = np.fromstring( data, np.uint8 ) # Convert string to an unsigned int array
	img = cv2.imdecode( data8uint, cv2.IMREAD_COLOR )

	renderResultOnImage( result, img )

	cv2.imshow("fef", cv2.resize(img, (640,480)))
	cv2.waitKey(0)

####################################
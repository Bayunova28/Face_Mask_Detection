#import library
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
from keras.models import load_model

#generate face model
face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('TrainingModel.h5')
threshold = 0.90

#load video
webcam = cv2.VideoCapture(0)
webcam.set(3, 640)
webcam.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

#define function to process the video
def getPreprocessing(image):
    img = image.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

#define function to face classification
def getClassName(classNumber):
	if classNumber == 0:
		return 'Mask'
	elif classNumber == 1:
		return 'No Mask'

#define function to call webcam
while True:
	retV, frame = webcam.read()
	face_mask = face_model.detectMultiScale(frame,1.3,5)

	for x, y, w, h in face_mask:
		cropped_img = frame[y : y + h, x : x + h]
		img = cv2.resize(cropped_img, (32, 32))
		img = getPreprocessing(img)
		img = img.reshape(1, 32, 32, 1)
		prediction = model.predict(img)
		classIndex = np.argmax(prediction, axis = 1)
		probValue = np.amax(prediction)

		if probValue > threshold:
			if classIndex == 0:
				cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
				cv2.rectangle(frame, (x, y -40), (x + w, y), (0, 255, 0), -2)
				cv2.putText(frame, str(getClassName(classIndex)), (x, y -10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
			elif classIndex == 1:
				cv2.rectangle(frame, (x, y),(x + w, y + h),(50, 50, 255), 2)
				cv2.rectangle(frame, (x , y -40), (x + w, y), (50, 50, 255), -2)
				cv2.putText(frame, str(getClassName(classIndex)), (x, y -10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

	cv2.imshow('Face Mask Detection', frame)
	key = cv2.waitKey(1)

	if key == ord('q'):
		break

webcam.release()
cv2.destroyAllWindows()















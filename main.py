#import library
import cv2

#generate model
face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0

#load video
webcam = cv2.VideoCapture(0)

#define function to turn on the video
while True:
    retV, frame = webcam.read()
    face_mask = face_model.detectMultiScale(frame, 1.3, 5)

    for x, y, w, h in face_mask:
        count = count + 1
        output =  './Images/Face_Without_Mask/' + str(count) + '.jpg'
        print('Creating Face Without Mask Image.......' + output)
        cv2.imwrite(output, frame[y : y + h, x : x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('Face Mask Detection', frame)
    cv2.waitKey(1)

    if count == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()


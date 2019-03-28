import cv2
import time

#For Face Detection System------------------------------------------------------------

FACE_CLASSIFIER = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread(r"katrina.jpg")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = FACE_CLASSIFIER.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
print(faces)

for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)


cv2.imshow("Gray", img)
cv2.waitKey(0)

cv2.destroyAllWindows()


#For Video Capturing with opencv-------------------------------------------------------

video = cv2.VideoCapture(0)
a = 1


while True:
    a = a+1
    check, frame = video.read()
    print(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Capture", gray)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

print(a)

video.release()

cv2.destroyAllwindows()



#smile / face recognition 

import cv2
import random
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_detec = cv2.CascadeClassifier("haarcascade_smile.xml")
#img = cv2.imread("AIZoePic.jpg")
webcam = cv2.VideoCapture(0)
while True:
    randomint02551 = random.randint(0,255)
    randomint02552 = random.randint(0,255)
    randomint02553 = random.randint(0,255)
    successful_frame_read, frame = webcam.read()
    if not successful_frame_read:
        break
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coords = trained_face_data.detectMultiScale(grayscaled_img)
    smiles = smile_detec.detectMultiScale(grayscaled_img, scaleFactor = 1.7, minNeighbors = 20)
    for (x,y,w,h) in face_coords:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randomint02551,randomint02552,randomint02553), 2)
    for (x,y,w,h) in smiles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,200), 2)
    cv2.imshow("Jake - Face detector", frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

webcam.release()
cv2.destroyAllWindows()


"""
face_coords = trained_face_data.detectMultiScale(grayscaled_img)
#for (x,y,w,h) in face_coords:
(x,y,w,h) = face_coords[1]
cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)


#print(face_coords)
cv2.imshow("Jake - Face detector", img)
cv2.waitKey()                       #hold tab open
"""


print("CC")
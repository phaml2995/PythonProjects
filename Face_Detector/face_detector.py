import cv2

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image = cv2.imread(".jpg") #pass your .jpg file as an argument here

grayscale_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

face = cascade.detectMultiScale(grayscale_image,scaleFactor = 1.05, minNeighbors = 5)

for x,y,w,h in face:
    image = cv2.rectangle(image, (x,y),(x+w,y+h),(0,255,0),2)


print(face)

resize = cv2.resize(image,(int(image.shape[1]/4),int(image.shape[0]/4)))

cv2.imshow("Gray",resize)
cv2.waitKey(0)
cv2.destroyAllWindows()

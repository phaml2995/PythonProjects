import cv2, time

video = cv2.VideoCapture(0) #trigger the camera
first_frame = None
a = 0
while True:
    check,frame = video.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(25,25),0)

    if a < 40:
        first_frame = gray
        a += 1
        continue

    diff_frame = cv2.absdiff(first_frame,gray)
    thresh_delta = cv2.threshold(diff_frame,30,255,cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta,None,iterations = 2)
    # resized_gray = cv2.resize(gray,(int(gray.shape[1]/4),int(gray.shape[0]/4)))
    # resize_diff = cv2.resize(diff_frame,(int(diff_frame.shape[1]/4),int(diff_frame.shape[0]/4)))

    (_,cnts,_) = cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour)<1000:
            continue
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)


    # cv2.imshow("Capturing",gray)
    # cv2.imshow("Delta", diff_frame)
    cv2.imshow("Thresh",thresh_delta)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)
    print(gray)
    print(diff_frame)
    if key == ord('q'):
        break

video.release() #release the camera (turn it off)
cv2.destroyAllWindows()

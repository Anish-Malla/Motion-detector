import cv2 as cv
import time
from datetime import datetime
import pandas as pd

video = cv.VideoCapture(0)

# the first frame is going to be the "background" this variable will be given value below
first_frame = None
a = 1
motion_list = [None, False]
times = []

df = pd.DataFrame(columns=["Start time", "End Time"])

while True:
    motion = False

    if a == 1:
        time.sleep(2)
        a+=1
    check, frame = video.read(0)

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Gaussian blue removes noise and helps to increase acuracy in calculating the difference between the first frame
    # and the other frames
    gray_frame = cv.GaussianBlur(gray_frame, (21, 21), 0)

    if first_frame is None:
        time.sleep(2)
        first_frame = gray_frame
        cv.imwrite("first_frame.png", first_frame)
        continue

    delta_frame = cv.absdiff(first_frame, gray_frame)

    # setting up a threshold, if the difference is greater than 30 then it will be assigned a white color
    thresh = cv.threshold(delta_frame, 30, 255, cv.THRESH_BINARY)[1]
    thresh = cv.dilate(thresh, (None), iterations=1)

    # contour detection: (check the area of the contour and if it is big enough
    (cnts,_) = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv.contourArea(contour) < 10000:
            continue
        motion = True
        (x,y,w,h) = cv.boundingRect(contour)
        cv.rectangle(frame, (x,y), (x + w, y + h), (0,0,255), 3)
    motion_list.append(motion)

    if motion_list[-1] == True and motion_list[-2] == False:
        times.append(datetime.now())
    if motion_list[-2] == True and motion_list[-1] == False:
        times.append(datetime.now())

    cv.imshow("recoorrding", gray_frame)
    cv.imshow("delta_frame", delta_frame)
    cv.imshow("thresh hold", thresh)
    cv.imshow("Color frame", frame)
    key = cv.waitKey(1)
    if key ==ord("q"):
        if motion:
            times.append(datetime.now())
        break

print(motion_list)

for i in range(0,len(times),2):
    df = df.append({"Start time": times[i], "End Time": times[i+1]}, ignore_index=True)

df.to_csv("motion_detector_times.csv")

video.release()
cv.destroyAllWindows()
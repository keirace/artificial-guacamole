import cv2, imutils
import numpy as np
from transform import four_point_transform

# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
# NOTE: using the 'eval' function is bad form, but for this example
# let's just roll with it -- in future posts I'll show you how to
# automatically determine the coordinates without pre-supplying them
close = False
vid = cv2.VideoCapture("zoom_0.mp4")

while True:
    b, frame = vid.read()
    if b:
        width, height, _ = frame.shape
        frame = frame[0:int(height / 1.5), :]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)

        # find contours in the edged image, keep only the largest
        # ones, and initialize our screen contour
        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None

        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)

            if len(approx) == 4:
                screenCnt = approx
                break

        pts = np.array(eval("[(132, 220), (205, 215), (215, 265), (135, 275)]"), dtype="float32")
        # apply the four point transform to obtain a "birds eye view" of
        # the image
        if b is not False:
            warped = four_point_transform(frame, pts)
        # show the original and warped images
        cv2.imshow("Edged", edged)
        cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 3)

        cv2.imshow("Video", frame)
        cv2.imshow("Warped", warped)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or cv2.getWindowProperty("Video", 0) \
                or cv2.getWindowProperty("Edged", 0) == -1:
            close = True
            break
        if close:
            break
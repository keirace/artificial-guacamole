import cv2
import numpy as np
from transform import four_point_transform

# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
# NOTE: using the 'eval' function is bad form, but for this example
# let's just roll with it -- in future posts I'll show you how to
# automatically determine the coordinates without pre-supplying them
vid = cv2.VideoCapture("zoom_0.mp4")
while True:
    b, frame = vid.read()
    pts = np.array(eval("[(132, 220), (205, 215), (215, 265), (135, 275)]"), dtype="float32")
    # apply the four point transform to obtain a "birds eye view" of
    # the image
    if b is not False:
        warped = four_point_transform(frame, pts)
    # show the original and warped images
    cv2.imshow("Original", frame)
    cv2.imshow("Warped", warped)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
import cv2
import numpy as np

img = cv2.imread("photo.source\\2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow("result", img)
cv2.imshow("gray", gray)
cv2.waitKey(0)


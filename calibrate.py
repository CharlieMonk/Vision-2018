import numpy as np
import cv2

img2 = cv2.imread("/Users/cbmonk/Downloads/cube2.jpeg")
img = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
cv2.imshow("img", img)
print(img)
lower = [360, 100, 100]
upper = [0, 0, 0]

for i in img:
    for j in i:
        for k in range(0,3):
            if(lower[k]>j[k]):
                lower[k] = j[k]
        for k in range(0,3):
            if(upper[k]<j[k]):
                upper[k] = j[k]
print("Lower: " + str(lower))
print("Upper: " + str(upper))

cv2.waitKey(0)
cv2.destroyAllWindows()

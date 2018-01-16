import numpy as np
import cv2

img = cv2.imread("/Users/cbmonk/Downloads/cube2.jpeg")
cv2.imshow("img", img)

lower = [255, 255, 255]
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

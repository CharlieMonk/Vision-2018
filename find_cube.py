import cv2
import numpy as np
import time
def removeNoise():
    kernel = np.ones((3,3), np.uint8)

    cube_color_lower = np.array([190, 160, 60])
    cube_color_upper = np.array([255, 255, 160])

video_capture = cv2.VideoCapture(0)

while(True):
    # Get the frame
    _, img = video_capture.read()

    # Enable line below if reading from precaptured image
    #img = cv2.imread("/Users/cbmonk/Downloads/cube1.jpg")

    # Kernal to use for removing noise
    kernel = np.ones((5,5), np.uint8)
    cv2.imshow("Original Image", img)

    # Set values for thresholding
    cube_color_lower = np.array([0, 156, 139])
    cube_color_upper = np.array([124, 183, 202])

    # Remove noise
    mask = cv2.inRange(img, cube_color_lower, cube_color_upper)
    close_gaps = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    no_noise = cv2.morphologyEx(close_gaps, cv2.MORPH_OPEN, kernel)
    dilate = cv2.dilate(no_noise, kernel, iterations=5)

    # Find boundary of object
    _, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if(contours != None):
        if(len(contours) > 0):
            largest_area = 0
            largest_contour = 0;
            for i in range(0, len(contours)):
                area = cv2.contourArea(contours[i])
                if(area > largest_area):
                    largest_area = area
                    largest_contour = contours[i]
            color = (0,0,255)
            for cnt in contours:
                if(cnt.all() == largest_contour.all()):
                    color = (255,0,0)
                #cnt = contours[0]
                # Extract boundary points of object
                left = tuple(cnt[cnt[:,:,0].argmin()][0])
                right = tuple(cnt[cnt[:,:,0].argmax()][0])
                top = tuple(cnt[cnt[:,:,1].argmin()][0])
                bottom = tuple(cnt[cnt[:,:,1].argmax()][0])

                # Use boundary points to find the top left and bottom right corners
                top_left = (left[0], top[1])
                bottom_right = (right[0], bottom[1])

                # Draw a rectangle bounding the object using top left and bottom right points
                cv2.rectangle(img, top_left, bottom_right, color, 3)

                # Find the center point of the object
                center = (int((top_left[0]+bottom_right[0])/2), int((top_left[1]+bottom_right[1])/2))

                # Draw circle at the center point
                cv2.circle(img, center, 5, (0,0,255), -1)

                # Show the images
                cv2.imshow("Scanned Image", img)
                cv2.imshow("Mask Image", dilate)   # This should be enabled for debugging purposes ONLY!

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

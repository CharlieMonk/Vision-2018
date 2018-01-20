import cv2
import numpy as np
import glob
import os
def removeNoise(img, kernelSize):
    # Kernal to use for removing noise
    kernel = np.ones(kernelSize, np.uint8)

    # Set values for thresholding
    cube_color_lower = np.array([0, 156, 139])
    cube_color_upper = np.array([114, 173, 192])

    # Remove noise
    mask = cv2.inRange(img, cube_color_lower, cube_color_upper)
    close_gaps = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    no_noise = cv2.morphologyEx(close_gaps, cv2.MORPH_OPEN, kernel)
    dilate = cv2.dilate(no_noise, np.ones((5,10), np.uint8), iterations=5)
    return dilate

def findObject(dilate):
    # Find boundary of object
    _, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if(contours != None):
        if(len(contours) > 0): # Don't proceed if no contours are found
            # Find the largest contour
            largest_area = 0
            cnt = 0;
            for i in range(0, len(contours)):
                area = cv2.contourArea(contours[i])
                if(area > largest_area):
                    largest_area = area
                    cnt = contours[i]
            color = (0,0,255)

            #for cnt in contours:
            # if(cnt.all() == largest_contour.all()):
            #     color = (255,0,0)
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
            #cv2.imshow("Scanned Image", img)
            cv2.imshow("Mask Image", dilate)   # This should be enabled for debugging purposes ONLY!
            return img

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, 10)

counter = 0
ranOnce = False
folder = "/Users/cbmonk/Downloads/ImageLogging/"
while(True):
    # Get the frame
    _, img = video_capture.read()

    # Enable line below if reading from precaptured image
    #img = cv2.imread("/Users/cbmonk/Downloads/testf.png")

    dilate = removeNoise(img, (5,5))
    img2 = findObject(dilate)
    cv2.imshow("Objects found!", img)

    #Default index to use if no previous logging folders exist
    logging_folder = "0001"
    # Change the current directory to the logging folder (defined before this for loop began)
    os.chdir(folder)
    sorted_glob = sorted(glob.glob("[0-9][0-9][0-9][0-9]"))
    if len(sorted_glob)>0 and (not ranOnce):
        # If this is not the first logging folder, make a new folder with a 4 digit
        # name one greater than the previous logging folder
        logging_folder = "{:04d}".format(int(sorted_glob[-1])+1)
        print(logging_folder)
    if not ranOnce:
        # If this is the first time the program has been run, make a logging folder
        folder += logging_folder
        os.mkdir(logging_folder)
    # Path for the image to be saved
    path = os.path.join(folder, str(counter) + ".jpg")
    if(counter%10 == 0):
        # Log every 10th image
        cv2.imwrite(path, img)
        print(path)
    counter+=1
    ranOnce = True
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
import glob
import os
import sys
from udp_channels import UDPChannel
import time
import json
def removeNoise(img, kernelSize, lower_color_range, upper_color_range):
    # Kernal to use for removing noise
    kernel = np.ones(kernelSize, np.uint8)

    # Convert image to binary
    mask = cv2.inRange(img, lower_color_range, upper_color_range)
    # Show the binary (masked) image
    cv2.imshow("img", mask)
    # Close the gaps (due to noise) in the masked image
    close_gaps = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Remove noisy parts of the masked image
    no_noise = cv2.morphologyEx(close_gaps, cv2.MORPH_OPEN, kernel)
    # Undo the erosion to the actual target done during noise removal
    dilate = cv2.dilate(no_noise, np.ones((5,10), np.uint8), iterations=5)
    return dilate

def findObject(dilate):
    # Find boundary of object
    _, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Only proceed if contours were found
    if(contours != None):
        if(len(contours) > 0):
            # Find the largest contour
            largest_area = 0
            cnt = 0;
            for i in range(0, len(contours)):
                area = cv2.contourArea(contours[i])
                if(area > largest_area):
                    largest_area = area
                    cnt = contours[i]
            color = (0,0,255)

            # Extract boundary points of object
            left = tuple(cnt[cnt[:,:,0].argmin()][0])
            right = tuple(cnt[cnt[:,:,0].argmax()][0])
            top = tuple(cnt[cnt[:,:,1].argmin()][0])
            bottom = tuple(cnt[cnt[:,:,1].argmax()][0])

            print(right[0] - left[0])
            # Use boundary points to find the top left and bottom right corners
            top_left = (left[0], top[1])
            bottom_right = (right[0], bottom[1])

            # Draw a rectangle bounding the object using top left and bottom right points
            cv2.rectangle(img3, top_left, bottom_right, color, 3)
            # Find the center point of the object
            center_point = (int((top_left[0]+bottom_right[0])/2), int((top_left[1]+bottom_right[1])/2))
            # Draw circle at the center point
            cv2.circle(img3, center_point, 5, (0,0,255), -1)
            # Find the angle to the center point
            angle = getAngle(center_point)
            if(not isTesting):
                sendData(angle)
            # Show the images
            # cv2.imshow("Scanned Image", img)
            cv2.imshow("Mask Image", dilate)   # This should be enabled for debugging purposes ONLY!
            return img

def getAngle(center_point):
    # Use the center_point, fov, and width to find the heading (angle to target)
    field_of_view = 65
    pixel_distance = center_point[0] - width/2
    heading = ((field_of_view/2.0) * pixel_distance)/(width/2)
    # print(heading)
    return int(heading)

def sendData(angle):
    # Put the data (to be sent to the RIO) in a dictionary
    data = {
        "sender" : "vision",
        "message" : angle,
        "distance" : distance
    }
    # Convert the data to JSON and send it to the RIO
    channel.send_to(json.dumps(data))

# Set up a counter, for use in logging images
counter = 0
# Track if the program has ran (if not, create a new folder for image logging)
ranOnce = False
# Folder path for logging images
folder = "/var/log/"
# Track if the program is being tested
isTesting = False
# If test is found in the cmd line arguments, then the program is testing
for arg in sys.argv:
    if(arg == "test"):
        # When testing, use an alternate filepath
        folder = "/Users/cbmonk/Downloads/ImageLogging/"
        isTesting = True
        break

# Setup UDP Channel
rio_ip = "10.10.76.2"
channel = None
if(not isTesting):
    while channel == None:
        try:
            channel = UDPChannel(remote_ip=rio_ip, remote_port=5880,
                                 local_ip='0.0.0.0', local_port=5888, timeout_in_seconds=0.001)
        except:
            print("Error creating UDP Channel.")
            time.sleep(1)

# Set up the webcam input
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, 10)
# Find the resolution of the webcam input
_, img3 = video_capture.read()
_, width, _ = img3.shape
# print("----"+"\n\n\nWidth: " + str(width)+"\n\n\n----")


while(True):
    # Read the frame from the video capture
    _, img3 = video_capture.read()
    # Convert the frame to HSV
    img = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
    # Enable line below if reading from precaptured image
    # img = cv2.imread("/Users/cbmonk/Downloads/testf.png")

    # Find the cube
    cube_hsv_lower = np.array([25, 100, 100])
    cube_hsv_upper = np.array([28, 255, 215])
    cube_dilate = removeNoise(img, (5,5), cube_hsv_lower, cube_hsv_upper)
    cube_img = findObject(cube_dilate)

    # Find the retroreflective tape
    # Use these HSV values if the LEDs are very bright
    retro_hsv_lower = np.array([0, 0, 255])
    retro_hsv_upper = np.array([0, 0, 255])
    # Enable the below values if LEDs are NOT bright enough
    # retro_hsv_lower = np.array([71, 248, 221])
    # retro_hsv_upper = np.array([91, 255, 249])
    retro_dilate = removeNoise(img, (5,5), retro_hsv_lower, retro_hsv_upper)
    retro_img = findObject(retro_dilate)

    # Display the BGR image with found objects bounded by rectangles
    cv2.imshow("Objects found!", img3)

    # Default index to use if no previous logging folders exist
    logging_folder = "0001"
    # Change the current directory to the logging folder (defined before this for loop began)
    os.chdir(folder)
    sorted_glob = sorted(glob.glob("[0-9][0-9][0-9][0-9]"))
    if len(sorted_glob)>0 and (not ranOnce):
        # Make a new folder with a 4 digit name one greater than the last logging folder
        logging_folder = "{:04d}".format(int(sorted_glob[-1])+1)
        # print(logging_folder)
    if not ranOnce:
        # If this is the first time the program has been run, make a logging folder
        folder += logging_folder
        os.mkdir(logging_folder)
    # Path for the image to be saved
    path = os.path.join(folder, str(counter) + ".jpg")
    if(counter%10 == 0):
        # Log every 10th image
        cv2.imwrite(path, img)
        # print(path)
    counter+=1
    ranOnce = True
    # Exit the loop when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows when q is pressed
video_capture.release()
cv2.destroyAllWindows()

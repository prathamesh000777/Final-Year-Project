# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
# allow the camera to warmup
time.sleep(0.1)
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    width, height = 640,480
    img = cv2.resize(image,(width, height))


    # Converting the image from RGB to HSV format
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #mask for detecting black line
    lower_range = np.array([0,0,0])
    upper_range = np.array([180,255,30])
    mask = cv2.inRange(hsv, lower_range, upper_range)


    #splitting the mask in two parts to get the direction of line
    top = mask[0:height//2,:]
    bot = mask[height//2 +1:,:]

    def mass_center(ip):
        # convert image to grayscale image
        #gray_image = cv2.cvtColor(ip, cv2.COLOR_BGR2GRAY)

        # convert the grayscale image to binary image
        ret,thresh = cv2.threshold(ip,127,255,0)

        # calculate moments of binary image
        M = cv2.moments(thresh)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return [cX,cY]

    [top_x,top_y] = mass_center(top)   #get the mass center co-ordinates
    [bot_x,bot_y] = mass_center(bot)
    bot_y = bot_y + height//2 + 1  #correction of halving the image
    cv2.circle(img, (top_x,top_y), 5, (255, 0, 0), -1)  # highlighting mass center
    cv2.circle(img, (bot_x,bot_y), 5, (255, 0, 0), -1)

    #calculate the slope
    if (top_x - bot_x) != 0:
        slope = (top_y - bot_y)/(top_x - bot_x)
        intercept = top_y - top_x*slope
    else:
        slope = -100
        intercept = top_y - top_x*slope


    quad_x = width//2                  #postion of quad
    quad_y = height//2
    cv2.circle(img,(quad_x ,quad_y),5,(0,0,255),-1)  #highlighting the quad position
    d = abs(-slope*quad_x + quad_y - intercept)/(np.sqrt(slope*slope + 1))

    #we need to give sign to distance for indicating the postion of quad wrt line
    sign_check = (quad_x - top_x)*(bot_y - top_y) - (quad_y - top_y)*(bot_x - top_x)
    if sign_check > 0:
        d = -d

    #similary for theta, which denotes the required yaw
    theta = (np.arctan(slope))*180*7/22
    rot_req = ((90 - theta) if (theta >= 0) else  -(90 - abs(theta)))
    #print('d=',d,'theta=',rot_req,'intercept=',intercept,'slope=',slope)

    # Add distance and orientation detail on display
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    cv2.putText(img, 'y_e = ' + str(round(d,3)), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(img, 'psi_e = '+str(round(rot_req,3)), (300,60),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,), 2)

    # show the frame
    cv2.imshow("Frame", flipped)
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

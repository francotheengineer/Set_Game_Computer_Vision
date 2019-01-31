

import cv2
import numpy as np



def check_colour(img):
    height, width, dim = img.shape

    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red_low = cv2.inRange(hsv,(0, 0, 50), (10, 255, 255))

    red_high = cv2.inRange(hsv,(170, 0, 50), (180, 255, 255))
    purple =  cv2.inRange(hsv,(140, 0, 0), (165, 200, 255))
    # green = cv2.inRange(hsv, (63, 20, 0), (85, 226, 200))
    green = cv2.inRange(hsv, (36, 0, 0), (80, 255, 255))
    ## final mask and masked
    mask = cv2.bitwise_or(green, red_low,red_high,purple)

    target = cv2.bitwise_and(img, img, mask=mask)

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', target)
    cv2.waitKey(0)

img = cv2.imread('IMG_20190106_161905.jpg')
# img = cv2.imread('test.jpg')
check_colour(img)
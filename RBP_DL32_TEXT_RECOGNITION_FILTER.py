# https://circuitdigest.com/microcontroller-projects/license-plate-recognition-using-raspberry-pi-and-opencv
import cv2
import numpy as np
import imutils
import math
import pytesseract

min_confidence = 0.5
file_name = "image/plate_01.jpg"

frame_size = 320
margin = 0
    
def processROI(image):
    # hsv transform - value = gray image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    #cv2.imshow('gray', value)
    
    # kernel to use for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # applying topHat/blackHat operations
    topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)

    # add and subtract between morphological operations
    add = cv2.add(value, topHat)
    subtract = cv2.subtract(add, blackHat)

    # applying gaussian blur on subtract image
    blur = cv2.GaussianBlur(subtract, (5, 5), 0)

    # thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

    # inverse black plate to white background
    invert = cv2.bitwise_not(value)
    
    # cv2.findCountours() function changed from OpenCV3 to OpenCV4: now it have only two parameters instead of 3
    cv2MajorVersion = cv2.__version__.split(".")[0]
    # check for contours on thresh
    if int(cv2MajorVersion) >= 4:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # get height and width
    height, width = thresh.shape

    # create a numpy array with shape given by threshed image value dimensions
    imageContours = np.zeros((height, width, 3), dtype=np.uint8)

    left_border = int(width * 0.3)
    right_border = int(width * 0.8)
    top_border = int(height * 0.5)
    bottom_border = int(height * 0.8)
    cv2.line(imageContours, (0, top_border), (width, top_border), (0, 255, 255), 2)
    cv2.line(imageContours, (0, bottom_border), (width, bottom_border), (0, 255, 255), 2)
    cv2.line(imageContours, (left_border, 0), (left_border, height), (0, 255, 255), 2)
    cv2.line(imageContours, (right_border, 0), (right_border, height), (0, 255, 255), 2)
    
    plateROI = invert[top_border:bottom_border, left_border:right_border]
    plateX = left_border
    plateW = right_border - left_border
    plateY = top_border
    plateH = bottom_border - top_border
    # Sort by area and filter top 10 
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    # loop to check if any (possible) char is found
    for i in range(0, len(contours)):
        # check which has a rectangle shape contour with four sides and closed figure
        peri = cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], 0.018 * peri, True)
        boundingRect = cv2.boundingRect(contours[i])
        [x, y, w, h] = boundingRect
        rectArea = x * y
        if (rectArea > 100 and x > 10 and y > 10 and len(approx) == 4):
            # draw contours based on actual found contours of thresh image
            cv2.drawContours(imageContours, contours, i, (255, 255, 255))

            if (x > left_border and y > top_border and x+w < right_border and y+h < bottom_border):
                plateROI = invert[y-margin:y+h+margin, x-margin:x+w+margin]
                plateX = x
                plateW = w
                plateY = y
                plateH = h
                break
            
    cv2.imshow("Plate Candiates Contours", imageContours)
    cv2.imshow("Plate ROI", plateROI)
    
    return ([plateX, plateY, plateW, plateH], plateROI)

	
def textRead(image):
    # apply Tesseract v4 to OCR 
    config = ("-l eng --oem 3 --psm 12")
    text = pytesseract.image_to_string(image, config=config)
    # display the text OCR'd by Tesseract
    print("OCR TEXT : {}\n".format(text))
    
    # strip out non-ASCII text 
    text = "".join([c if c.isalnum() else "" for c in text]).strip()
    print("Alpha numeric TEXT : {}\n".format(text))
    return text

# Loading image
img = cv2.imread(file_name)
img_copy = img.copy()
#([x, y, w, h], car_image) = carROI(img)
#([startX, startY, endX, endY], text_image) = textROI(car_image)

([x, y, w, h], process_image) = processROI(img)

text = textRead(process_image)

cv2.rectangle(img_copy, (x-margin, y-margin), (x+w+margin, y+h+margin), (0, 255, 0), 2)
cv2.putText(img_copy, text, (x, y-margin-10),
    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

# show the output image
cv2.imshow("OCR Text Recognition : "+text, img_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()

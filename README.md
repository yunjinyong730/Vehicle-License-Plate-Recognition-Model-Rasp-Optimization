
## 🛺 차량 번호판 인식 모델, Rasp 최적화 
# 최종 결과물

https://github.com/user-attachments/assets/3fee0053-b047-47ba-b2ff-e11905a71c74




- 프로젝트에 사용할 이미지 (대표 예시)
![1](https://github.com/user-attachments/assets/6c8f4fef-68ec-4086-b8ec-1b484e1dd87e)



## Tesseract OCR 사용 text recognition with YOLOv3



https://github.com/user-attachments/assets/0bd5e550-76a5-4df9-80a5-5d9af3f3e9ba



- **코드 요약 설명**
    - 차량 번호판을 감지하고, 해당 번호판에서 텍스트를 인식하는 Python 코드이다. 이 코드는 OpenCV를 활용하여 객체 탐지 및 이미지 처리 작업을 수행하며, YOLO와 EAST 모델을 사용하여 차량 및 텍스트 감지를 구현한다
    - 번호판 텍스트는 Tesseract OCR을 통해 읽는다

```python
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import pytesseract

min_confidence = 0.5
file_name = "image/plate_01.jpg"

east_decorator = 'frozen_east_text_detection.pb'

frame_size = 320
padding = 0.05

# Load Yolo
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
def carROI(image):
    height, width, channels = image.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    confidences = []
    boxes = []
    img_cars = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Filter only 'car'
            if class_id == 2 and confidence > min_confidence:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            img_cars.append(image[y:y+h, x:x+w])
            return (boxes[i], image[y:y+h, x:x+w])

def textROI(image):
    # load the input image and grab the image dimensions
    orig = image.copy()
    (origH, origW) = image.shape[:2]
 
    # set the new width and height and then determine the ratio in change
    rW = origW / float(frame_size)
    rH = origH / float(frame_size)
    newW = int(origW / rH)
    center = int(newW / 2)
    start = center - int(frame_size / 2)
 
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, frame_size))  
    scale_image = image[0:frame_size, start:start+frame_size]
    (H, W) = scale_image.shape[:2]

    cv2.imshow("orig", orig)
    cv2.imshow("resize", image)
    cv2.imshow("scale_image", scale_image)
    
    # define the two output layer names for the EAST detector model 
    layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
    
    # load the pre-trained EAST text detector
    net = cv2.dnn.readNet(east_decorator)

    # construct a blob from the image 
    blob = cv2.dnn.blobFromImage(image, 1.0, (frame_size, frame_size),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities)
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):

                if scoresData[x] < min_confidence:
                        continue

                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
    
    # apply non-maxima suppression 
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # initialize the list of results
    results = []

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:

            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            dX = int((endX - startX) * padding)
            dY = int((endY - startY) * padding)

            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(origW, endX + (dX * 2))
            endY = min(origH, endY + (dY * 2))

            # extract the actual padded ROI
            return ([startX, startY, endX, endY], orig[startY:endY, startX:endX])

	
def textRead(image):
    # apply Tesseract v4 to OCR 
    config = ("-l eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(image, config=config)
    # display the text OCR'd by Tesseract
    print("OCR TEXT : {}\n".format(text))
    
    # strip out non-ASCII text 
    text = "".join([c if c.isalnum() else "" for c in text]).strip()
    print("Alpha numeric TEXT : {}\n".format(text))
    return text

def processROI(image):
    # hsv transform - value = gray image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    cv2.imshow("value", value)
    # kernel to use for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # applying topHat operations
    topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)

    # applying blackHat operations
    blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)

    # add and subtract between morphological operations
    add = cv2.add(value, topHat)
    subtract = cv2.subtract(add, blackHat)

    # applying gaussian blur on subtract image
    blur = cv2.GaussianBlur(subtract, (5, 5), 0)
    cv2.imshow("blur", blur)

    # thresholding
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imshow("thresh", thresh)    
    return image

# Loading image
img = cv2.imread(file_name)
img_copy = img.copy()
([x, y, w, h], car_image) = carROI(img)

([startX, startY, endX, endY], text_image) = textROI(car_image)

process_image = processROI(text_image)

text = textRead(process_image)

cv2.rectangle(img_copy, (x+startX, y+startY), (x+endX, y+endY), (0, 255, 0), 2)

cv2.putText(img_copy, text, (x+startX, y+startY-10),
    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

# show the output image
cv2.imshow("OCR Text Recognition : "+text, img_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()

```

### 1. **라이브러리 임포트**

- **`cv2`**: OpenCV 라이브러리로 이미지 처리 및 객체 탐지를 담당
- **`numpy`**: 배열 및 수학적 계산을 위한 라이브러리
- **`imutils.object_detection`**: 비최대 억제(non-max suppression)를 적용하여 객체 탐지의 정확도를 높임
    - ***Non-Max Suppression은 동일한 객체에 대해 다수의 겹치는 경계 상자가 예측되는 상황에서, 가장 신뢰도가 높은 하나의 경계 상자만 남기고 나머지 경계 상자를 제거하는 기법***
- **`pytesseract`**: OCR(Optical Character Recognition)을 위해 사용
    - ***광학 문자 인식(Optical Character Recognition)은 이미지 또는 스캔된 문서에서 텍스트를 추출하는 기술이다. 컴퓨터가 사람이 작성하거나 인쇄한 텍스트를 읽고 이를 디지털 데이터(텍스트 파일)로 변환할 수 있게 한다***
    - ***이미지 전처리 → 문자 영역 감지 → 문자 분리 → 문자 인식 → 결과 출력***
    - ***Tesseract는 가장 널리 사용되는 OCR 오픈소스 엔진 중 하나, Python의 pytesseract 라이브러리는 Tesseract 엔진을 파이썬에서 쉽게 사용할 수 있도록 도와준다***

### 2. **모델 초기화**

- **YOLO 모델 초기화**:
    - 가중치(`yolov3.weights`)와 설정 파일(`yolov3.cfg`) 로드
    - 출력 레이어 이름을 추출하여 모델 출력 구성
- **EAST 모델 초기화**:
    - 사전 학습된 `frozen_east_text_detection.pb` 모델 로드

### 3. **함수 정의**

- **`carROI(image)`**:
    - **차량 객체**를 감지하여 해당 영역을 반환
    - YOLO를 사용하여 객체 감지 수행
    - 감지된 '차량' 클래스만 필터링하고 **비최대 억제 적용(non-max suppression)**
- **`textROI(image)`**:
    - **번호판 텍스트**가 있을 법한 영역을 **EAST 모델**을 사용하여 탐지
        - **EAST 모델은 Efficient and Accurate Scene Text Detection의 약자**
        - **이미지에서 텍스트를 빠르고 정확하게 탐지하기 위한 딥러닝 기반 텍스트 검출 모델**
    - 감지된 텍스트 영역을 **ROI**로 반환
- **`textRead(image)`**:
    - Tesseract OCR을 통해 ROI의 텍스트를 인식
    - OCR 결과를 알파벳 및 숫자만 포함한 텍스트로 정제
- **`processROI(image)`**:
    - 이미지의 번호판 텍스트를 선명하게 하기 위한 전처리 작업
    - HSV 변환, top-hat, black-hat 연산, 블러링, 이진화 적용

### 4. **메인 코드**

- 이미지 파일 로드
- **`carROI()`**: 차량 ROI 추출 (ROI는 Region of interest의 약자)
- **`textROI()`**: 번호판 ROI 추출


<img width="455" alt="3" src="https://github.com/user-attachments/assets/31de77d7-2ffd-4658-8141-5649d961ab9d" />
<img width="417" alt="4" src="https://github.com/user-attachments/assets/b5bec913-9f53-499d-89c8-7d7db0fc9821" />


    
- **`processROI()`**: ROI 이미지 전처리

<img width="420" alt="5" src="https://github.com/user-attachments/assets/1d97f4ac-d0c5-4287-9809-c877b0f8188c" />
<img width="418" alt="6" src="https://github.com/user-attachments/assets/b9d79293-78e4-48f6-869b-8cce61915e89" />
<img width="419" alt="7" src="https://github.com/user-attachments/assets/afa08b2d-2a6d-47c4-8364-049d64bc4e01" />

    

- **`textRead()`**: OCR로 텍스트 추출
- 추출된 텍스트를 원본 이미지에 시각적으로 표시

![9](https://github.com/user-attachments/assets/8c1a2461-ab99-4a40-b016-a6597a6b9618)
<img width="692" alt="8" src="https://github.com/user-attachments/assets/a5579c98-3a52-461a-ba5e-5c7830155b0e" />


    

## ProcessROI로 번호판 영역 직접 추출, text recognition은 Tesseract


https://github.com/user-attachments/assets/cbaabb76-4f89-43b1-a3e7-2f766165d1fb




- **코드 요약 설명**
    - 차량 번호판 영역에서 OCR(광학 문자 인식)을 수행

```python
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

```

- **앞선 코드와의 차별점**
    - processROI 구현 차이
        - 새로운 코드에서는 processROI가 차량의 번호판 영역을 직접 추출한다 (앞에서는 textROI가 해당 역할)
        - 이전 코드에서는 YOLO 모델을 사용해 차량을 감지한 후 EAST를 통해 텍스트 ROI를 찾는 방식
        - 현재 코드는 이미지 처리와 컨투어를 활용하여 번호판 영역을 탐지
    - YOLO 및 EAST 모델 미사용
        - 새 코드는 YOLO 및 EAST와 같은 사전 학습된 모델을 사용하지 않고, 이미지 전처리와 컨투어 기반 탐지 방식을 사용한다
        - 경계면 및 건투어의 면적 등을 활용해 번호판 추출을 수행
    - 번호판 ROI 추출 범위 설정
        - 새 코드에서는 이미지 내부 특정 영역(상단 50%~80%, 좌우 30%~80%)을 기준으로 번호판 후보를 탐색한다
        - 이렇게 영역 축소를 통해 연산량을 줄인다
    - Tesseract 설정
        - 새 코드에서는 —psm 12 (텍스트 감지용 단일 텍스트 행)과 —oem 3 (최적의 OCR 모델 사용) 설정을 사용한다
        - 이전 코드에서는 —psm 7 (단일 텍스트 라인)과 —oem 1 (레거시와 신경망 혼합) 설정을 사용
    - 컨투어 기반 탐지
        - 컨투어를 기반으로 번호판 영역을 필터링하고, 이를 approxPolyDP로 다각형 근사화를 수행하여 사각형만 추출
    
    ## 코드 진행 순서
    
    1. 입력 이미지를 로드
    2. processROI
        1. 이미지를 HSV로 변환하고, top-hat 및 black-hat 연산으로 조명 효과 제거
        2. Gaussian Blur와 Adaptive Threshold로 번호판 후보 영역 강조
        3. 특정 영역(상단 50%~80%, 좌우 30%~80%)내에서 번호판 후보 탐색
    <img width="692" alt="10" src="https://github.com/user-attachments/assets/3e5b5c15-9ea2-4b58-81e9-0dadab7d1831" />

           
            
        4. 컨투어를 필터링하여 번호판 ROI를 추출
            <img width="421" alt="11" src="https://github.com/user-attachments/assets/f5be5bd7-b110-4933-9751-2746ef290767" />

          
            
    4. textRead
        1. 번호판 ROI에서 Tesseract OCR로 텍스트를 추출
        2. 텍스트 결과를 정제하여 알파벳 및 숫자만 남김
    5. 번호판 텍스트와 ROI를 원본 이미지에 표시
        
     <img width="696" alt="12" src="https://github.com/user-attachments/assets/6c5e5dcd-6c6c-4772-b5c8-21902100a1ca" />

        
    6. 결과를 화면에 출력
     <img width="1105" alt="13" src="https://github.com/user-attachments/assets/a439adfe-313e-475a-959d-d1fe80ad3b77" />



        

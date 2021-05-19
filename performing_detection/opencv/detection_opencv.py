import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os


def detection_with_opencv(frame):
    #load the class labels our YOLO model was trained on
    labelsPath = '../data/names/obj.names'
    LABELS = open(labelsPath).read().strip().split("\n")

    #load weights and cfg
    weightsPath = '../data/weights/' + 'crop_weed_detection.weights'
    configPath = '../data/cfg/crop_weed.cfg'

    #color selection for drawing bbox
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    # print("[INFO] loading YOLO from disk...")

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    #load our input image and grab its spatial dimensions
    # image = cv2.imread('../data/images/weed_7.jpg')
    image = frame
    # print(type(image))

    (H, W) = image.shape[:2]

    #parameters
    confi = 0.5
    thresh = 0.5
    
    #determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    #construct a blob from the input image and then perform a forward
    #pass of the YOLO object detector, giving us our bounding boxes and
    #associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    #show timing information on YOLO
    # print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    #initialize our lists of detected bounding boxes, confidences, and
    #class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    #loop over each of the layer outputs
    for output in layerOutputs:
	    #loop over each of the detections
	    for detection in output:
		    #extract the class ID and confidence (i.e., probability) of
		    #the current object detection
		    scores = detection[5:]
		    classID = np.argmax(scores)
		    confidence = scores[classID]

		    #filter out weak predictions by ensuring the detected
		    #probability is greater than the minimum probability
		    if confidence > confi:
		    	#scale the bounding box coordinates back relative to the
		    	#size of the image, keeping in mind that YOLO actually
		    	#returns the center (x, y)-coordinates of the bounding
		    	#box followed by the boxes' width and height
		    	box = detection[0:4] * np.array([W, H, W, H])
		    	(centerX, centerY, width, height) = box.astype("int")

		    	#use the center (x, y)-coordinates to derive the top and
		    	#and left corner of the bounding box
		    	x = int(centerX - (width / 2))
		    	y = int(centerY - (height / 2))

		    	#update our list of bounding box coordinates, confidences,
		    	#and class IDs
		    	boxes.append([x, y, int(width), int(height)])
		    	confidences.append(float(confidence))
		    	classIDs.append(classID)

    #apply non-maxima suppression to suppress weak, overlapping bounding
    #boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)

    #ensure at least one detection exists
    if len(idxs) > 0:
        #loop over the indexes we are keeping
        for i in idxs.flatten():
            #extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            #draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
            print(text+"center is "+str(x+h/2)+","+str(y+h/2))
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    else:
        print("No target: Weed or crop!!!")

    # plt.figure(figsize=(12,8))
    # plt.imshow(image)
    # pylab.show()
    return image

# detection_with_opencv()

def detection_with_camera():
    cv2.namedWindow('Window', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    cap = cv2.VideoCapture(0)
    print('摄像头是否开启： {}'.format(cap.isOpened()))
    # 显示缓存数
    print(cap.get(cv2.CAP_PROP_BUFFERSIZE))
    # 设置缓存区的大小
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    #调节摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    #设置FPS
    print('setfps', cap.set(cv2.CAP_PROP_FPS, 25))
    print(cap.get(cv2.CAP_PROP_FPS))

    while(True):
    #逐帧捕获
        ret, frame = cap.read()  #第一个参数返回一个布尔值（True/False），代表有没有读取到图片；第二个参数表示截取到一帧的图片
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        temp = detection_with_opencv(frame)
        # print(type(frame))
        cv2.imshow('Window', temp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #当一切结束后，释放VideoCapture对象
    cap.release()
    cv2.destroyAllWindows()
    

detection_with_camera()
# detection_with_opencv()
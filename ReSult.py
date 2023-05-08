import cv2

import time 
import posemodul as pd
cap = cv2.VideoCapture(1)
#---------
'''''
thres = 0.5
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
'''
#---------
w =800
h =800
pTime = 0
detector = pd.PoseDetector()

frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
circleCenter = (round(frameWidth/2), round(frameHeight/2))
while cap.isOpened(): 
    success, img = cap.read()
    if(success==True):
        img = cv2.resize(img, (w,h))
        img = cv2.flip(img, 1)
        # 05.匯入姿態偵測器
        
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img)
        detector.findLenght(img,frameWidth,frameHeight)
        # 02. 顯示FPS (frame rate)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (920-120, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
        '''''
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        if len(classIds) !=0:
            for classId, confidence, box in zip(classIds.flatten(),confs.flatten(), bbox):
                x,y,cx,cy = box
                cv2.circle(img, (abs(cx-x) // 2, y), 10, (255, 0, 0), 3)
                cv2.rectangle(img,box,color=(0,255,0), thickness=2)
                cv2.putText(img, classNames[classId-1],(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
                cv2.putText(img, str(confidence),(box[0]+200,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
        '''
        cv2.imshow("Image", img)
        # 01. 按Q離開
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()







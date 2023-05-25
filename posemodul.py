import imp
import cv2
import mediapipe as mp
import math
import serial
import scipy.spatial
class PoseDetector:
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.Xposition = 180
        self.Yposition = 180
        self.ser = serial.Serial('COM3', 9600, timeout=1)
        
        
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.handsModule = mp.solutions.hands
        self.distanceModule = scipy.spatial.distance
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)  
        self.face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
    def toBuzzer(self):
        self.ser.write(('c').encode('utf-8'))
    def draws(self,img,classId, confidence,box,classNames):
        '''''
        cv2.circle(img, (center_x,center_y), 10, (255, 0, 0), 3)
        cv2.circle(img, (abs(x2-x1) // 2, y1), 10, (255, 0, 0), 3)
        cv2.circle(img, (abs(x2+x1) // 2, y2), 10, (255, 0, 0), 3)
        cv2.circle(img, (x1, abs(y2-y1) // 2), 10, (255, 0, 0), 3)
        cv2.circle(img, (x2, abs(y2+y1) // 2), 10, (255, 0, 0), 3)
        cv2.line(img, (cx, cy), (abs(x2-x1) // 2, y1), (255, 255, 255), 3)
        cv2.line(img, (cx, cy), (abs(x2+x1) // 2, y2), (255, 255, 255), 3)
        cv2.line(img, (cx, cy), (x1, abs(y2-y1) // 2), (255, 255, 255), 3)
        cv2.line(img, (cx, cy), (x2, abs(y2+y1) // 2), (255, 255, 255), 3)
        cv2.putText(img, f'Length:{int(length)}', (320, 50), cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,0), 3)
        cv2.line(img, (cx, cy), (center_x, center_y), (255, 255, 255), 3)
        cv2.rectangle(img, bbox, (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        '''''
        pass
    def toSerial(self, img, cx, cy):
        rows, cols, _ = img.shape
        #print("rows:",rows,"cols:",cols)
        center_x = int(rows / 2)
        center_y = int(cols / 2)
        #print("centerX:",center_x,"centeY:",center_y)
        medium_x = int(cx)
        medium_y = int(cy)
        #print("mediumX:",medium_x,"mediumY:",medium_y)
        v=4
        m=80
        if medium_x > center_x + m:
            self.Xposition += v
            if self.Xposition>= 180:
                self.Xposition = 180

        if medium_x < center_x - m:
            self.Xposition -= v
            if self.Xposition < 0:
                self.Xposition = 0
        #######################################
        if medium_y > center_y + m:
            self.Yposition += v
            if self.Yposition>= 180:
                self.Yposition = 180


        if medium_y < center_y-m:
            self.Yposition -= v
            if self.Yposition < 0:
                self.Yposition = 0
        self.ser.write(('a'+ str(int(self.Xposition))+'b'+ str(int(self.Yposition))).encode('utf-8'))
    def findPose(self, img, draw=True):
            # 02. 取得pose資料
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.pose.process(imgRGB)
            
            if self.results.pose_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mpPose.POSE_CONNECTIONS,
                                            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())           
            return img
    def findPosition(self, img, frameWidth,frameHeight):
        # 04. 劃出方格
        self.lmList = []
        self.bboxInfo = {}
        
        
        thres = 0.5
        classNames = []
        classFile = 'coco.names'
        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')
        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'frozen_inference_graph.pb'
        net = cv2.dnn_DetectionModel(weightsPath, configPath)
        net.setInputSize(400, 400)
        net.setInputScale(1.0/ 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                self.lmList.append([id, cx, cy, cz])
            
            ad = abs(self.lmList[12][1] - self.lmList[11][1]) // 2 

            
            if (self.lmList[16][1] - ad < self.lmList[12][1] - ad):
                x1 = self.lmList[16][1] - ad    
            else:
                x1 = self.lmList[12][1] - ad    
            if (self.lmList[15][1] - ad > self.lmList[11][1] - ad):
                x2 = self.lmList[15][1] + ad    
            else:
                x2 = self.lmList[11][1] + ad    
            if (self.lmList[29][2] + ad > self.lmList[30][2] + ad):
                y2 = self.lmList[29][2] + ad      
            else:
                y2 = self.lmList[30][2] + ad
            if (self.lmList[30][2] + ad >900):
                y2 = self.lmList[26][2] + ad
                if (self.lmList[26][2]>800):
                    y2 = self.lmList[24][2] + ad
                    if (self.lmList[24][2]>750):
                        y2 = self.lmList[12][2] + ad       
            y1 = self.lmList[1][2] - ad
              
            
            if len(classIds) !=0:
                for classId, confidence,box in zip(classIds.flatten(),confs.flatten(), bbox):
                        if(classNames[classId-1],(box[0]+10,box[1]+30)):
                                        circleRadius = (box[0]+10,box[1]+30)
                                        cv2.putText(img, classNames[classId-1],(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
                                        with self.handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
                                            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                                            

                                            if results.multi_hand_landmarks != None:

                                                normalizedLandmark = results.multi_hand_landmarks[0].landmark[self.handsModule.HandLandmark.INDEX_FINGER_TIP]
                                                pixelCoordinatesLandmark = self.mpDraw._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                                                        normalizedLandmark.y,
                                                                                                                        frameWidth,
                                                                                                                        frameHeight)

                                                
                                                for i in (classNames[classId-1],(box[0]+10,box[1]+30)):
                                                    if(i == 'scissors'):
                                                        if(pixelCoordinatesLandmark < (box[0]+10,box[1]+30)):
                                                            self.toBuzzer()
                        else:
                            pass  
            #print(self.lmList)
            bbox = (x1, y1, x2 - x1, y2 - y1)
            cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + bbox[3] // 2
            self.bboxInfo = {"bbox": bbox, "center": (cx, cy)}
            # add another functions
            self.toSerial(img, cx, cy)
        else:
            print("not recognize")
        return self.lmList, self.bboxInfo

import cv2
import time 
import posemodul as pd
cap = cv2.VideoCapture(2)

w =800
h =800
pTime = 0
detector = pd.PoseDetector()


while cap.isOpened(): 
    success, img = cap.read()
    if(success==True):
        #video_out.write(img)
        img = cv2.resize(img, (w,h))
        img = cv2.flip(img, 1)
        # 05.匯入姿態偵測器
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img)
        # 02. 顯示FPS (frame rate)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (920-120, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
        cv2.imshow("Image", img)
        # 01. 按Q離開
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

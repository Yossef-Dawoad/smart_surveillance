import cv2 #########? install
import mediapipe as mp #########? install
import utils
import numpy as np
from pathlib import Path




######################### setup recoding folder
Path('./recordings').mkdir(parents=True, exist_ok=True)
recPath = Path('recordings').absolute()


##################### classes initilizations
fastfacedetect = mp.solutions.face_detection
tt = utils.TimeTracker()
fourcc = cv2.VideoWriter_fourcc(*'XVID') ##### video encoding setup,   *'mp4v' -> mp4

####################### video recording flags
RECORDING_EN = False
STOPED_DETECTION_FLAG = False
STOP_TOLERANCE_IN_SECONDS = 5
####################### Camera configrations
cap = cv2.VideoCapture(0)
ImageRead = cap.read
ColorConversion = cv2.cvtColor
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #3
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #4

########################################### adjust model result
def handel_net_result(dim):
    xmin, ymin, width, height = dim.xmin , dim.ymin, dim.width, dim.height
    width = width * WIDTH
    height = height * HEIGHT
    xmin = xmin * (128 + WIDTH) - (width/2)
    ymin = ymin * (128 + HEIGHT) - (height/2)
    return  int(xmin), int(ymin) , int(width), int(height)
############################################################## 



################################## main program
with fastfacedetect.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    FaceDetect = face_detection.process
    while cap.isOpened():
        success, image = ImageRead()
        if not success: continue ###### Ignoring empty camera frame
        # image as not writeable to pass by reference. //performance-ish Tip
        image.flags.writeable = False
        RGBimage = ColorConversion(image, cv2.COLOR_BGR2RGB)
        results = FaceDetect(RGBimage)
        if results.detections:
            ############################# Video Recorder Handler
            if not RECORDING_EN:
                RECORDING_EN =True
                videooutput = cv2.VideoWriter(str(recPath.joinpath(f"{tt.format_time()}.avi")),
                                                                    fourcc, 30.0, (WIDTH, HEIGHT))
                print('[RECODEING STARTED] face detected recording started....')
            else:
                STOPED_DETECTION_FLAG = False
            ###################################################
            faces_detected = ((res.score[0], res.location_data.relative_bounding_box) for res in results.detections)
            for score, faceRoI in faces_detected:
                xmin, ymin, width, height = handel_net_result(faceRoI) 
                faceRoI = image[ymin-50:ymin+height, xmin:xmin+width+50] ###### get the face bounding box  coordinates

        ############################ continue Video Recorder Handler
        else: ##### No Detection
            if not STOPED_DETECTION_FLAG and RECORDING_EN:
                STOPED_DETECTION_FLAG = True
                TIMER_START = tt.current_time()
            elif STOPED_DETECTION_FLAG:
                if tt.current_time() - TIMER_START >= STOP_TOLERANCE_IN_SECONDS:
                    RECORDING_EN =False
                    STOPED_DETECTION_FLAG = False
                    videooutput.release() #### stop recording that file
                    print('[RECODEING STOPED] the video has been written to the disk.')
            
        ####################################################
        if RECORDING_EN: videooutput.write(image)
        fps = tt.fps_calculate(); print(fps, end="\r") ####### display the fps 
        cv2.imshow('MediaPipe Face Detection', image) ######## cv2.flip(image, 1)
        if cv2.waitKey(3) & 0xFF == 27: break
    ############################ when session terminated by the user 
    cap.release()
    videooutput.release()
    print('[RECODEING OR CAMERA STOPED] camera terminated by the user.')
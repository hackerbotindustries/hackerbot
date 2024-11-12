#Import the necessary Packages for this software to run
import mediapipe
import cv2
import time


#Use MediaPipe to draw the hand framework over the top of hands it identifies in Real-Time
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

#Use CV2 Functionality to create a Video stream and add some values
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)

prev_frame_time = 0
new_frame_time = 0

#Add confidence values and extra settings to MediaPipe hand tracking. As we are using a live video stream this is not a static
#image mode, confidence values in regards to overall detection and tracking and we will only let two hands be tracked at the same time
#More hands can be tracked at the same time if desired but will slow down the system
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:

#Create an infinite loop which will produce the live feed to our desktop and that will search for hands
     while True:
           ret, frame = cap.read()
           
           #Produces the hand framework overlay ontop of the hand, you can choose the colour here too)
           results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
           
           #In case the system sees multiple hands this if statment deals with that and produces another hand overlay
           if results.multi_hand_landmarks != None:
              for handLandmarks in results.multi_hand_landmarks:
                  drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
                  
                  #Below is Added Code to find and print to the shell the Location X-Y coordinates of Index Finger, Uncomment if desired
                  for point in handsModule.HandLandmark:
                      
                      normalizedLandmark = handLandmarks.landmark[point]
                      pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, 640, 480)
                      
                      #Using the Finger Joint Identification Image we know that point 8 represents the tip of the Index Finger
                      if point == 8:
                          print(point)
                          print(pixelCoordinatesLandmark)
                          print(normalizedLandmark)
            


           new_frame_time = time.time()
           fps = 1/(new_frame_time - prev_frame_time)
           prev_frame_time = new_frame_time
           fps = int(fps)
           fps = "FPS: " + str(fps)
           cv2.putText(frame, fps, (7, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, 2)

           #Below shows the current frame to the desktop 
           cv2.imshow("Hand Pose Tracker", frame);
           key = cv2.waitKey(1) & 0xFF
           
           #Below states that if the |q| is press on the keyboard it will stop the system
           if key == ord("q"):
              break


import time
import cv2
import numpy as np
import mediapipe as mp

def createLandmarker():
   # landmarker object
   landmarker = mp.tasks.vision.HandLandmarker

   # callback function options
   # print, from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream
   def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
      print('hand landmarker result: {}'.format(result.handedness))
   # next is to add a callback that'll draw on the video stream

   # options for running it
   options = mp.tasks.vision.HandLandmarkerOptions( 
      base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"), # path to model
      running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
      num_hands = 2, # track both hands
      min_hand_detection_confidence = 0.5, # default value
      min_hand_presence_confidence = 0.5, # default value
      min_tracking_confidence = 0.5, # default value
      result_callback=print_result)
   
   # create and return landmarker object
   return landmarker.create_from_options(options)

def main():
   # access webcam
   cap = cv2.VideoCapture(0)

   # create landmarker
   hand_landmarker = createLandmarker()


   # open landmarker and stream data
   with hand_landmarker as landmarker:
      while True:
         ret, frame = cap.read()
         frame = cv2.flip(frame, 1)
         # convert frame to mp image file
         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
         # process image
         landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

         cv2.imshow('frame',frame)
         if cv2.waitKey(1) == ord('q'):
            break
   
   # release everything
   cap.release()
   cv2.destroyAllWindows()

if __name__ == "__main__":
   main()

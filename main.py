
import time
import cv2
import numpy as np
import mediapipe as mp
# for visualizing results
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

result_landmarks = mp.tasks.vision.HandLandmarkerResult

def createLandmarker():
   # landmarker object
   landmarker = mp.tasks.vision.HandLandmarker

   # callback function options
   # print, from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream
   def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
      print('hand landmarker result: {}'.format(result))

   # next is to add a callback that'll draw on the video stream
   def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
      global result_landmarks
      result_landmarks = result
   
   # options for running it
   options = mp.tasks.vision.HandLandmarkerOptions( 
      base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"), # path to model
      running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
      num_hands = 2, # track both hands
      min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
      min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
      min_tracking_confidence = 0.3, # lower than value to get predictions more often
      result_callback=update_result)
   
   # create and return landmarker object
   return landmarker.create_from_options(options)

def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
   """Courtesy of https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
   try:
      if detection_result.hand_landmarks == []:
         return rgb_image
      else:
         hand_landmarks_list = detection_result.hand_landmarks
         handedness_list = detection_result.handedness
         annotated_image = np.copy(rgb_image)

         # Loop through the detected hands to visualize.
         for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
               landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            solutions.drawing_utils.draw_landmarks(
               annotated_image,
               hand_landmarks_proto,
               solutions.hands.HAND_CONNECTIONS,
               solutions.drawing_styles.get_default_hand_landmarks_style(),
               solutions.drawing_styles.get_default_hand_connections_style())

            # # Get the top left corner of the detected hand's bounding box.
            # height, width, _ = annotated_image.shape
            # x_coordinates = [landmark.x for landmark in hand_landmarks]
            # y_coordinates = [landmark.y for landmark in hand_landmarks]
            # text_x = int(min(x_coordinates) * width)
            # text_y = int(min(y_coordinates) * height) - 10

            # # Draw handedness (left or right hand) on the image.
            # cv2.putText(annotated_image, f"{handedness[0].category_name}",
            #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
            #             2, (0,0,255), 3, cv2.LINE_AA)

         return annotated_image
   except:
      return rgb_image

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
         # draw landmarks on frame
         frame = draw_landmarks_on_image(frame,result_landmarks)

         cv2.imshow('frame',frame)
         if cv2.waitKey(1) == ord('q'):
            break
   
   # release everything
   cap.release()
   cv2.destroyAllWindows()

if __name__ == "__main__":
   main()

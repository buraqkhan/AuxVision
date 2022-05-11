import argparse
import sys
import time
import cv2

from .object_detector import ObjectDetector
from .object_detector import ObjectDetectorOptions
from . import utils


def tf_run(model = 'obj_detect_pi/efficientdet_lite0.tflite', camera_id = 0, width = 550, height = 370, num_threads = 4,
        enable_edgetpu = False):
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """
  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  options = ObjectDetectorOptions(
      num_threads=num_threads,
      score_threshold=0.3,
      max_results=3,
      enable_edgetpu=enable_edgetpu)
  detector = ObjectDetector(model_path=model, options=options)

  # Continuously capture images from the camera and run inference
  coordinates = []
  while len(coordinates) == 0:
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    counts = {}
    # Run object detection estimation using the model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect(rgb_image)

    if(len(detections) > 0):
      for i in range(0, len(detections)):
        # print(detections[i].bounding_box.left)
        # print(detections[i].categories[0].label)
        temp = detections[i].bounding_box
        coordinates.append((temp.left, temp.top, temp.right, temp.bottom, detections[i].categories[0].label))  

    print(coordinates)
    # Draw keypoints and edges on input image
    image = utils.visualize(image, detections)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()
  
  return coordinates


# def main():
  # tf_run()
#   parser = argparse.ArgumentParser(
#       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#   parser.add_argument(
#       '--model',
#       help='Path of the object detection model.',
#       required=False,
#       default='efficientdet_lite0.tflite')
#   parser.add_argument(
#       '--cameraId', help='Id of camera.', required=False, type=int, default=0)
#   parser.add_argument(
#       '--frameWidth',
#       help='Width of frame to capture from camera.',
#       required=False,
#       type=int,
#       default=640)
#   parser.add_argument(
#       '--frameHeight',
#       help='Height of frame to capture from camera.',
#       required=False,
#       type=int,
#       default=480)
#   parser.add_argument(
#       '--numThreads',
#       help='Number of CPU threads to run the model.',
#       required=False,
#       type=int,
#       default=4)
#   parser.add_argument(
#       '--enableEdgeTPU',
#       help='Whether to run the model on EdgeTPU.',
#       action='store_true',
#       required=False,
#       default=False)
#   args = parser.parse_args()
# 
  # run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      # int(args.numThreads), bool(args.enableEdgeTPU))

# 
# if __name__ == '__main__':
#   main()

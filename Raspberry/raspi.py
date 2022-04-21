from base64 import encode
import socket
import cv2
import pickle
import struct
import pyttsx3
HOST = "192.168.100.70"  # The server's hostname or IP address
PORT = 65430  # The port used by the server

# Setting voice engine properties
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', 140)
voices = engine.getProperty('voices')
engine.setProperty('voice', "english")
images_sent = 0

# cap_right = cv2.VideoCapture(0)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),90]

pic_to_send = "right"
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    while True:
        try:
            s.connect((HOST, PORT))
            while True:
                cap_right = cv2.VideoCapture(0)                    
                cap_left =  cv2.VideoCapture(2)
                frame_left = None
                frame_right = None
                ret_left,frame_left = cap_left.read()
                cap_left.release()
                ret_right,frame_right = cap_right.read()
                cap_right.release()
                if ret_left == False:
                    print("Left camera didnt read!")
                    continue
                elif ret_right == False:
                    print("Right camera didnt read!")
                    continue
                _,frame_left = cv2.imencode('.jpg',frame_left,encode_param)
                _,frame_right = cv2.imencode('.jpg',frame_right,encode_param)
                data_left = pickle.dumps(frame_left,0)
                data_right = pickle.dumps(frame_right,0)
                size_left = len(data_left)
                size_right = len(data_right)


                s.sendall(struct.pack(">L",size_right) + data_right)
                right_confirmation = s.recv(1024)
                print("Right image sent.")


                s.sendall(struct.pack(">L",size_left) + data_left)
                left_confirmation = s.recv(1024)
                print("Left image sent.")

                message = None
                message = s.recv(1024)
                message = message.decode('UTF-8')
                print("Message received :", message)

                if message != "left" and message != "right":
                    if message.count(',') == 1:
                        engine.say(message)
                        engine.runAndWait()
                        # engine.stop()
                    else:
                        messages = message.split(',')
                        for message in messages:
                            if message == "":
                                break
                            print("HELLO! : ", message)
                            engine.say(message)
                            engine.runAndWait()
                            # engine.stop()

                if message == " end":
                    break

            s.close()
        except:
            print("Waiting to connect...")
        
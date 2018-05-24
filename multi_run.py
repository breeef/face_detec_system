import numpy as np
import cv2
import sys
from time import time
import socket
import winsound

import kcftracker
import face
import requests
import json
from multiprocessing import Pool
import flask
from signin import *
selectingObject = False
initTracking = True
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 1
duration = 0.01
face_recog = face.Recognition(1)

def face_recognition(faceroi):

    faces = face_recog.identify(faceroi)

    return faces

def board_detect(faces):
    url = 'http://127.0.0.1:5000'
    headers = {
        'content-type': 'application/json'
    }
    for face in faces:
        print(face.name)
        data = {
            'name': face.name
        }
        requests.post(url, headers=headers, data=json.dumps(data))

def add_overlays(frame, faces):
    if faces is not None:
        for face in faces:
            store_board_time(face.name,10)
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                if face.name == -1:
                    face.name = "stranger"
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

    video_source = 0



if __name__ == '__main__':


    # if len(sys.argv) == 1:
    #     cap = cv2.VideoCapture(0)
    # elif len(sys.argv) == 2:
    #     if sys.argv[1].isdigit():  # True if sys.argv[1] is str of a nonnegative integer
    #         cap = cv2.VideoCapture(int(sys.argv[1]))
    #     else:
    #
    #         cap = cv2.VideoCapture(sys.argv[1])
    #         inteval = 30
    # else:
    #     assert 0, "too many arguments"

    # tracker = kcftracker.KCFTracker(True, True, True)  # hog, fixed_window, multiscale
    # if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.
    cap = cv2.VideoCapture(0)
    
    #cv2.setMouseCallback('tracking',draw_boundingbox)
    color = (0, 255, 0)
    count_faces = None
    tracker = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if(frame_count %10 == 0):
            initTracking = True
            onTracking = False
        if initTracking:
            tracker = []
            faces = face_recognition(frame)

            if len(faces) > 0:
                print("ok")
                for face_index in range(len(faces)):
                    temp = kcftracker.KCFTracker(True, True, True)
                    tracker.append(temp)
                    temp.init([faces[face_index].bounding_box[0], faces[face_index].bounding_box[1], faces[face_index].bounding_box[2]-faces[face_index].bounding_box[0], faces[face_index].bounding_box[3] - faces[face_index].bounding_box[1]], frame)

              
                add_overlays(frame, faces)
                count_faces = faces
                initTracking = False
                onTracking = True

        elif onTracking:
           
            for i in range(len(tracker)):
                boundingbox = tracker[i].update(frame)
                if boundingbox == [0., 0., 0., 0.]:
                    initTracking = True
                    onTracking = False
                else:

                    boundingbox = list(map(int, boundingbox))
                    if count_faces[i] is not None:
                        count_faces[i].bounding_box[0] = boundingbox[0]
                        count_faces[i].bounding_box[1] = boundingbox[1]
                        count_faces[i].bounding_box[2] = boundingbox[0] + boundingbox[2]
                        count_faces[i].bounding_box[3] = boundingbox[1] + boundingbox[3]
                       
            add_overlays(frame, count_faces)
            
            
           
           
        cv2.imshow('tracking', frame)
       
        c = cv2.waitKey(inteval) & 0xFF
        if c == 27 or c == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

import numpy as np
import cv2
import sys
from time import time

import kcftracker
import face

selectingObject = False
initTracking = True
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 1
duration = 0.01
face_recog = face.Recognition()

def face_recognition(faceroi):

    faces = face_recog.identify(faceroi)

    return faces


def add_overlays(frame, face):
    if face is not None:

        face_bb = face.bounding_box.astype(int)
        cv2.rectangle(frame,
                      (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                      (0, 255, 0), 2)
        if face.name is not None:
            if face.name == -1:
                face.name = "stronger"
            cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)


if __name__ == '__main__':


    if len(sys.argv) == 1:
        cap = cv2.VideoCapture(0)
    elif len(sys.argv) == 2:
        if sys.argv[1].isdigit():  # True if sys.argv[1] is str of a nonnegative integer
            cap = cv2.VideoCapture(int(sys.argv[1]))
        else:
            cap = cv2.VideoCapture(sys.argv[1])
            inteval = 30
    else:
        assert 0, "too many arguments"

    tracker = kcftracker.KCFTracker(True, True, True)  # hog, fixed_window, multiscale
    # if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.

    cv2.namedWindow('tracking')
    #cv2.setMouseCallback('tracking',draw_boundingbox)
    color = (0, 255, 0)
    count_faces = None
    while cap.isOpened():
        print("ok")
        ret, frame = cap.read()
        if not ret:
            break

        if initTracking:
            t0=time()
            faces = face_recognition(frame)
            t1=time()
            us=t1-t0
            #print(1/(us))
            if len(faces) > 0:
                print("ok")
                if len(faces) ==1:
                    #cv2.rectangle(frame, (face.bounding_box[0] - 10, face.bounding_box[1] - 10), (face.bounding_box[2] + 10, face.bounding_box[3] + 10), color, 2)
                    add_overlays(frame, faces[0])
                    face =faces[0]
                    tracker.init([face.bounding_box[0], face.bounding_box[1], face.bounding_box[2]-face.bounding_box[0], face.bounding_box[3] - face.bounding_box[1]], frame)
                    count_faces = faces
                initTracking = False
                onTracking = True

        elif onTracking:
            t0 = time()
            boundingbox = tracker.update(frame)
            if boundingbox == [0., 0., 0., 0.]:
                initTracking = True
                onTracking = False
            else:
                t1 = time()
                boundingbox = list(map(int, boundingbox))
                if count_faces is not None:
                    if len(count_faces) ==1:
                        count_faces[0].bounding_box[0] = boundingbox[0]
                        count_faces[0].bounding_box[1] = boundingbox[1]
                        count_faces[0].bounding_box[2] = boundingbox[0] + boundingbox[2]
                        count_faces[0].bounding_box[3] = boundingbox[1] + boundingbox[3]
                        print(count_faces[0].bounding_box)
                        add_overlays(frame, count_faces[0])
                duration = 0.8 * duration + 0.2 * (t1 - t0)
                # duration = t1-t0
                cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(inteval) & 0xFF
        if c == 27 or c == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

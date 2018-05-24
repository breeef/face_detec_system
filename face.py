# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import os

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

import detect_face
import facenet


gpu_memory_fraction = 0.3
facenet_model_checkpoint = "20170512-110547.pb"
classifier_model = "myclassf1.pkl"
debug = False
face_list = os.listdir(os.path.dirname(__file__) + "/face_lib")
face_lib = []
for file in face_list:
    if os.path.splitext(file)[1] == '.npy':
        face_lib.append(file)

class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Recognition:
    def __init__(self, detect_method=1):

        self.detect = CvDetection()
        self.encoder = Encoder()
        self.identifier = Identifier()

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return face

    def identify(self, image):
        faces = self.detect.find_faces(image)
        if faces :
            faces_embeddings = self.encoder.generate_embedding(faces)
            for i, face_embedding in enumerate(faces_embeddings):
                faces[i].embedding = face_embedding
                faces[i].name = self.identifier.identify(face_embedding)
                print(faces[i].name)
        return faces



class Identifier:
    # def __init__(self):
    #     with open(classifier_model, 'rb') as infile:
    #         self.model, self.class_names = pickle.load(infile)
    #         print(self.model)
    #         print(self.class_names)

    def cosine_dist(self, face_feature, face_lib):
        dist_list = []
        for i, face_npy in enumerate(face_lib):
           # print(face_npy)
            face = np.load(os.path.join("./face_lib", face_npy))
            #print(face)
            #print(face_feature.shape)
            dist1 = 1 - np.dot(face_feature, face) / (np.linalg.norm(face_feature) * np.linalg.norm(face))
            dist_list.append(dist1)
        return dist_list

    def identify(self, face_embedding):
        # if face.embedding is not None:
        #     predictions = self.model.predict_proba([face.embedding])
        #
        #     best_class_indices = np.argmax(predictions, axis=1)
        #     return self.class_names[best_class_indices[0]]
        if face_embedding is not None:
            cos_list = self.cosine_dist(face_embedding, face_lib)
            print(cos_list)
            best_class_indices = np.argmin(cos_list, axis=0)
            if cos_list[best_class_indices] < 0.3:
                return face_lib[best_class_indices].split('.')[0]
            else:
                return -1

class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        if isinstance(face, list):
            #print(len(face))
            prewhiten_face=[]
            for i, face in enumerate(face):

                prewhiten_face.append(facenet.prewhiten(face.image))

            feed_dict = {images_placeholder: prewhiten_face, phase_train_placeholder: False}

            return self.sess.run(embeddings, feed_dict = feed_dict)

        else:
            prewhiten_face = facenet.prewhiten(face.image)
            #Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
            return self.sess.run(embeddings, feed_dict=feed_dict)[0]

def crop_region(image, boundingbox, face_crop_margin, face_crop_size):
    face = Face()
    face.container_image = image
    face.bounding_box = np.zeros(4, dtype=np.int32)

    img_size = np.asarray(image.shape)[0:2]
    face.bounding_box[0] = np.maximum(boundingbox[0] - face_crop_margin / 2, 0)
    face.bounding_box[1] = np.maximum(boundingbox[1] - face_crop_margin / 2, 0)
    face.bounding_box[2] = np.minimum(boundingbox[2] + face_crop_margin / 2, img_size[1])
    face.bounding_box[3] = np.minimum(boundingbox[3] + face_crop_margin / 2, img_size[0])
    cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
    face.image = misc.imresize(cropped, (face_crop_size, face_crop_size), interp='bilinear')

    return face

class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        bounding_boxes, _ = detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)

        for bb in bounding_boxes:
            face = crop_region(image, bb, self.face_crop_margin, self.face_crop_size)
            faces.append(face)

        return faces


class CvDetection():
    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.model_path = "haarcascade_frontalface_alt2.xml"
        self.scaleFactor = 1.2
        self.minNeighbors = 3
        self.minSize = (32,32)
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def find_faces(self, image):
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        classfier = cv2.CascadeClassifier(self.model_path)

        faceRects = classfier.detectMultiScale(grey, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors, minSize=self.minSize)
        faces = []
        for facerect in faceRects:
            x, y, w, h = facerect
            bb = [x, y, x+w, y+h]
            #print(bb)
            face =crop_region(image, bb, self.face_crop_margin, self.face_crop_size)
            faces.append(face)

        return faces
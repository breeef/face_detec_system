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

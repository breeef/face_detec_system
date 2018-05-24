#!/usr/bin/env python
# -*- coding: utf-8 -*-
# start.py
from flask import Flask, session, request, render_template, jsonify, redirect,url_for,flash
# from flask import redirect, url_for, escape, make_response
from PIL import Image
from flask_login import LoginManager
from signin import *
import numpy as np
import face
import sys
import cv2 ##opencv
import msvcrt
import threading
import time
login_manager = LoginManager()

app = Flask(__name__)



@app.route('/edit-avatar', methods=['GET', 'POST'])
def change_avatar():
    status = 'fail'
    file = request.files['file']
    filename = file.filename
    img = Image.open(file)
    im = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    if file :
        recognition = face.Recognition(1)
        print(filename)
        img.save(os.path.join('C:\\Users\\xiaomiao\\Desktop\\facenet\\face_lib','%s.png'%filename))
        save_face = recognition.add_identity(im, filename)
        if save_face:
            cv2.imwrite("C:\\Users\\xiaomiao\\Desktop\\facenet\\face_lib\\%s.jpg"%filename, im)
            np.save("C:\\Users\\xiaomiao\\Desktop\\facenet\\face_lib\\%s.npy" % filename, save_face.embedding)
            status = 'success'
    return render_template('success.html',data = status ,people = filename)
@app.route('/edit', methods=['GET', 'POST'])
def edit():
    return render_template('put_pic.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    data = search('maomao')
    if  data:
        data1 = '是'
    else:
        data1 = '否'
    return render_template('index.html',data = data1)



if __name__ == '__main__':

    app.run(host='0.0.0.0', port=9000)

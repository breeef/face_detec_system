from PIL import Image
from flask import Flask, session, request, render_template, jsonify, redirect, url_for, flash




@app.route('/edit-avatar', methods=['GET', 'POST'])
def change_avatar():
    if request.method == 'POST':
        file = request.files['file']
        im = Image.open(file)
        if file :
            filename = file.filename
            print(filename)
            im.save(os.path.join('/home/shi/','%s.png'%filename))
    return render_template('put_pic.html')

@app.route('/edit', methods=['GET', 'POST'])
def edit():
    return render_template('put_pic.html')




@app.route('/edit-avatar', methods=['GET', 'POST'])
def change_avatar():
    if request.method == 'POST':
        file = request.files['file']
        im = Image.open(file)
        if file :
            filename = file.filename
            print(filename)
            im.save(os.path.join('/home/shi/','%s.png'%filename))
    return render_template('put_pic.html')

@app.route('/edit', methods=['GET', 'POST'])
def edit():
    return render_template('put_pic.html')

from flask import Flask, render_template, send_from_directory, Response
from pathlib import Path
from camera import Camera

import argparse

camera = Camera(20,0)
camera.run()

app = Flask(__name__)

@app.after_request
def add_header(r):
	"""
	Add headers to both force latest IE rendering or Chrome Frame,
	and also to cache the rendered page for 10 minutes
	"""
	r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
	r.headers["Pragma"] = "no-cache"
	r.headers["Expires"] = "0"
	r.headers["Cache-Control"] = "public, max-age=0"
	return r

@app.route("/")
def entrypoint():
	return render_template("index.html")

@app.route("/r")
def capture():
	im = camera.get_frame(_bytes=False)
	capture_and_save(im)
	return render_template("send_to_init.html")

def last_frame(camera):
        frame = camera.get_frame()
        return frame
        return (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

@app.route("/image.jpg")
def last_frame0():
	return Response(last_frame(camera),mimetype="multipart/x-mixed-replace; boundary=frame")



def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
			   b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

@app.route("/video_feed0")
def video_feed0():
	return Response(gen(camera),mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__=="__main__":
	app.run(host='0.0.0.0',port=80)

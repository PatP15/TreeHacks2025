#!/usr/bin/env python3
import cv2
import time
import threading
import subprocess
import random
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
from datetime import datetime
from PIL import Image

##########################################
# 1) INITIALIZE FLASK
##########################################
app = Flask(__name__)

##########################################
# 2) VIDEO STREAMING SETUP (Using Video Files)
##########################################
video_camera_path = "demo_vids/camera.mp4"
video_realtimekin_path = "demo_vids/kin.mov"

cap_camera = cv2.VideoCapture(video_camera_path)
cap_realtimekin = cv2.VideoCapture(video_realtimekin_path)
fps_kin = cap_realtimekin.get(cv2.CAP_PROP_FPS)
skip_frames_kin = int(4 * fps_kin)
cap_realtimekin.set(cv2.CAP_PROP_POS_FRAMES, skip_frames_kin)

total_frames_kin = int(cap_realtimekin.get(cv2.CAP_PROP_FRAME_COUNT))
end_frame_kin = total_frames_kin - int(4 * fps_kin)


def gen_frames(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        cv2.waitKey(int(1000 // fps))
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

##########################################
# 3) FLASK ROUTES
##########################################

@app.route('/')
def home():
    return redirect(url_for('platoon'))

@app.route('/platoon')
def platoon():
    """ Load the Platoon page (platoon.html). """
    return render_template('platoon.html')

@app.route('/messages')
def messages():
    """ Load the Messages page (messages.html). """
    return render_template('messages.html')

@app.route('/video_feed')
def video_feed():
    """ Streams the camera feed (video file). """
    return Response(gen_frames(cap_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/realtimekin_feed')
def realtimekin_feed():
    """ Streams the RealTime Kin feed (video file). """
    return Response(gen_frames(cap_realtimekin),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

##########################################
# 4) MAP TRACKING & PPG PROCESSING
##########################################

@app.route('/location.json')
def location_json():
    """ Returns the main (green) location and five red flashing dots. """
    return jsonify({
        "green": { "x": 0.7, "y": 0.6 },
        "red": [
            {"x": 0.7, "y": 0.8},
            {"x": 0.8, "y": 0.7},
            {"x": 0.3, "y": 0.2},
            {"x": 0.21, "y": 0.3},
            {"x": 0.4, "y": 0.21}
        ]
    })

def process_ppg_data(file_path):
    """ Extract and normalize PPG signal from CSV file. """
    df = pd.read_csv(file_path)
    ppg_signal = df["IR"].values
    ppg_signal_normalized = (ppg_signal - ppg_signal.min()) / (ppg_signal.max() - ppg_signal.min())
    return pd.DataFrame({"timestamp": df["sample_index"].values, "ppg_signal": ppg_signal_normalized})

@app.route('/ppg_data')
def ppg_data():
    """ Returns either Healthy or AFib PPG data based on request. """
    file_type = request.args.get("type", "healthy")
    file_path = "regular_2.csv" if file_type == "healthy" else "afib_2.csv"
    ppg_df = process_ppg_data(file_path)
    return jsonify(ppg_df["ppg_signal"].tolist())

##########################################
# 5) AFib ALERT SYSTEM
##########################################

@app.route('/run_script')
def run_script():
    """ Runs a background Python script when AFib is detected. """
    subprocess.Popen(["python3", "/mnt/data/test_guardian.py"])
    return jsonify({"status": "Script triggered"})

##########################################
# 6) CHATBOT (FAKE RESPONSE)
##########################################

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """ Fake chatbot: echoes user message with a timestamp. """
    data = request.get_json()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time.sleep(1)  # Simulate processing time
    return jsonify({"response": f"Received message on {now}"})

##########################################
# 7) START THE APP
##########################################

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

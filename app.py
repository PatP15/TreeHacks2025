#!/usr/bin/env python3
import cv2
import time
import threading
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
# import torch
# from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
from datetime import datetime



##########################################
# 1) INITIALIZE FLASK
##########################################
app = Flask(__name__)


##########################################
# 3) CAMERA SETUP
##########################################
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit(1)


##########################################
# 6) MJPEG STREAM GENERATOR
##########################################
def gen_frames():
    """
    Yields horizontally flipped MJPEG frames for the browser.
    Updates global 'latest_frame' and 'latest_frame_id' each time.
    """
    global latest_frame, latest_frame_id

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

##########################################
# 7) FLASK ROUTES
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
    """ Live camera stream (MJPEG). """
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/realtimekin_feed')
def realtimekin_feed():
    """ Streams the RealTime Kin feed (video file). """
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

import random

@app.route('/location.json')
def location_json():
    """ Returns the main (green) location and five red flashing dots around a home position. """
    return jsonify({
        "green": { "x": 0.7, "y": 0.6 },
        "red": [
            {"x": 0.7, "y": 0.8},  # Center position
            {"x": 0.8, "y": 0.7},
            {"x": 0.3, "y": 0.2},
            {"x": 0.21, "y": 0.3},
            {"x": 0.4, "y": 0.21}
        ]
    })
import numpy as np
from scipy.signal import find_peaks
import pandas as pd

def process_ppg_data(file_path):
    """
    Processes the original CSV file to extract and normalize the PPG (IR) signal.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame with 'timestamp' and 'ppg_signal' (normalized).
    """
    import pandas as pd

    # Load CSV file
    df = pd.read_csv(file_path)

    # Extract IR signal (assuming it's the best for PPG analysis)
    ppg_signal = df["IR"].values

    # Normalize PPG signal between 0 and 1 for smoother visualization
    ppg_signal_normalized = (ppg_signal - ppg_signal.min()) / (ppg_signal.max() - ppg_signal.min())

    # Create a cleaned DataFrame
    ppg_df = pd.DataFrame({
        "timestamp": df["sample_index"].values,  # Keeping original timestamps
        "ppg_signal": ppg_signal_normalized  # Normalized PPG signal
    })

    return ppg_df

import subprocess

@app.route('/ppg_data')
def ppg_data():
    """ Returns either healthy or AFib PPG data based on request. """
    file_type = request.args.get("type", "healthy")  # Default to healthy
    file_path = "regular_2.csv" if file_type == "healthy" else "afib_2.csv"
    
    # Load the selected PPG dataset
    ppg_df = process_ppg_data(file_path)
    return jsonify(ppg_df["ppg_signal"].tolist())

@app.route('/run_script')
def run_script():
    """ Runs a background Python script when AFib is detected. """
    subprocess.Popen(["python3", "test_guardian.py"])
    return jsonify({"status": "Script triggered"})

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """ Fake chatbot: echoes user message. """
    data = request.get_json()
    # msg = data.get('message', '')
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time.sleep(1)  # Simulate processing time
    return jsonify({"response": f"Received message on {now}"})

##########################################
# 8) START THE APP & THREAD
##########################################
if __name__ == '__main__':
    # threading.Thread(target=caption_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5001, debug=True)

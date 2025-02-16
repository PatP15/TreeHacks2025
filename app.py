#!/usr/bin/env python3
import cv2
import time
import threading
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, Response, redirect, url_for
from io import BytesIO
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

#############################################
# 1) INITIALIZE FLASK
#############################################
app = Flask(__name__)

#############################################
# 2) LOAD SmolVLM MODEL
#############################################
DEVICE = "cuda" if torch.cuda.is_available() else "mps"

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)
model.eval()

print(f"Loaded SmolVLM on device: {DEVICE}")

#############################################
# 3) CAMERA SETUP
#############################################
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit(1)

# Global variables for skipping frames while busy
latest_frame = None         
latest_frame_id = 0         
processing_frame_id = 0     
current_caption = "Initializing..."
model_busy = False          

#############################################
# 4) LOAD HEART RATE DATA FROM CSV
#############################################

# Load CSV containing heart rate data
csv_file = "afib_2.csv"  # Change to "regular_2.csv" if needed
df = pd.read_csv(csv_file)  
heart_rate_data = df["IR"].values  # Use IR channel for heart rate signal

# Parameters for "live" stream
num_samples = len(heart_rate_data)  
window_size = 100  # Number of data points shown at a time (e.g., last 2–5 sec)
current_index = 0  # Track position in dataset

def generate_live_heart_trace():
    """
    Loops through heart rate data, showing a rolling ECG-like trace.
    """
    global current_index

    while True:
        start_idx = current_index
        end_idx = start_idx + window_size
        if end_idx > num_samples:
            end_idx = num_samples

        data_slice = heart_rate_data[start_idx:end_idx]

        # Generate ECG-like plot with hospital-style colors
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(data_slice, color="#00FF00", linewidth=2)  # Hospital ECG green
        ax.set_ylim([np.min(heart_rate_data) - 5, np.max(heart_rate_data) + 5])  
        ax.set_facecolor("black")  # Black background like hospital monitors
        ax.spines['bottom'].set_color('#00FF00')  # Make x-axis green
        ax.spines['left'].set_color('#00FF00')  # Make y-axis green
        ax.tick_params(axis='x', colors='#00FF00')
        ax.tick_params(axis='y', colors='#00FF00')
        ax.axis("off")  # Hide axis for a clean ECG-style look

        # Convert to PNG
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", facecolor="black")
        buffer.seek(0)
        plt.close(fig)

        # Update index (loop back if necessary)
        current_index = (current_index + 1) % (num_samples - window_size)

        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + buffer.getvalue() + b'\r\n')

#############################################
# 5) STREAM HEART TRACE IN FLASK
#############################################

@app.route('/heart_trace')
def heart_trace():
    """ Streams the simulated heart rate graph as an image. """
    return Response(generate_live_heart_trace(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#############################################
# 6) STREAM CAMERA FEED
#############################################
def gen_frames():
    """
    Yields horizontally flipped MJPEG frames for the browser.
    """
    global latest_frame, latest_frame_id

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)  # Flip horizontally

        latest_frame = frame
        latest_frame_id += 1

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """ Live camera stream (MJPEG). """
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#############################################
# 7) FLASK ROUTES
#############################################

@app.route('/')
def home():
    return redirect(url_for('platoon'))

@app.route('/platoon')
def platoon():
    return render_template('platoon.html')

@app.route('/medical')
def medical():
    return render_template('medical.html')

@app.route('/messages')
def messages():
    return render_template('messages.html')

#############################################
# 8) START FLASK APP & BACKGROUND THREAD
#############################################
if __name__ == '__main__':
    threading.Thread(target=generate_live_heart_trace, daemon=True).start()
    app.run(host='0.0.0.0', port=5001, debug=True)

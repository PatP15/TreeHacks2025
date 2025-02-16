#!/usr/bin/env python3
import cv2
import time
import threading
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from datetime import datetime

##########################################
# 1) INITIALIZE FLASK
##########################################
app = Flask(__name__)

##########################################
# 2) LOAD SmolVLM MODEL
##########################################
DEVICE = "cuda" if torch.cuda.is_available() else "mps"

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)
model.eval()

print(f"Loaded SmolVLM on device: {DEVICE}")

##########################################
# 3) CAMERA SETUP
##########################################
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

##########################################
# 4) SMOLVLM CAPTION FUNCTION
##########################################
def generate_caption(opencv_frame):
    """
    Convert an OpenCV BGR frame to a PIL image, run SmolVLM,
    and return the generated text.
    """
    rgb = cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Can you describe this image?"}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[pil_img], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)

    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts[0] if generated_texts else "No caption generated."

##########################################
# 5) BACKGROUND CAPTION THREAD
##########################################
def caption_loop():
    global latest_frame_id, processing_frame_id, latest_frame
    global model_busy, current_caption

    while True:
        if (latest_frame_id != processing_frame_id) and (not model_busy):
            model_busy = True
            frame_to_process = latest_frame.copy()

            caption = generate_caption(frame_to_process)

            # remove anything before Assistant: in the caption
            caption = caption.split("Assistant: ")[1]
            
            # and remove everything after the final period out of the paragraph
            caption = caption.split(".")[:-1]
            caption = ".".join(caption) + "."

            current_caption = caption

            
            processing_frame_id = latest_frame_id
            model_busy = False

        # time.sleep(0.1)

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

        latest_frame = frame
        latest_frame_id += 1

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

@app.route('/get_caption')
def get_caption():
    """ Returns the latest SmolVLM caption. """
    return jsonify({"caption": current_caption})

import random

@app.route('/location.json')
def location_json():
    """
    Returns a simulated user position in the form of percentages.
    (0,0) is top-left of the map, (1,1) is bottom-right.
    """
    return jsonify({
        "x": round(random.uniform(0.2, 0.8), 2),  # Random position for demo
        "y": round(random.uniform(0.2, 0.8), 2)
    })

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
    threading.Thread(target=caption_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5001, debug=True)

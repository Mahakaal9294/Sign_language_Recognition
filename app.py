# importing the lobraries
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import preprocessing.image_processing as cm
import time
from joblib import load
import pyttsx3


# Loading the models
model = load('final_model/rf_model.joblib')
norm = load('final_model/minmax_scaler.joblib')


# loading the Labels
with open('labels/alphabets.txt', 'r') as f:
    label_list = [line.strip() for line in f if line.strip()]


# flask implementation
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/supported_gestures')
def supported_gestures():
    return render_template('supported_gestures.html')

@app.route('/landmark')
def landmark():
    return render_template('landmark.html')

@app.route('/custom_sign')
def custom_sign():
    return render_template('custom_sign.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/configuration', methods=['GET', 'POST'])
def configuration():
    current_resolution = None
    for key, (w, h) in cm.RESOLUTIONS.items():
        if w == cm.VIDEO_WIDTH and h == cm.VIDEO_HEIGHT:
            current_resolution = key
            break
    if request.method == 'POST':
        new_res = request.form.get('resolution')
        if new_res in cm.RESOLUTIONS:
            cm.VIDEO_WIDTH, cm.VIDEO_HEIGHT = cm.RESOLUTIONS[new_res]
            current_resolution = new_res
        return redirect(url_for('configuration'))
    return render_template('configuration.html', current_resolution=current_resolution, resolutions=cm.RESOLUTIONS)

@app.route('/video_feed')
def video_feed():
    return Response(cm.gen_frames(model, norm, label_list), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_landmarks', methods=['POST'])
def toggle_landmarks():
    cm.show_landmarks = not cm.show_landmarks
    print("SHOW_LANDMARKS is now", cm.show_landmarks)
    return jsonify({"landmarks_enabled": cm.show_landmarks})

@app.route('/ml_stream')
def ml_stream():
    def generate():
        # You could push an update on every frame, but in a simple implementation,
        # we yield the current text in a tight loop with a very short delay.
        while True:
            # Yield the latest text from your ML process.
            # Make sure cm.current_ml_text is updated in your frame processing.
            yield f"data: {cm.ml_result}\n\n"
            time.sleep(0.05)  # adjust the sleep time for how fast you want updates (here ~20 updates/sec)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    text = data.get('text', '')
    if text:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    return jsonify({"status": "spoken", "text": text})

@app.route('/update_text', methods=['POST'])
def update_text():
    data = request.get_json()
    text = data.get("text", None)
    # Set the custom_text and its expiration (current time + 5 seconds).
    cm.custom_label = text
    cm.text_expires_at  = time.time() + 5
    print("Custom text updated to:", text)
    return jsonify({"status": "success", "text": text})

if __name__ == '__main__':
    app.run(debug=True)

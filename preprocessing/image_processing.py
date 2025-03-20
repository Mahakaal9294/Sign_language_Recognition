import cv2
import mediapipe as mp
import sys

import preprocessing.process_keypoints as kp




VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
# Predefined resolutions for convenience
RESOLUTIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080)
}


# Initialize MediaPipe Hands modules.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands_model = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

show_landmarks = True
# ml_result = 'Wating for pose ....'

custom_label = None
text_expires_at  = 0

def process_frame(frame, model, norm, label_list):

    global ml_result
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(rgb_frame)
    results_pose = pose_model.process(rgb_frame) 

    if custom_label != None:
        row = kp.capture_data(results, results_pose, VIDEO_WIDTH, VIDEO_HEIGHT,text_expires_at,custom_label)
    else:
        row = kp.capture_data(results, results_pose, VIDEO_WIDTH, VIDEO_HEIGHT,text_expires_at)
    if row:
        ml_result = kp.gen_result(row, model, norm,label_list)
    if show_landmarks and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame


def gen_frames(model, norm, label_list):

    global ml_result
    ml_result = 'Wating for pose ....'

    if sys.platform.startswith('win'):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)


    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                break
            frame = process_frame(frame,model,norm, label_list)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()
        print("Camera released.")
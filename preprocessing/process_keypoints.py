import mediapipe as mp
import pandas as pd
import os
import time

pose_landmark_labels = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER",
    "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT"
]

hand_landmark_labels = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]


def capture_data(hand, pose, VIDEO_WIDTH, VIDEO_HEIGHT, expires_at, Label = None):
    row = {}

    if hand.multi_hand_landmarks:
        origin_x = hand.multi_hand_landmarks[0].landmark[0].x * VIDEO_WIDTH
        origin_y = hand.multi_hand_landmarks[0].landmark[0].y * VIDEO_HEIGHT

        for hand_landmarks in hand.multi_hand_landmarks:
                    # iterate over each key points
                    for name, landmark in zip(hand_landmark_labels[1:], hand_landmarks.landmark[1:]):
                        #storing the x & y coordinates
                        row[f'{name}_x'] = (landmark.x * VIDEO_WIDTH) - origin_x
                        row[f'{name}_y'] = (landmark.y * VIDEO_HEIGHT) - origin_y
    
        if pose.pose_landmarks:             # only if face is detected
                for name, landmark in zip(pose_landmark_labels[0:1], pose.pose_landmarks.landmark[0:1]):
                    #storing the x & y coordinates
                    row[f'{name}_x'] = (landmark.x * VIDEO_WIDTH) - origin_x
                    row[f'{name}_y'] = (landmark.y * VIDEO_HEIGHT)- origin_y


    now = time.time()
    if now < expires_at and Label and len(row) >=42:
        row['label'] = Label
        df = pd.DataFrame([row])
        csv_file = 'keypoints/custom_keypoint.csv'
        write_header = not os.path.exists(csv_file)
        df.to_csv(csv_file, mode='a', header=write_header, index=False)
        return
    return row



def gen_result(row, model, normalizer, label_list):
    try:
        df = pd.DataFrame([row])
        normalized_value = normalizer.transform(df)
        res_idx = model.predict(normalized_value)[0]
        result = label_list[res_idx]
    except:
         result = 'Wating for pose ....'

    return result
import cv2
import time
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results


def draw_styled_landmarks(image, results):

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(66, 245, 129), thickness=1, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(66, 245, 173), thickness=1, circle_radius=2)
                              )

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=2)
                              )


def extract_keypoints(results):

    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)

    return np.concatenate([ lh, rh, ])


DATA_PATH = os.path.join('MP_Data')
# Actions
#actions = np.array(['clear','Excuse Me','Good Morning','Who are you','Call me','How you know','I','want','water','Pen','You'])
actions = np.array(['I','clear','You','water','book','need','help'])
# 10 videos worth of data
no_sequences = 20
# 30 frames
sequence_length = 20
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)
# Mediapipe Model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop through Actions
    for action in actions:
        # Loop through Videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):
                # Read Feed
                ret, frame = cap.read()
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # Draw Styled Landmarks
                draw_styled_landmarks(image, results)
                # Wait Logic
                if frame_num == 0:
                    #cv2.waitKey(400)
                    cv2.putText(image, ' Input your Gestures', (120, 180),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.waitKey(100)
                    cv2.putText(image, ' Frames for \"{}\" Video Number \"{}\"'.format(action, sequence),
                                (15, 45),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.waitKey(200)
                    cv2.putText(image, ' Frames for \"{}\" Video Number \"{}\"'.format(action, sequence),
                                (15, 45),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Show to Screen
                cv2.imshow('Data Collection', image)

                # Breaking the Feed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
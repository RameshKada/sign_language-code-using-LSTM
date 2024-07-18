import time
import pyttsx3
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

import cv2
import pyttsx3

# Define a function for moving average
def moving_average(predictions, window_size=5):
    smoothed_predictions = []
    for i in range(len(predictions)):
        start_index = max(0, i - window_size + 1)
        end_index = i + 1
        smoothed_predictions.append(sum(predictions[start_index:end_index]) / (end_index - start_index))
    return smoothed_predictions

def text_to_speech(text):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

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
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=4)
                              )

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=4)
                              )


def extract_keypoints(results):

    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([ lh, rh])


colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = np.zeros((750, 1500, 3), dtype=np.uint8)
    output_frame = input_frame.copy()

    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame


DATA_PATH = os.path.join('MP_Data')

# Actions
#actions = np.array(['Excuse Me','clear','Who are you','Good Morning','Call me','How you know','I','want','water','Pen','You'])
actions = np.array(['I','clear','You','water','book','need','help'])
#actions = np.array(['book','clear','help','I','need','pen','water','You'])
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

label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(20, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

res = [.2, 0.7, .01]

actions[np.argmax(res)]
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
actions[np.argmax(res[0])]

model.load_weights('action.h5')

# New Detection Variables
sequence = []
sentence = []
threshold = .4

cap = cv2.VideoCapture(0)

# Mediapipe Model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read Feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw Styled Landmarks
        draw_styled_landmarks(image, results)
        #time.sleep(0.25)
        #print('processing')

        # Prediciton Logic
        keypoints = extract_keypoints(results)
        sequence.insert(0, keypoints)
        sequence = sequence[:20]

        if len(sequence) == 20:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

        # Visualization
        if res[np.argmax(res)] > threshold:
            if len(sentence) > 0:
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
            else:
                sentence.append(actions[np.argmax(res)])

        if len(sentence) > 3:
            sentence = sentence[-3:]
            # Inside the while loop
        if actions[np.argmax(res)] == 'clear':
            sentence = []

        # Viz probability
        # image = prob_viz(res,actions,image,colors)


        cv2.rectangle(image, (0, 0), (640, 40), (8, 225, 2), -1)

        cv2.putText(image,' '.join(sentence), (115, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        #cv2.imshow('output_frame', image)

        # Show to Screen

        # cv2.namedWindow('output_frame', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('output_frame', 1400, 750)
        # cv2.imshow('output_frame', image)
        cv2.imshow('Input your Gestures', image)
        text_to_speech(' '.join(sentence))

        # cv2.imshow('Input your Gestures', image)
        # cv2.imshow('Input your Gestures', output_frame)

        # Breaking the Feed
        #if cv2.waitKey(1) & 0x0F == ord('q'):
            #break

    cap.release()
    cv2.destroyAllWindows()
    # text_to_speech(' '.join(sentence))
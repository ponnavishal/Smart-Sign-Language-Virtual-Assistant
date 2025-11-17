import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import mediapipe as mp

# SETTINGS
IMAGE_SIZE = 64
DATASET_DIR = r"C:\Users\india\OneDrive - Chandigarh University\Desktop\project\asl_alphabet_train"  # CHANGE THIS

# Load Model + Class Names
model = tf.keras.models.load_model("asl_model.h5")
class_names = sorted(os.listdir(DATASET_DIR))

# Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Streamlit page
st.title("✋ ASL Realtime Sentence Builder")
st.write("Show any symbol to detect a letter. After detecting, it asks for next symbol.")

FRAME_WINDOW = st.empty()

# Sentence memory
if "sentence" not in st.session_state:
    st.session_state.sentence = ""

start = st.button("Start Detection")
clear = st.button("Clear Sentence")

if clear:
    st.session_state.sentence = ""

if start:
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not detected.")
                break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            # Process frame with mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            detected_letter = None

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:

                    # Get bounding box
                    x_vals = [lm.x for lm in handLms.landmark]
                    y_vals = [lm.y for lm in handLms.landmark]

                    xmin = int(min(x_vals) * w) - 20
                    ymin = int(min(y_vals) * h) - 20
                    xmax = int(max(x_vals) * w) + 20
                    ymax = int(max(y_vals) * h) + 20

                    xmin, ymin = max(0, xmin), max(0, ymin)
                    xmax, ymax = min(w, xmax), min(h, ymax)

                    # Draw rectangle around hand
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 3)

                    # Crop hand area
                    hand_img = frame[ymin:ymax, xmin:xmax]

                    if hand_img.size != 0:
                        # Prepare for model
                        hand_img = cv2.resize(hand_img, (IMAGE_SIZE, IMAGE_SIZE))
                        hand_img = hand_img.astype("float32") / 255.0
                        hand_img = np.expand_dims(hand_img, axis=0)

                        pred = model.predict(hand_img)
                        detected_letter = class_names[np.argmax(pred)]

                        # Show detected letter above box
                        cv2.putText(frame, f"{detected_letter}", (xmin, ymin - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # If a letter detected → add to sentence
            if detected_letter is not None:
                st.session_state.sentence += detected_letter

            # Text Instructions
            cv2.putText(frame, "Show Symbol", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(frame, "Show Next Symbol", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,150,255), 2)

            # Sentence output
            cv2.putText(frame, f"Sentence: {st.session_state.sentence}",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()

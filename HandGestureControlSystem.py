#Gen AI Hackathon 
#Team 70
#Team Lead:B.NISSI APEKSHA(23WH1A0523)
#Member1:P.TEENA PRANATHI(23WH1A0508)
#Member2:B.SAI DEVI SRI(23WH1A0513)
#Member3:P.KEERTHI(23WH1A0534)

import cv2
import mediapipe as mp
import pyautogui
import speech_recognition as sr
import google.generativeai as genai
import os
import pyttsx3

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
recognizer = sr.Recognizer()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY')) #environment variable
model = genai.GenerativeModel('gemini-pro')
engine = pyttsx3.init()

def detect_gesture(landmarks):
    index_tip = landmarks[8].y
    thumb_tip = landmarks[4].y
    if index_tip < thumb_tip:
        return "play"
    elif index_tip > thumb_tip:
        return "pause"
    return None

def recognize_speech():
    with sr.Microphone() as source:
        print("Listening for voice commands...")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            print(f"You said: {command}")
            return command
        except sr.WaitTimeoutError:
            print("No voice command detected.")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None

def get_gemini_response(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "Sorry, I encountered an error with Gemini."

def count_fingers(landmarks):
    if not landmarks:
        return 0
    fingers = []
    for hand_landmarks in landmarks:
        hand_landmarks = hand_landmarks.landmark
        thumb_tip = hand_landmarks[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks[mp_hands.HandLandmark.PINKY_TIP]
        thumb_ip = hand_landmarks[mp_hands.HandLandmark.THUMB_IP]
        index_pip = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_pip = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_pip = hand_landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_pip = hand_landmarks[mp_hands.HandLandmark.PINKY_PIP]

        fingers.append(thumb_tip.x > thumb_ip.x)
        fingers.append(index_tip.y < index_pip.y)
        fingers.append(middle_tip.y < middle_pip.y)
        fingers.append(ring_tip.y < ring_pip.y)
        fingers.append(pinky_tip.y < pinky_pip.y)

        finger_count = sum(fingers)
        return finger_count
    return 0

video_playing = True
voice_mode = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_count = count_fingers(results.multi_hand_landmarks)

            if finger_count == 3 and not voice_mode:
                voice_mode = True
                cv2.putText(frame, "Voice Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                engine.say("Voice mode activated.")
                engine.runAndWait()
            elif finger_count == 4 and voice_mode:
                voice_mode = False
                cv2.putText(frame, "Voice Mode Off", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                engine.say("Voice mode off.")
                engine.runAndWait()
            elif voice_mode:
                command = recognize_speech()
                if command:
                    if "volume up" in command:
                        pyautogui.press("up")
                        engine.say("Volume up.")
                    elif "volume down" in command:
                        pyautogui.press("down")
                        engine.say("Volume down.")
                    elif "mute" in command:
                        pyautogui.press("m")
                        engine.say("Mute.")
                    else:
                        ai_response = get_gemini_response(command)
                        print(f"Gemini Response: {ai_response}")
                        engine.say(ai_response)
                else:
                    cv2.putText(frame, "No Command", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                engine.runAndWait()
            else:
                gesture = detect_gesture(hand_landmarks.landmark)
                if gesture == "play" and not video_playing:
                    video_playing = True
                    pyautogui.press("k")
                    cv2.putText(frame, "Play", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif gesture == "pause" and video_playing:
                    video_playing = False
                    pyautogui.press("k")
                    cv2.putText(frame, "Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

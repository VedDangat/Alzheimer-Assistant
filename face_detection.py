import cv2
import imutils
import numpy as np
import os
import urllib.request
from deepface import DeepFace
import pyttsx3
import sys
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from tkinter import Tk, Button, messagebox
import geocoder

# -------------------------------
# Email configuration for SOS
# -------------------------------
SENDER_EMAIL = "veddangat123@gmail.com"        # your Gmail
SENDER_PASSWORD = "kdhd awub batm yzdx"        # Gmail App Password
RECEIVER_EMAIL = "vijayadangat2582@gmail.com"  # caregiver email

# -------------------------------
# Paths and model files
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

prototxt = os.path.join(BASE_DIR, "deploy.prototxt")
model = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Download model files if missing
if not os.path.exists(prototxt):
    print("[INFO] Downloading deploy.prototxt...")
    urllib.request.urlretrieve(PROTO_URL, prototxt)

if not os.path.exists(model):
    print("[INFO] Downloading res10_300x300_ssd_iter_140000.caffemodel...")
    urllib.request.urlretrieve(MODEL_URL, model)

# Load face detector
print("[INFO] Loading face detector...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# -------------------------------
# Text to Speech setup
# -------------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

def speak(text):
    print("[AUDIO]:", text)
    engine.say(text)
    engine.runAndWait()

# -------------------------------
# Known person DB (Ved’s photo)
# -------------------------------
VED_IMG = os.path.join(BASE_DIR, "people_db", "ved.jpg")
if not os.path.exists(VED_IMG):
    print(f"[ERROR] Ved's image not found at {VED_IMG}")
    print("👉 Please add a clear photo of Ved named 'ved.jpg' inside 'people_db' folder.")
    sys.exit(1)

# -------------------------------
# SOS email function
# -------------------------------
def send_sos_email():
    try:
        # Use IP-based geolocation
        g = geocoder.ipinfo('me')  
        latitude, longitude = g.latlng
        location_info = f"Patient needs help!\nLocation: https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"

        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = "SOS Alert!"
        msg.attach(MIMEText(location_info, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        messagebox.showinfo("SOS Alert", "SOS Email Sent Successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to send email:\n{e}")


# -------------------------------
# Main face recognition function
# -------------------------------
def run_face_detection():
    # Start Tkinter window with SOS button
    form = Tk()
    form.title("SOS Panel")
    form.geometry("300x150")

    sos_button = Button(form, text="SOS", font=("Arial", 20), fg="white", bg="red", command=send_sos_email)
    sos_button.pack(pady=30)

    cap = cv2.VideoCapture(0)
    last_speak_time = 0
    cooldown = 5
    VED_THRESHOLD = 0.4  # 40% similarity threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # Face detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        face_found = False
        message = None

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                face_found = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

                face_roi = frame[startY:endY, startX:endX]
                if face_roi.size > 0:
                    try:
                        temp_face = "temp_face.jpg"
                        cv2.imwrite(temp_face, face_roi)

                        result = DeepFace.verify(img1_path=VED_IMG, img2_path=temp_face,
                                                 enforce_detection=True, detector_backend="opencv")
                        similarity = (1 - result["distance"])
                        verified = similarity >= VED_THRESHOLD

                        if verified:
                            text = f"Ved ({similarity*100:.2f}%)"
                            message = "The person is identified. His name is Ved."
                            color = (0, 255, 0)
                        else:
                            text = f"Unknown ({similarity*100:.2f}%)"
                            message = "The person is not identified."
                            color = (0, 0, 255)

                        cv2.putText(frame, text, (startX, startY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        os.remove(temp_face)
                    except Exception as e:
                        print("[ERROR] DeepFace failed:", e)

                break

        if not face_found:
            cv2.putText(frame, "No Face", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            message = "No face detected."

        # Speak only if cooldown passed
        if message and (time.time() - last_speak_time > cooldown):
            speak(message)
            last_speak_time = time.time()

        cv2.imshow("Face Recognition", frame)
        form.update()  # update SOS window

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    form.destroy()

import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import tkinter as tk
from tkinter import messagebox
import csv
import time
import threading
import sys

# Initialize MTCNN for face detection
detector = MTCNN()

# Load the pre-trained FaceNet model
embedder = FaceNet()

# Function to load saved embeddings and names
def load_embeddings():
    embeddings = []
    names = []
    with open('face_embeddings.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            name = row[0]
            embedding = np.array(row[1:], dtype=float)
            embeddings.append(embedding)
            names.append(name)
    return np.array(embeddings), names

# Function to find the closest match
def find_closest_match(embedding, stored_embeddings, names):
    distances = np.linalg.norm(stored_embeddings - embedding, axis=1)
    min_distance_index = np.argmin(distances)
    if distances[min_distance_index] < 1.0:  # Set a threshold for recognition
        return names[min_distance_index]
    else:
        return "Unknown"
    
# Function to capture face and save embedding (clears the old data)
def capture_and_save_embeddings(name, roll):
    # Clear the CSV file before saving new data
    with open('face_embeddings.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    captured = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = detector.detect_faces(frame)
        
        if faces:
            # Only consider the first detected face (most prominent)
            face = faces[0]
            x1, y1, width, height = face['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            
            # Extract the face
            face_img = frame[y1:y2, x1:x2]
            face_img = cv2.resize(face_img, (160, 160))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display the frame with detected face
            cv2.imshow("Face Capture", frame)
            
            # Capture face for 5-10 seconds
            if time.time() - start_time > 10 and not captured:
                face_embedding = embedder.embeddings([face_img])[0]
                with open('face_embeddings.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name + "_" + roll] + face_embedding.tolist())
                captured = True
                messagebox.showinfo("Success", f"Face data for {name} saved.")
                break
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to handle cleanup and exit
def cleanup_and_exit(cap):
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

# Function to start face recognition with a single frame capture every 3 seconds
def start_face_recognition():
    stored_embeddings, names = load_embeddings()
    cap = cv2.VideoCapture(0)

    # Countdown before starting
    for i in range(3, 0, -1):
        messagebox.showinfo("Countdown", f"Starting in {i}...")
        time.sleep(1)

    warnings = 0
    last_capture_time = time.time()  # Initialize the last capture time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if 3 seconds have passed since the last capture
        current_time = time.time()
        if current_time - last_capture_time >= 3.0:
            last_capture_time = current_time  # Update last capture time

            # Detect faces
            faces = detector.detect_faces(frame)

            # If no faces are detected, increment warnings
            if not faces:
                warnings += 1
                if warnings == 1:
                    messagebox.showwarning("Warning", "No face detected! Please check the camera.")
                if warnings >= 3:
                    messagebox.showerror("Error", "Suspicious Activity Detected, Closing the exam.")
                    cleanup_and_exit(cap)

            else:
                # warnings = 0  # Reset warnings on successful detection
                # Only process the first detected face (most prominent)
                face = faces[0]
                x1, y1, width, height = face['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height

                # Extract and preprocess the face
                face_img = frame[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, (160, 160))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                # Get the embedding for the detected face
                face_embedding = embedder.embeddings([face_img])[0]

                # Find the closest match
                name = find_closest_match(face_embedding, stored_embeddings, names)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Handle unknown faces
                if name == "Unknown":
                    warnings += 1
                    if warnings < 3:
                        messagebox.showwarning("Warning", "Suspicious activity detected! Please check the person in the frame.")
                    if warnings >= 3:
                        messagebox.showerror("Error", "Suspicious Activity Detected, Closing the exam.")
                        cleanup_and_exit(cap)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Break the loop after a set time or if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cleanup_and_exit(cap)

# GUI for taking input and starting face recognition
def start_gui():
    root = tk.Tk()
    root.title("Face Recognition Exam System")

    tk.Label(root, text="Enter Name:").grid(row=0, column=0)
    name_entry = tk.Entry(root)
    name_entry.grid(row=0, column=1)

    tk.Label(root, text="Enter Roll Number:").grid(row=1, column=0)
    roll_entry = tk.Entry(root)
    roll_entry.grid(row=1, column=1)

    def on_start_capture():
        name = name_entry.get()
        roll = roll_entry.get()
        if name and roll:
            threading.Thread(target=capture_and_save_embeddings, args=(name, roll)).start()
        else:
            messagebox.showerror("Error", "Please enter both name and roll number.")

    def on_start_recognition():
        threading.Thread(target=start_face_recognition).start()

    tk.Button(root, text="Capture Face", command=on_start_capture).grid(row=2, column=0)
    tk.Button(root, text="Start Recognition", command=on_start_recognition).grid(row=2, column=1)

    root.mainloop()

# Run the GUI
start_gui()

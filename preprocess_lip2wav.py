import os
import cv2
import librosa
import numpy as np

# Set the paths to the GRID corpus dataset
data_path = "/path/to/grid/corpus"
video_path = os.path.join(data_path, "videos")
audio_path = os.path.join(data_path, "audio")

# Set the output paths for the preprocessed data
out_path = "/path/to/preprocessed/data"
lip_path = os.path.join(out_path, "lips")
audio_file = os.path.join(out_path, "audio.npy")
split_file = os.path.join(out_path, "split.npy")

# Set the size of the lip images
lip_size = (64, 64)

# Set the validation split ratio
split_ratio = 0.2

# Create the output directories if they don't exist
os.makedirs(lip_path, exist_ok=True)

# Initialize the arrays for the lip images and speech audio
lip_array = []
audio_array = []

import cv2

def detect_lips(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the face using the Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Extract the region of interest (ROI) corresponding to the face
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect the lips using the Canny edge detector
        edges = cv2.Canny(roi_gray, 100, 200)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000 and area < 3000:
                # Extract the lips from the ROI
                x, y, w, h = cv2.boundingRect(cnt)
                lips = roi_color[y:y+h, x:x+w]

                # Apply a median blur to smooth out the image
                lips = cv2.medianBlur(lips, 5)

                return lips
    return None

# Loop over all the video files in the GRID corpus dataset
for file_name in os.listdir(video_path):
    # Extract the subject and sentence ID from the file name
    subject_id, sentence_id = file_name.split("_")[:2]

    # Load the video file and extract the lip images
    video_file = os.path.join(video_path, file_name)
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Detect the lips in the frame
        # and resize them to the desired size
        # before saving them to disk
        lip = detect_lips(frame)
        lip = cv2.resize(lip, lip_size)
        lip_file = os.path.join(lip_path, f"{subject_id}_{sentence_id}_{cap.get(cv2.CAP_PROP_POS_FRAMES):04d}.jpg")
        cv2.imwrite(lip_file, lip)
    cap.release()

    # Load the corresponding audio file and extract the speech audio
    audio_file = os.path.join(audio_path, f"{subject_id}_{sentence_id}.wav")
    audio, sr = librosa.load(audio_file)
    audio_array.append(audio)

# Convert the lip images and speech audio to numpy arrays
lip_array = np.array(lip_array)
audio_array = np.array(audio_array)

# Save the lip images and speech audio arrays to disk
np.save(audio_file, audio_array)

# Split the data into training and validation sets
n_samples = len(lip_array)
n_val_samples = int(n_samples * split_ratio)
val_indices = np.random.choice(n_samples, size=n_val_samples, replace=False)
train_indices = np.setdiff1d(np.arange(n_samples), val_indices)
np.save(split_file, {"train": train_indices, "val": val_indices})
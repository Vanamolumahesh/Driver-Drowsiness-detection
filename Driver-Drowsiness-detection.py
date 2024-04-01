!pip install playsound
!pip install Pillow
import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance
from playsound import playsound
from IPython.display import Audio
def eye_aspect_ratio(eye):
A = distance.euclidean(eye[1], eye[5])
B = distance.euclidean(eye[2], eye[4])
C = distance.euclidean(eye[0], eye[3])
ear = (A + B) / (2.0 * C)
return ear
cap = cv2.VideoCapture(0)
if not cap.isOpened():
print("Error: Could not open webcam.")
# Handle the error accordingly (e.g., exit the script)
cap =
cv2.VideoCapture("/content/drive/MyDrive/shape_predictor_68_face_landma
rks.dat")
if not cap.isOpened():
print("Error: Could not open the video file.")
# Handle the error accordingly (e.g., exit the script)
cap = cv2.VideoCapture(0) # or your video source
while True:
ret, frame = cap.read()
if not ret:
print("Error: Failed to read frame.")
break # This break statement is inside the loop
# Process the frame or perform other operations inside the loop
# ...
key = cv2.waitKey(1) & 0xFF
if key == ord("q"):
break # This break statement is also inside the loop
# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
def eye_aspect_ratio(eye):
A = distance.euclidean(eye[1], eye[5])
B = distance.euclidean(eye[2], eye[4])
C = distance.euclidean(eye[0], eye[3])
# compute the eye aspect ratio
ear = (A + B) / (2.0 * C)
# return the eye aspect ratio
return ear
thresh = 0.25
frame_check = 20
flag = 0
detect = dlib.get_frontal_face_detector()
predict =
dlib.shape_predictor("/content/drive/MyDrive/shape_predictor_68_face_la
ndmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
import cv2
import numpy as np
from playsound import playsound
# Load pre-trained Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
'haarcascade_eye.xml')
# Initialize the video capture
cap = cv2.VideoCapture(0)
def eye_aspect_ratio(eye):
A = np.linalg.norm(eye[1] - eye[5])
B = np.linalg.norm(eye[2] - eye[4])
C = np.linalg.norm(eye[0] - eye[3])
ear = (A + B) / (2.0 * C)
return ear
def play_alarm():
# Play a sound alarm (change the file path to your alarm sound)
playsound(r"C:\Users\Lenovo\OneDrive\Desktop\mixkit-critical-alarm1004.wav")
def capture_image(frame):
# Save the captured frame as an image file (JPEG)
cv2.imwrite('captured_frame.jpg', frame)
print("Image captured!")
while True:
ret, frame = cap.read()
if not ret:
break
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
minNeighbors=5, minSize=(30, 30))
for (x, y, w, h) in faces:
roi_gray = gray[y:y+h, x:x+w]
roi_color = frame[y:y+h, x:x+w]
eyes = eye_cascade.detectMultiScale(roi_gray)
for (ex, ey, ew, eh) in eyes:
eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]
ear = eye_aspect_ratio([[ex, ey], [ex+ew//2, ey], [ex+ew,
ey],
[ex, ey+eh//2], [ex+ew, ey+eh//2],
[ex, ey+eh], [ex+ew//2, ey+eh],
[ex+ew, ey+eh]])
cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255,
0), 2)
if ear < 0.25:
cv2.putText(frame, "DROWSY", (x, y - 10),
cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
play_alarm()
capture_image(frame)
cv2.imshow('Driver Drowsiness Detection', frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
break
cap.release()
cv2.destroyAllWindows()
import cv2
from google.colab.patches import cv2_imshow
# Open the camera (default camera index is 0)
cap = cv2.VideoCapture(0)
# Check if the camera is opened successfully
if not cap.isOpened():
print("Error: Could not open camera.")
else:
# Capture a single frame
ret, frame = cap.read()
# Check if the frame was successfully captured
if ret:
# Display the captured image
cv2_imshow(frame)
# Save the captured image (optional)
cv2.imwrite(r"C:\Users\Lenovo\OneDrive\Desktop\pavan img.png",
frame)
else:
print("Error: Failed to capture image.")
# Release the camera
cap.release()
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
import cv2
import numpy as np
from PIL import Image as PILImage
from io import BytesIO
import os
from base64 import b64decode
def take_photo(directory=("C:/Users/Lenovo/Downloads/Driver Drowsiness
Dataset (DDD)"), filename="Drowsy.jpg", quality=0.8):
# Check if the directory exists, and create it if not
if not os.path.exists(directory):
os.makedirs(directory)
filepath = os.path.join(directory, filename)
js = Javascript('''
async function takePhoto(quality) {
const div = document.createElement('div');
const capture = document.createElement('button');
capture.textContent = 'Capture';
div.appendChild(capture);
const video = document.createElement('video');
video.style.display = 'block';
const stream = await navigator.mediaDevices.getUserMedia({
'video': true });
document.body.appendChild(div);
div.appendChild(video);
video.srcObject = stream;
await video.play();
// Resize the output to fit the video element.
google.colab.output.setIframeHeight(document.documentElemen
t.scrollHeight, true);
// Wait for Capture to be clicked.
await new Promise((resolve) => capture.onclick = resolve);
const canvas = document.createElement('canvas');
canvas.width = video.videoWidth;
canvas.height = video.videoHeight;
canvas.getContext('2d').drawImage(video, 0, 0);
stream.getVideoTracks()[0].stop();
div.remove();
return canvas.toDataURL('image/jpeg', quality);
}
''')
display(js)
data = eval_js('takePhoto({})'.format(quality))
binary = b64decode(data.split(',')[1])
with open(filepath, 'wb') as f:
f.write(binary)
return filepath
# Capture a photo
photo_path = take_photo()
# Display the captured photo using PIL
img = cv2.imread(photo_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
display(PILImage.fromarray(img))
import ipywidgets as widgets
from IPython.display import display
def on_capture_click(b):
print("Capture button clicked")
# Add your capture logic here
capture_button = widgets.Button(description="Capture")
capture_button.on_click(on_capture_click)
display(capture_button)

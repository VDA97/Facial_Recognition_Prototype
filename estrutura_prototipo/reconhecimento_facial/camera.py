# app_name/camera.py
import cv2
import os
import threading
from django.conf import settings
from .models import User, TrainedModel


class VideoCamera(object):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.video = None  # Initializes the camera as None
        self.is_streaming = False  # Flag to control streaming state
        self.camera_lock = threading.Lock()  # Lock for safe camera access

        self.img_dir = "../tmp"
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        # --- Recognition logic ---
        self.recognizer = cv2.face.EigenFaceRecognizer_create(num_components=100, threshold=8000)
        self.model_loaded = False
        try:
            training = TrainedModel.objects.first()
            if training:
                model_path = os.path.join(settings.MEDIA_ROOT, training.model_file.name)
                if os.path.exists(model_path):
                    self.recognizer.read(model_path)
                    self.model_loaded = True
                    print("Facial recognition model loaded successfully.")
                else:
                    print("Model path not found.")
            else:
                print("Training model not found in the database.")
        except Exception as e:
            print(f"Error loading the model: {e}")

    def __del__(self):
        self.stop_camera()

    def start_camera(self):
        """Initializes the camera if it is not already open."""
        with self.camera_lock:
            if not self.video or not self.video.isOpened():
                self.video = cv2.VideoCapture(0)
                if not self.video.isOpened():
                    print("Error starting the camera.")
                    self.video = None
                    return False
            self.is_streaming = True
            return True

    def stop_camera(self):
        """Safely releases the camera resource."""
        with self.camera_lock:
            if self.video and self.video.isOpened():
                self.video.release()
            self.video = None
            self.is_streaming = False

    def get_frame(self, retries=3):
        """
        Reads a frame from the camera safely.
        Returns the frame and a boolean indicating the success of the read.
        """
        with self.camera_lock:
            if not self.video or not self.video.isOpened():
                return None, False

            for _ in range(retries):
                ret, frame = self.video.read()
                if ret and frame is not None:
                    return frame, True
        return None, False

    def recognize_face(self):
        """
        Reads a frame, performs face detection and recognition.
        Returns the frame in bytes for streaming and the user ID if successful,
        otherwise, returns (None, None).
        """
        frame, success = self.get_frame()
        if not success:
            return None, None

        frame = cv2.resize(frame, (480, 360))
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Copy the frame before drawing for the recognition logic
        frame_copy = frame.copy()

        detected_faces = self.face_cascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(400, 400))

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        conf_text = "No face detected"
        text_color = (0, 0, 255)
        user_id = None  # Initializes with None

        face_x, face_y, face_w, face_h = 0, 0, 0, 0

        for (x, y, l, a) in detected_faces:
            face_x, face_y, face_w, face_h = x, y, l, a

            face_image = gray_image[y:y + a, x:x + l]
            face_image = cv2.resize(face_image, (220, 220))
            face_image = cv2.equalizeHist(face_image)
            face_image = cv2.normalize(face_image, None, 0, 255, cv2.NORM_MINMAX)

            cv2.rectangle(frame_copy, (x, y), (x + l, y + a), (0, 255, 0), 2)

            if self.model_loaded:
                label, confidence = self.recognizer.predict(face_image)

                if confidence < 8000:
                    try:
                        user = User.objects.get(id=label)
                        conf_text = f"{user.name} ({int(confidence)})"
                        text_color = (0, 255, 0)
                        user_id = user.id  # Returns the ID
                    except User.DoesNotExist:
                        conf_text = "Unknown"
                        text_color = (0, 0, 255)
                else:
                    conf_text = "Low confidence"
                    text_color = (0, 0, 255)
            else:
                conf_text = "Model not loaded"
                text_color = (0, 0, 255)

        frame_copy = cv2.flip(frame_copy, 1)

        if len(detected_faces) > 0:
            final_pos_x = frame_copy.shape[1] - (face_x + face_w)
            final_pos_y = face_y + face_h + 30
            cv2.putText(frame_copy, conf_text, (final_pos_x, final_pos_y), font, 1, text_color, 2)
        else:
            cv2.putText(frame_copy, conf_text, (10, 30), font, 1, text_color, 2)

        ret, jpeg = cv2.imencode('.jpg', frame_copy)
        return jpeg.tobytes(), user_id

    def detect_face(self):
        """
        Returns only the frame with the visual face detection, without recognition.
        """
        frame, success = self.get_frame()
        if not success:
            return None

        height, width, _ = frame.shape
        center_x, center_y = int(width / 2), int(height / 2)
        a, b = 140, 180
        x1, y1 = center_x - a, center_y - b
        x2, y2 = center_x + a, center_y + b

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        cv2.ellipse(frame, (center_x, center_y), (a, b), 0, 0, 360, (0, 0, 255), 10)

        for (x, y, w, h) in faces:
            cv2.ellipse(frame, (center_x, center_y), (a, b), 0, 0, 360, (0, 255, 0), 10)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def sample_faces(self):
        """
        Captures face frames for data collection.
        """
        frame, success = self.get_frame()
        if not success:
            return None

        frame = cv2.flip(frame, 180)
        frame = cv2.resize(frame, (480, 360))

        faces = self.face_cascade.detectMultiScale(
            frame, minNeighbors=20, minSize=(30, 30), maxSize=(400, 400))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cropped_face = frame[y:y + h, x:x + w]
            return cropped_face
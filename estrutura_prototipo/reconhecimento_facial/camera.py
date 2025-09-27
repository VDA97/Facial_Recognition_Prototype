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
        self.video = None
        self.is_streaming = False
        self.camera_lock = threading.Lock()

        self.img_dir = os.path.join(settings.BASE_DIR, 'tmp')
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        # --- Recognition logic ---
        self.recognizer = cv2.face.EigenFaceRecognizer_create(num_components=100, threshold=8000)
        self.model_loaded = self._load_model()

    def __del__(self):
        self.stop_camera()

    def _load_model(self):
        """Loads the trained facial recognition model."""
        try:
            training = TrainedModel.objects.first()
            if training:
                model_path = os.path.join(settings.MEDIA_ROOT, training.model_file.name)
                if os.path.exists(model_path):
                    self.recognizer.read(model_path)
                    print("Facial recognition model loaded successfully.")
                    print(f"Camera Modelo de treinamento carregado: {training.model_file.name}")

                    return True
                else:
                    print("Model path not found.")
            else:
                print("Training model not found in the database.")
        except Exception as e:
            print(f"Error loading the model: {e}")
        return False

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

    def _frame_to_bytes(self, frame):
        """Converts an OpenCV frame to JPEG bytes."""
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def _get_face_detection_params(self):
        """Returns standard parameters for face detection."""
        return {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30), 'maxSize': (400, 400)}

    # --- Methods for Photo Collection (Used in take_photos_stream) ---

    def draw_face_area(self):
        """
        Returns the frame with the visual face detection area, without recognition.
        """
        frame, success = self.get_frame()
        if not success:
            return None

        # Espelha a imagem para o usuário (experiência visual)
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        center_x, center_y = int(width / 2), int(height / 2)
        a, b = 140, 180
        x1, y1 = center_x - a, center_y - b

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, **self._get_face_detection_params())

        cv2.ellipse(frame, (center_x, center_y), (a, b), 0, 0, 360, (0, 0, 255), 10)

        for _ in faces:
            cv2.ellipse(frame, (center_x, center_y), (a, b), 0, 0, 360, (0, 255, 0), 10)

        return self._frame_to_bytes(frame)

    def crop_face(self):
        """
        Captures and crops face frames for data collection.
        """
        frame, success = self.get_frame()
        if not success:
            return None

        faces = self.face_cascade.detectMultiScale(
            frame, minNeighbors=20, minSize=(30, 30), maxSize=(400, 400))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cropped_face = frame[y:y + h, x:x + w]
            return cropped_face

    # --- Methods for Face Recognition (Used in face_recognition_stream) ---

    def recognize_face(self):
        """
        Reads a frame, performs face detection and recognition, and draws the result.
        Returns the frame in bytes for streaming and the user ID if successful,
        otherwise, returns (None, None).
        """
        frame, success = self.get_frame()
        if not success:
            return None, None

        # Processamento e detecção na imagem original (sem flip)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(
            gray_image, **self._get_face_detection_params())

        user_id = None
        for (x, y, w, h) in detected_faces:
            user_id = self._process_and_recognize_face(frame, gray_image, x, y, w, h)
            if user_id:
                break

        # Espelha o frame apenas para exibição no frontend (após o processamento)
        frame = cv2.flip(frame, 1)

        return self._frame_to_bytes(frame), user_id

    def _process_and_recognize_face(self, frame, gray_image, x, y, w, h):
        """Processes a single face for recognition and draws the result."""
        face_image = gray_image[y:y + h, x:x + w]
        face_image = cv2.resize(face_image, (220, 220))
        face_image = cv2.equalizeHist(face_image)
        face_image = cv2.normalize(face_image, None, 0, 255, cv2.NORM_MINMAX)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        conf_text = "Model not loaded"
        text_color = (0, 0, 255)
        user_id = None

        if self.model_loaded:
            label, confidence = self.recognizer.predict(face_image)

            if confidence < 8000:
                try:
                    user = User.objects.get(id=label)
                    conf_text = f"{user.name} ({int(confidence)})"
                    text_color = (0, 255, 0)
                    user_id = user.id
                except User.DoesNotExist:
                    conf_text = "Unknown"
                    text_color = (0, 0, 255)
            else:
                conf_text = "Low confidence"
                text_color = (0, 0, 255)

        self._draw_recognition_info(frame, x, y, w, h, conf_text, text_color)
        return user_id

    def _draw_recognition_info(self, frame, x, y, w, h, text, color):
        """Draws the recognition info text on the frame."""
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        text_width = cv2.getTextSize(text, font, 1, 2)[0][0]
        text_x = x + (w - text_width) // 2
        text_y = y - 10
        cv2.putText(frame, text, (text_x, text_y), font, 1, color, 2)
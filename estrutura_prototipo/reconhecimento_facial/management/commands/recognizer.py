import cv2
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from reconhecimento_facial.models import User, TrainedModel


class Command(BaseCommand):
    help = "Command to test facial recognition with a live camera display."

    def handle(self, *args, **kwargs):
        self.recognize_faces()

    def recognize_faces(self):
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        recognizer = cv2.face.EigenFaceRecognizer_create(num_components=100, threshold=8000)

        # Load the training model
        training = TrainedModel.objects.first()
        if not training:
            self.stdout.write(self.style.ERROR("Training model not found."))
            return

        model_path = os.path.join(settings.MEDIA_ROOT, training.model_file.name)
        recognizer.read(model_path)

        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            self.stdout.write(self.style.ERROR("Unable to open camera."))
            return

        width, height = 220, 220
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.stdout.write(self.style.SUCCESS("Camera opened successfully. Press 'q' to exit."))

        while True:
            ret, frame = camera.read()
            if not ret:
                self.stdout.write(self.style.ERROR("Error accessing the camera."))
                break

            frame = cv2.resize(frame, (480, 360))
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(
                gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(400, 400)
            )

            for (x, y, l, a) in detected_faces:
                face_image = gray_image[y:y + a, x:x + l]
                face_image = cv2.resize(face_image, (width, height))

                # Apply the same pre-processing used in training
                face_image = cv2.equalizeHist(face_image)
                face_image = cv2.normalize(face_image, None, 0, 255, cv2.NORM_MINMAX)

                cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
                label, confidence = recognizer.predict(face_image)
                print(f"The recognition confidence value is: {confidence}")

                # Only show recognition if confidence is good
                if confidence < 9000:  # adjust this value as needed
                    try:
                        user = User.objects.get(id=label)
                        name = str(user.name).strip("(),'")
                        conf_text = f"{name} ({int(confidence)})"
                        cv2.putText(frame, conf_text, (x, y + a + 30), font, 1, (0, 255, 0), 2)
                    except User.DoesNotExist:
                        cv2.putText(frame, "Unknown", (x, y + a + 30), font, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Low confidence", (x, y + a + 30), font, 1, (0, 0, 255), 2)

            frame = cv2.flip(frame, 1)
            cv2.imshow("Facial Recognition Prototype", frame)

            # Stop by pressing the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()
        self.stdout.write(self.style.SUCCESS('Camera closed.'))
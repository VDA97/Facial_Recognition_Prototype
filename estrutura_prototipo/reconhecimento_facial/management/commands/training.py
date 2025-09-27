import os
import numpy as np
import cv2
from django.conf import settings
from django.core.files import File
from django.core.management.base import BaseCommand
from reconhecimento_facial.models import ProcessedPhotos, TrainedModel


class Command(BaseCommand):
    help = "Trains the Eigen classifier for facial recognition."

    def handle(self, *args, **kwargs):
        self.train_faces()

    def train_faces(self):
        self.stdout.write(self.style.WARNING("Starting training with the information base."))
        print(f"OpenCV version: {cv2.__version__}")

        # Initialize the EigenFace classifier
        eigen_face = cv2.face.EigenFaceRecognizer_create(num_components=100, threshold=8000)

        faces, labels = [], []
        error_count = 0

        # Process each image in ProcessedPhotos
        for photo in ProcessedPhotos.objects.all():
            image_file = photo.image.url.replace('/media/roi/', '')
            image_path = os.path.join(settings.MEDIA_ROOT, 'roi', image_file)

            if not os.path.exists(image_path):
                print(f"Path not found: {image_path}")
                error_count += 1
                continue

            # Load and process the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading the image: {image_path}")
                error_count += 1
                continue

            # Improved image pre-processing
            face_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_image = cv2.resize(face_image, (220, 220))

            # Histogram equalization to improve contrast
            face_image = cv2.equalizeHist(face_image)

            # Image normalization
            face_image = cv2.normalize(
                face_image, None, 0, 255, cv2.NORM_MINMAX)

            faces.append(face_image)
            labels.append(photo.user.id)

        # If there are no faces, stop the training
        if not faces:
            print("No faces found for training.")
            return

        # Perform the model training
        try:
            eigen_face.train(np.array(faces), np.array(labels))
            print(f"{len(faces)} images trained successfully.")

            # Save the trained model to a temporary file
            tmp_dir = "./tmp"
            os.makedirs(tmp_dir, exist_ok=True)

            model_filename = os.path.join(tmp_dir, "eigenClassifier.xml")

            eigen_face.write(model_filename)

            # Save the model to the database
            with open(model_filename, 'rb') as f:
                trained_model, created = TrainedModel.objects.get_or_create()
                trained_model.model_file.save('eigenClassifier.yml', File(f))

            # Remove the temporary file and display status messages
            os.remove(model_filename)
            self.stdout.write(self.style.ERROR(
                f"Images with loading errors: {error_count}"))
            self.stdout.write(self.style.SUCCESS("TRAINING COMPLETED"))

        except Exception as e:
            print(f"Error during training: {e}")
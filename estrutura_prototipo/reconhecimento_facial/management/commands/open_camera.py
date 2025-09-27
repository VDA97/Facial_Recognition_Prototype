from django.core.management.base import BaseCommand
import cv2


class Command(BaseCommand):
    help = 'Opens the camera and displays a real-time video feed.'

    def handle(self, *args, **kwargs):
        # Open the camera (0 is the default index)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self.stdout.write(self.style.ERROR('Error opening the camera.'))
            return

        self.stdout.write(self.style.SUCCESS('Camera opened successfully. Press "q" to exit.'))

        while True:
            # Capture frame by frame
            ret, frame = cap.read()

            if not ret:
                self.stdout.write(self.style.ERROR('Error capturing the frame.'))
                break

            frame = cv2.flip(frame, 1)  # Flips the frame horizontally for a mirror effect
            # Display the frame
            cv2.imshow('Camera', frame)

            # Exit the loop by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()
        self.stdout.write(self.style.SUCCESS('Camera closed.'))
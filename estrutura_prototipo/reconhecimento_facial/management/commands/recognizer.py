import cv2
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from reconhecimento_facial.models import User, TrainedModel


class Command(BaseCommand):
    help = "Command to test facial recognition with a live camera display."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.face_cascade = self._load_cascade()
        self.recognizer = self._load_recognizer()
        self.camera = None
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.CONFIDENCE_THRESHOLD = 9000

    def handle(self, *args, **kwargs):
        self.recognize_faces()

    # --- Métodos de Inicialização ---

    def _load_cascade(self):
        """Carrega o classificador de detecção de rosto."""
        # Note: 'haarcascade_frontalface_default.xml' deve estar no mesmo diretório ou acessível pelo path.
        cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        if cascade.empty():
            self.stdout.write(self.style.ERROR(
                "Erro: Não foi possível carregar o classificador 'haarcascade_frontalface_default.xml'"))
            return None
        return cascade

    def _load_recognizer(self):
        """Carrega o modelo de treinamento e o reconhecedor."""
        recognizer = cv2.face.EigenFaceRecognizer_create(num_components=100, threshold=8000)
        try:
            training = TrainedModel.objects.first()
            if not training:
                self.stdout.write(self.style.ERROR("Modelo de treinamento não encontrado no banco de dados."))
                return None

            model_path = os.path.join(settings.MEDIA_ROOT, training.model_file.name)
            if not os.path.exists(model_path):
                self.stdout.write(self.style.ERROR(f"Arquivo de modelo não encontrado: {model_path}"))
                return None

            recognizer.read(model_path)
            self.stdout.write(self.style.SUCCESS(f"Modelo de treinamento carregado: {training.model_file.name}"))
            return recognizer
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Erro ao carregar o modelo: {e}"))
            return None

    # --- Lógica Principal do Reconhecimento ---

    def recognize_faces(self):
        """Inicia a câmera e o loop de reconhecimento facial."""
        if not self.face_cascade or not self.recognizer:
            return

        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.stdout.write(self.style.ERROR("Não foi possível abrir a câmera."))
            return

        self.stdout.write(self.style.SUCCESS("Câmera aberta com sucesso. Pressione 'q' para sair."))

        while True:
            ret, frame = self.camera.read()
            if not ret:
                self.stdout.write(self.style.ERROR("Erro ao acessar a câmera."))
                break

            frame = self._process_frame(frame)

            cv2.imshow("Protótipo de Reconhecimento Facial", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()
        self.stdout.write(self.style.WARNING("Câmera fechada."))

    def _process_frame(self, frame):
        """Processa um único frame para detecção e reconhecimento de rosto."""
        frame = cv2.resize(frame, (480, 360))
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_faces = self.face_cascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(400, 400)
        )

        for (x, y, l, a) in detected_faces:
            face_image = gray_image[y:y + a, x:x + l]
            self._recognize_single_face(frame, face_image, x, y, l, a)

        return cv2.flip(frame, 1)

    def _recognize_single_face(self, frame, face_image, x, y, l, a):
        """Reconhece e desenha a informação de um único rosto no frame."""
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)

        face_image = cv2.resize(face_image, (220, 220))
        face_image = cv2.equalizeHist(face_image)
        face_image = cv2.normalize(face_image, None, 0, 255, cv2.NORM_MINMAX)

        label, confidence = self.recognizer.predict(face_image)
        self.stdout.write(f"Valor de confiança: {confidence}")

        if confidence < self.CONFIDENCE_THRESHOLD:
            try:
                user = User.objects.get(id=label)
                name = str(user.name).strip("(),'")
                text = f"{name} ({int(confidence)})"
                color = (0, 255, 0)  # Green
            except User.DoesNotExist:
                text = "Desconhecido"
                color = (0, 0, 255)  # Red
        else:
            text = "Baixa confiança"
            color = (0, 0, 255)  # Red

        cv2.putText(frame, text, (x, y - 10), self.font, 1, color, 2)
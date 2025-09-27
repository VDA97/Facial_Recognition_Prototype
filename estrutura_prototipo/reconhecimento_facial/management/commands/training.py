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
        """Método principal que orquestra todo o processo de treinamento."""
        self.stdout.write(self.style.WARNING("Iniciando o treinamento com a base de dados."))

        # 1. Carrega e pré-processa as imagens
        faces, labels = self._load_and_preprocess_images()

        if not faces:
            self.stdout.write(self.style.ERROR("Não há rostos para treinar. Encerrando."))
            return

        # 2. Treina o classificador
        eigen_face = self._train_classifier(faces, labels)

        # 3. Salva o modelo e limpa os arquivos temporários
        self._save_and_clean_model(eigen_face)

    def _load_and_preprocess_images(self):
        """Carrega e pré-processa todas as imagens do banco de dados."""
        faces, labels = [], []
        error_count = 0

        self.stdout.write(self.style.SUCCESS("Processando imagens..."))

        for photo in ProcessedPhotos.objects.all():
            image_path = os.path.join(settings.MEDIA_ROOT, 'roi', photo.image.name)

            if not os.path.exists(image_path):
                self.stdout.write(self.style.ERROR(f"Arquivo não encontrado: {image_path}"))
                error_count += 1
                continue

            image = cv2.imread(image_path)
            if image is None:
                self.stdout.write(self.style.ERROR(f"Erro ao carregar a imagem: {image_path}"))
                error_count += 1
                continue

            # Chama a função auxiliar para processar a imagem
            face_image = self._preprocess_single_image(image)

            faces.append(face_image)
            labels.append(photo.user.id)

        self.stdout.write(self.style.ERROR(f"Imagens com erro de carregamento: {error_count}"))
        return faces, labels

    def _preprocess_single_image(self, image):
        """Aplica o pré-processamento de imagem para um único rosto."""
        face_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(face_image, (220, 220))
        face_image = cv2.equalizeHist(face_image)
        face_image = cv2.normalize(face_image, None, 0, 255, cv2.NORM_MINMAX)
        return face_image

    def _train_classifier(self, faces, labels):
        """Cria e treina o classificador EigenFace com os dados fornecidos."""
        self.stdout.write(self.style.SUCCESS("Iniciando o treinamento do modelo..."))

        eigen_face = cv2.face.EigenFaceRecognizer_create(num_components=100, threshold=8000)
        eigen_face.train(np.array(faces), np.array(labels))

        self.stdout.write(self.style.SUCCESS(f"{len(faces)} imagens treinadas com sucesso."))
        return eigen_face

    def _save_and_clean_model(self, eigen_face):
        """Salva o modelo treinado no banco de dados e limpa arquivos temporários."""
        try:
            tmp_dir = "./tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            model_filename = os.path.join(tmp_dir, "eigenClassifier.xml")
            eigen_face.write(model_filename)

            with open(model_filename, 'rb') as f:
                trained_model, created = TrainedModel.objects.get_or_create()
                trained_model.model_file.save('eigenClassifier.yml', File(f))

            os.remove(model_filename)
            self.stdout.write(self.style.SUCCESS("TREINAMENTO CONCLUÍDO"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Erro ao salvar o modelo: {e}"))
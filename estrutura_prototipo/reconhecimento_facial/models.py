from django.db import models
from django.core.exceptions import ValidationError
from django.utils.text import slugify
from random import randint
import os

class User(models.Model):
    photo = models.ImageField(upload_to='photos/')
    name = models.CharField(max_length=100)
    unique_id = models.CharField(max_length=20, unique=True, blank=True)

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.unique_id:
            unique_name = slugify(self.name)
            random_number = str(randint(1000000, 9999999))
            self.unique_id = f'{unique_name}-{random_number}'

        super().save(*args, **kwargs)

def user_directory_path(instance, filename):
    # Use o campo 'name' da sua classe User
    username = instance.user.name.replace(' ', '_').lower()

    # Pega a extensão original do arquivo
    ext = filename.split('.')[-1]

    # Cria o novo nome do arquivo: username_timestamp.ext
    # O timestamp é para evitar que dois uploads com o mesmo nome se sobreponham
    new_filename = f'{username}_{instance.id}.{ext}'

    # Retorna o caminho completo
    return os.path.join('roi', new_filename)

class ProcessedPhotos(models.Model):
    user = models.ForeignKey(User,
                             on_delete=models.CASCADE, related_name='user_processed_photos')
    image = models.ImageField(upload_to=user_directory_path)

class TrainedModel(models.Model):
    model_file = models.FileField(upload_to='training/')

    class Meta:
        verbose_name = 'Trained Model'
        verbose_name_plural = 'Trained Models'

    def __str__(self):
        return 'Frontal Face Trained Models'

    def clean(self):
        model = self.__class__
        if model.objects.exclude(id=self.id).exists():
            raise ValidationError('Only one file can be saved.')
from django.db import models
from django.core.exceptions import ValidationError
from django.utils.text import slugify
from random import randint


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


class ProcessedPhotos(models.Model):
    user = models.ForeignKey(User,
                             on_delete=models.CASCADE, related_name='user_processed_photos')
    image = models.ImageField(upload_to='roi/')


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
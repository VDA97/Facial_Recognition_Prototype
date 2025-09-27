from django.urls import path
from .views import (
    create_user,
    take_photos_manager,
    take_photos_stream,
    face_recognition,
    face_recognition_stream,
    face_recognition_check,
    recognized_user
)

urlpatterns = [
    # URL principal para criar um novo usuário
    path('', create_user, name='create_user'),

    # URL para o gerenciamento do fluxo de coleta de fotos
    path('take_photos/<int:user_id>/', take_photos_manager, name='take_photos_manager'),

    # URL para o streaming de vídeo durante a coleta de fotos
    path('take_photos_stream/', take_photos_stream, name='take_photos_stream'),

    # URL principal para o reconhecimento facial (renderiza o template)
    path('face_recognition/', face_recognition, name='face_recognition'),

    # URL para o streaming de vídeo do reconhecimento facial
    path('face_recognition_stream/', face_recognition_stream, name='face_recognition_stream'),

    # URL chamada via JavaScript para verificar o status do reconhecimento
    path('face_recognition_check/', face_recognition_check, name='face_recognition_check'),

    # URL que exibe a página de sucesso com os dados do usuário reconhecido
    path('recognized_user/<int:user_id>/', recognized_user, name='recognized_user')
]
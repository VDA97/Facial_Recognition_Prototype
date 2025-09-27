from django.urls import path
from .views import (
    create_user,
    take_photos,
    face_detection,
    face_recognition,
    face_recognition_stream,
    face_recognition_check,
    recognized_user
)

urlpatterns = [
    # Main URL for creating a new user
    path('', create_user, name='create_user'),

    # URL for photo collection (3-step flow)
    path('take_photos/<int:user_id>/', take_photos, name='take_photos'),

    # URL for face detection streaming (without recognition)
    path('face_detection/', face_detection, name='face_detection'),

    # Main URL for face recognition (renders the template)
    path('face_recognition/', face_recognition, name='face_recognition'),

    # New URLs for the recognition flow
    # This URL provides the video stream for the <img> tag in the template
    path('face_recognition_stream/', face_recognition_stream, name='face_recognition_stream'),

    # This URL is called via JavaScript to check the recognition status
    path('face_recognition_check/', face_recognition_check, name='face_recognition_check'),

    # This URL displays the success page with the recognized user's data
    path('recognized_user/<int:user_id>/', recognized_user, name='recognized_user')
]
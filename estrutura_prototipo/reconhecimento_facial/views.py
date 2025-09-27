import cv2
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.http import StreamingHttpResponse, JsonResponse
from .forms import UserForm, ProcessedPhotosForm
from .models import User, ProcessedPhotos
from .camera import VideoCamera
from datetime import datetime
from django.utils import timezone

# Instance of the VideoCamera class
camera = VideoCamera()


# --- Views for Face Detection and Photo Collection ---

def gen_detect_face(camera):
    """Generator for the facial detection streaming."""
    camera.start_camera()
    while camera.is_streaming:
        frame = camera.detect_face()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    camera.stop_camera()


def face_detection(request):
    """View to display the facial detection streaming."""
    return StreamingHttpResponse(gen_detect_face(camera),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def create_user(request):
    """Creates a new user and redirects to the photo collection page."""
    if request.method == 'POST':
        form = UserForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()
            return redirect(f'{reverse("take_photos", args=[user.id])}?step=1')
    else:
        form = UserForm()
    return render(request, 'create_user.html', {'form': form})


def extract(camera, user_id):
    """
    Function to extract and return the file_path of the faces.
    """
    sample = 0
    number_of_samples = 30
    width, height = 220, 220
    file_paths = []

    camera.start_camera()
    while sample < number_of_samples:
        crop = camera.sample_faces()
        if crop is not None:
            sample += 1
            face = cv2.resize(crop, (width, height))
            gray_image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = f'./tmp/{user_id}_{sample}.jpg'
            cv2.imwrite(file_name_path, gray_image)
            file_paths.append(file_name_path)
        else:
            print("Face not found")

        if sample >= number_of_samples:
            break

    camera.stop_camera()
    return file_paths


def face_extract(context, user):
    """
    Function to manage the process of face extraction and saving.
    """
    num_photos = ProcessedPhotos.objects.filter(user__id=user.id).count()
    print(num_photos)

    if num_photos >= 90:
        context['error'] = 'Maximum number of collections reached.'
    else:
        files_paths = extract(camera, user.id)
        print(files_paths)

        for path in files_paths:
            processed_photo = ProcessedPhotos.objects.create(user=user)
            processed_photo.image.save(os.path.basename(path), open(path, 'rb'))
            os.remove(path)

        context['file_paths'] = ProcessedPhotos.objects.filter(
            user__id=user.id)
        context['extraction_ok'] = True

    return context


def take_photos(request, user_id):
    """View for the user's photo collection flow."""
    step = int(request.GET.get('step', 1))
    extraction_ok = request.GET.get('extraction_ok', 'False') == 'True'
    image_map = {1: 'center.png', 2: 'right.png', 3: 'left.png'}
    instruction_image = image_map.get(step, 'center.png')

    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return redirect(reverse('create_user'))

    if request.method == 'GET' and request.GET.get('clicked') == 'True':
        print(f"Starting face extraction at step {step}...")
        face_extract({}, user)
        return redirect(f'{reverse("take_photos", args=[user.id])}?extraction_ok=True&step={step}')

    context = {
        'user': user,
        'step': step,
        'extraction_ok': extraction_ok,
        'instruction_image': instruction_image,
    }

    if extraction_ok:
        context['file_paths'] = ProcessedPhotos.objects.filter(
            user=user
        ).order_by('-id')[:30]

    return render(request, 'take_photos.html', context)


# --- New Views for Recognition and Redirection ---

def face_recognition(request):
    """
    View that only renders the recognition template.
    """
    return render(request, 'face_recognition.html')


def face_recognition_stream(request):
    """
    View that returns the StreamingHttpResponse to the frontend.
    """

    def gen_recognize_face_stream():
        camera.start_camera()
        while camera.is_streaming:
            frame, _ = camera.recognize_face()
            if frame is None:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        camera.stop_camera()

    return StreamingHttpResponse(gen_recognize_face_stream(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def face_recognition_check(request):
    """
    View called via AJAX to check if a face has been recognized.
    """
    frame, user_id = camera.recognize_face()

    if user_id:
        camera.stop_camera()
        return JsonResponse({
            'status': 'success',
            'user_id': user_id
        })
    return JsonResponse({
        'status': 'waiting',
        'message': 'No face recognized yet.'
    })


def recognized_user(request, user_id):
    """
    View that displays the page with the recognized user's data.
    """
    user = get_object_or_404(User, pk=user_id)
    recognized_at_str = request.GET.get('recognized_at')
    recognized_at = None
    if recognized_at_str:
        try:
            cleaned_str = recognized_at_str.replace('Z', '')
            if '.' in cleaned_str:
                cleaned_str = cleaned_str.split('.')[0]
            dt_naive = datetime.fromisoformat(cleaned_str)
            recognized_at = timezone.make_aware(dt_naive, timezone.utc)
        except ValueError:
            print(f"Error converting date string: {recognized_at_str}")

    context = {
        'user': user,
        'recognized_at': recognized_at,
    }
    return render(request, 'recognized_user.html', context)
from django import forms
from .models import User, ProcessedPhotos


# ModelForm is from the forms import.
class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['photo', 'name']
        # The 'photo' and 'name' fields will be displayed when rendering the User model.
        # The 'unique_id' field is not included as it's generated automatically.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # For each field in the User model, add the "form-control" Bootstrap class to standardize the layout.
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'


# DOC: https://docs.djangoproject.com/en/5.1/topics/http/file-uploads/#uploading-multiple-files
# Multiple Files

# Django has a structure to work with multiple files.
class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = [single_file_clean(data, initial)]
        return result


class ProcessedPhotosForm(forms.ModelForm):
    images = MultipleFileField()

    class Meta:
        model = ProcessedPhotos
        fields = ['images']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'
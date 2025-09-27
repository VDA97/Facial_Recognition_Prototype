from django import forms
from .models import User, ProcessedPhotos

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['photo', 'name']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'

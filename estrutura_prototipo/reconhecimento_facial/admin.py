from django.contrib import admin
from .models import User, ProcessedPhotos, TrainedModel

class ProcessedPhotosInline(admin.StackedInline):
    model = ProcessedPhotos
    extra = 0

class UserAdmin(admin.ModelAdmin):
    # The 'unique_id' field is now a read-only field in the admin panel.
    readonly_fields = ['unique_id']
    inlines = (ProcessedPhotosInline,)

admin.site.register(User, UserAdmin)
admin.site.register(TrainedModel)
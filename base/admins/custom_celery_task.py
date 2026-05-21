from django.contrib import admin
from django_celery_results.models import TaskResult
from django_celery_results.admin import TaskResultAdmin

class CustomTaskResultAdmin(TaskResultAdmin):
    # Extend the default list_display and add extra fields
    list_display = TaskResultAdmin.list_display + ('result',)

admin.site.unregister(TaskResult)
admin.site.register(TaskResult, CustomTaskResultAdmin)







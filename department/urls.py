from django.urls import path
from . import views

app_name = 'department'

urlpatterns = [
    path('', views.list_departments, name='list_departments'),
    path('create/', views.create_department, name='create_department'),
    path('<uuid:dept_id>/update/', views.update_department, name='update_department'),
    path('<uuid:dept_id>/delete/', views.delete_department, name='delete_department'),
]








from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/preprocess/', views.api_preprocess, name='api_preprocess'),
    path('api/train/', views.api_train, name='api_train'),
    path('api/evaluate/', views.api_evaluate, name='api_evaluate'),
    path('api/capture/', views.api_capture, name='api_capture'),
    path('api/detect/', views.api_detect, name='api_detect'),
    path('api/task_status/', views.api_task_status, name='api_task_status'),
    path('api/export_log/', views.api_export_log, name='api_export_log'),
    path('api/result_image/<str:image_name>/', views.api_result_image, name='api_result_image'),
    path('api/capture_data/', views.api_capture_data, name='api_capture_data'),
    path('api/detection_result/', views.api_detection_result, name='api_detection_result'),
]

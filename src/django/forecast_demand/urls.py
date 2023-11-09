from . import views
from django.urls import path
from rest_framework_simplejwt import views as jwt_views

urlpatterns = [
    path('delete_model/', views.eliminar_modelo.as_view()),
    path('model_description/', views.detalles_modelo.as_view()),
    path('models_list/', views.lista_modelos.as_view()),
    path('create_model/', views.crear_modelo.as_view()),
    path('model_inference/', views.inferencia.as_view()),
    
]

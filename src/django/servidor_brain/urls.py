"""servidor_brain URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView
from rest_framework_simplejwt import views as jwt_views
# Documentacion Automatica de Swagger



urlpatterns = [
    path('admin/', admin.site.urls),
    path('classifier/', include('classifier.urls')),
    path('forecast_demand/', include('forecast_demand.urls')),
    path('schema/', SpectacularAPIView.as_view(), name="schema"),
    path('docs/', SpectacularSwaggerView.as_view(url_name="schema")),
    path('authentication/get-token/', jwt_views.TokenObtainPairView.as_view()),
    path('authentication/refresh-token/', jwt_views.TokenRefreshView.as_view()),
]
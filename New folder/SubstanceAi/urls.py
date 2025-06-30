"""
URL configuration for test_pi project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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
from django.urls import include, path
from authapp import urls
from profilapp import urls
from .views import *


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/auth/', include('authapp.urls')),
    path('api/profil/', include('profilapp.urls')),
    #path('profilapp/', include('profilapp.urls')),
    
    path('api/auth/upload-pdf/',upload_pdf, name='upload_pdf'),
    path('api/auth/list-pdfs/<str:user_id>/',list_pdfs, name='list_pdfs'),
    path('api/auth/download_pdf/<str:file_id>/', download_pdf, name='download_pdf'),
    path('api/rag-question/', send_to_rag, name='rag_question'),

]


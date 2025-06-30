from django.urls import path , include
from .views import register, login
from .views import  google_callback

urlpatterns = [
    path('register/', register, name='register'),
    path('login/', login, name='login'),
    path('login/google/', google_callback, name='google_callback'),
    path('login/google/callback/', google_callback, name='google_callback'), 
     
]

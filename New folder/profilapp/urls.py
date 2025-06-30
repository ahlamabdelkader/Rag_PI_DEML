from django.urls import path
from .views import enregistrer_reponses,recuperer_reponses


urlpatterns = [
    path("enregistrer_reponses/", enregistrer_reponses, name="enregistrer_reponses"),
    path("recuperer_reponses/<str:user_id>/", recuperer_reponses, name="recuperer_reponses"),
    
]



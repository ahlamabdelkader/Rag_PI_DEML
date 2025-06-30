from django.contrib.auth.backends import BaseBackend
from werkzeug.security import check_password_hash
from .models import UserManager  # Assure-toi que UserManager est importé depuis ton modèle

class MongoDBBackend(BaseBackend):
    def authenticate(self, request, username=None, password=None):
        user_data = UserManager().get_user_by_email(username)
        if user_data and check_password_hash(user_data['password'], password):
            return user_data
        return None

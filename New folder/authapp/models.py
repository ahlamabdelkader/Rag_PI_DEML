
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
# Connexion à MongoDB
MONGO_URI = "mongodb://localhost:27017/SubstanceAi"
client = MongoClient(MONGO_URI)
db = client.get_database()
users_collection = db["users"]  # Collection MongoDB pour stocker les utilisateurs

class UserManager:
    def create_user(self, email, fullname, username, password=None):
        if not email:
            raise ValueError("L'utilisateur doit avoir un email")
        
        password_hash = generate_password_hash(password)
        user_data = {
            "email": email,
            "fullname": fullname,
            "username": username,
            "password": password_hash,
            "is_active": True,
            "is_admin": False,
            "created_at": datetime.datetime.utcnow()
        }
        users_collection.insert_one(user_data)
        return user_data

    def create_superuser(self, email, fullname, username, password):
        user_data = self.create_user(email, fullname, username, password)
        users_collection.update_one(
            {"email": email},
            {"$set": {"is_admin": True}}
        )
        return user_data

class User:
    def __init__(self, email, fullname, username, password):
        self.email = email
        self.fullname = fullname
        self.username = username
        self.password_hash = generate_password_hash(password)
        self.is_active = True
        self.is_admin = False
        self.created_at = datetime.datetime.utcnow() 

    def save(self):
        user_data = {
            "email": self.email,
            "fullname": self.fullname,
            "username": self.username,
            "password": self.password_hash,
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "created_at": datetime.datetime.utcnow()
        }
        users_collection.insert_one(user_data)


    @staticmethod
    def find_by_username(username):
        # Recherche d'un utilisateur par username
        return users_collection.find_one({"username": username})
    @staticmethod
    def find_by_email(email):
        return users_collection.find_one({"email": email})

    @staticmethod
    def verify_password(stored_password, provided_password):
        return check_password_hash(stored_password, provided_password)










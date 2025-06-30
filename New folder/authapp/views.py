
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from .models import User, UserManager
from .serializers import RegisterSerializer, UserSerializer
from django.shortcuts import redirect
from social_django.utils import psa
import requests
from django.http import JsonResponse
from django.contrib.auth import login
from pymongo import MongoClient
from rest_framework_simplejwt.tokens import RefreshToken
import jwt
from datetime import datetime, timedelta
from django.conf import settings
from bson import ObjectId
from rest_framework.decorators import api_view
from .decorators import jwt_authentication


# Connexion à MongoDB
MONGO_URI = "mongodb://localhost:27017/SubstanceAi"
client = MongoClient(MONGO_URI)
db = client.get_database()
user_manager = UserManager()
# Accéder à la collection "users"
users_collection = db.users

def create_jwt_token(user):
    payload = {
        "user_id": str(user['_id']),
        "email": user['email'],
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow()
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
    return token
@api_view(['POST'])
def register(request):
    serializer = RegisterSerializer(data=request.data)
    if serializer.is_valid():
        data = serializer.validated_data
        # Utilisation du UserManager pour créer un utilisateur dans MongoDB
        user = user_manager.create_user(
            email=data['email'],
            fullname=data['fullname'],
            username=data['username'],
            password=data['password'],
                
        )
        token = create_jwt_token(user)

        return Response({
            "message": "Inscription réussie",
            "token": token,
            "user": {
                "email": user['email'],
                "username": user['username'],
                "_id": str(user['_id'])
            },
            "redirect_url": "http://localhost:3000/home"

        }, status=status.HTTP_201_CREATED)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def login(request):
    identifier = request.data.get('identifier')
    password = request.data.get('password')

    if '@' in identifier:
        user_data = User.find_by_email(identifier)
    else:
        user_data = User.find_by_username(identifier)

    if user_data and User.verify_password(user_data['password'], password):
        user_data['_id'] = str(user_data['_id'])
        
        token = create_jwt_token(user_data)
        
        return Response({
            "message": "Connexion réussie", 
            "user": user_data,
            "token": token,
            "redirect_url": f"http://localhost:3000/home"
        }, status=status.HTTP_200_OK)

    return Response({"error": "Email/Username ou mot de passe incorrect"}, status=status.HTTP_401_UNAUTHORIZED)


GOOGLE_CLIENT_ID = "76497721292-2fmahu68t6r2vaiupdmq6rbbtqsm3jq5.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"
REDIRECT_URI = "http://localhost:8000/api/auth/login/google/callback/"


def google_callback(request):

    code = request.GET.get("code")

    if not code:

        return JsonResponse({"error": "Code not provided"}, status=400)


    # Récupération du token

    token_url = "https://oauth2.googleapis.com/token"

    data = {

        "code": code,

        "client_id": GOOGLE_CLIENT_ID,

        "client_secret": GOOGLE_CLIENT_SECRET,

        "redirect_uri": REDIRECT_URI,

        "grant_type": "authorization_code",

    }

    response = requests.post(token_url, data=data)

    token_info = response.json()


    if "access_token" not in token_info:

        return JsonResponse({"error": "Failed to obtain access token"}, status=400)


    access_token = token_info["access_token"]


    # Récupération des infos utilisateur

    user_info_url = "https://www.googleapis.com/oauth2/v1/userinfo"

    headers = {"Authorization": f"Bearer {access_token}"}

    user_response = requests.get(user_info_url, headers=headers)


    if user_response.status_code != 200:

        return JsonResponse({"error": "Failed to fetch user info"}, status=400)


    user_data = user_response.json()


    # Vérifier si l'utilisateur existe déjà

    user = users_collection.find_one({"email": user_data['email']})

    if not user:
        result = users_collection.insert_one({
            "email": user_data['email'],
            "name": user_data.get("name"),
            "profile_picture": user_data.get("picture"),
            "google_id": user_data.get("id"),
            "verified_email": user_data.get("verified_email"),
        })
        user = users_collection.find_one({"_id": result.inserted_id})

    # Génère un token JWT
    token = create_jwt_token(user)
    print("Token JWT généré :", token)
    # Redirige vers le front en passant le token dans l'URL (ou autre moyen sécurisé)
    redirect_url = f"http://localhost:3000/google-auth-success?token={token}"
    return redirect(redirect_url)



@api_view(['GET'])
@jwt_authentication
def home(request):
    # Ici tu as accès à request.user_payload avec les données du JWT
    user_info = request.user_payload
    return Response({"message": f"Bienvenue {user_info.get('username', 'Utilisateur')}"})


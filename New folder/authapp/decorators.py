from rest_framework.response import Response
from rest_framework import status
import jwt
from django.conf import settings

def jwt_authentication(function):
    def wrapper(request, *args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return Response({"error": "Authorization header manquant ou invalide"}, status=status.HTTP_401_UNAUTHORIZED)
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])
            request.user_payload = payload  # Stocker les infos utilisateur dans la requête
        except jwt.ExpiredSignatureError:
            return Response({"error": "Token expiré"}, status=status.HTTP_401_UNAUTHORIZED)
        except jwt.InvalidTokenError:
            return Response({"error": "Token invalide"}, status=status.HTTP_401_UNAUTHORIZED)
        return function(request, *args, **kwargs)
    return wrapper

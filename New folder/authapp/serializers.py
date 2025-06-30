
from rest_framework import serializers
from .models import User
import datetime
class UserSerializer(serializers.Serializer):
    email = serializers.EmailField()
    fullname = serializers.CharField(max_length=255)
    username = serializers.CharField(max_length=255)
    password = serializers.CharField(write_only=True)
    created_at = serializers.DateTimeField()

    # def create(self, validated_data):
    #     # Tu pourrais ici utiliser UserManager pour créer un utilisateur dans MongoDB
    #     user = UserManager().create_user(
    #         validated_data['email'],
    #         validated_data['fullname'],
    #         validated_data['username'],
    #         validated_data['password']
    #     )
    #     return user

    # def update(self, instance, validated_data):
    #     instance.email = validated_data.get('email', instance.email)
    #     instance.fullname = validated_data.get('fullname', instance.fullname)
    #     instance.username = validated_data.get('username', instance.username)
    #     instance.password = validated_data.get('password', instance.password)
    #     # Ajoute ici la logique pour mettre à jour un utilisateur
    #     return instance

class RegisterSerializer(serializers.Serializer):
    email = serializers.EmailField()
    fullname = serializers.CharField(max_length=255)
    username = serializers.CharField(max_length=255)
    password = serializers.CharField(write_only=True)

    # def create(self, validated_data):
    #     # Utilise UserManager pour créer un utilisateur dans MongoDB
    #     return UserManager().create_user(
    #         validated_data['email'],
    #         validated_data['fullname'],
    #         validated_data['username'],
    #         validated_data['password']
    #     )

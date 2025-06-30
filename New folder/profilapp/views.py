from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from pymongo import MongoClient

# Connexion à MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["SubstanceAi"]
collection = db["profile"]

@csrf_exempt
def enregistrer_reponses(request):
    """API pour enregistrer ou mettre à jour les réponses du test de profilage via Streamlit."""
    if request.method == "POST":
        try:
            data = json.loads(request.body)  # Récupération des données JSON
            user_id = data.get("user_id")  # Assurez-vous que user_id est inclus dans les données

            # Vérifier si un profil pour cet utilisateur existe déjà
            existing_profile = collection.find_one({"user_id": user_id})

            if existing_profile:
                # Mettre à jour le profil existant
                collection.update_one(
                    {"_id": existing_profile["_id"]},
                    {"$set": data},  # Mettre à jour toutes les informations du profil
                    upsert=False  # Pas besoin d'insertion si le profil existe déjà
                )
                return JsonResponse({"message": "Profil mis à jour avec succès"}, status=200)
            else:
                # Si aucun profil trouvé, créer un nouveau profil
                data["user_id"] = user_id  # Assurez-vous que l'ID utilisateur est dans les données
                collection.insert_one(data)  # Insertion du profil dans MongoDB
                return JsonResponse({"message": "Nouveau profil créé"}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    
    return JsonResponse({"error": "Méthode non autorisée"}, status=405)

@csrf_exempt
def recuperer_reponses(request, user_id):
    """API pour récupérer les réponses d’un utilisateur donné."""
    try:
        reponses = list(collection.find({"user_id": user_id}, {"_id": 0}))  # Ne pas inclure l'_id
        return JsonResponse({"reponses": reponses}, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)
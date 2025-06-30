from django.http import FileResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from bson import ObjectId
from .utils import *
from pymongo import MongoClient
from bson import ObjectId
from gridfs import GridFS
from gradio_client import Client
import tempfile

@csrf_exempt
def upload_pdf(request):
    if request.method == 'POST' and request.FILES:
        user_id = request.POST.get('user_id')
        message = request.POST.get('message', '')
        uploaded_file_ids = []

        try:
            files = request.FILES.getlist('file')
            for f in files:
                file_id = save_pdf_to_mongodb(f, user_id, message)
                uploaded_file_ids.append(str(file_id))

            return JsonResponse({
                'message': 'Fichiers enregistrés avec succès',
                'file_ids': uploaded_file_ids
            })

        except Exception as e:
            traceback.print_exc()  # pour debug dans la console
            return JsonResponse({'error': 'Erreur lors de l\'enregistrement', 'details': str(e)}, status=500)

    return JsonResponse({'error': 'Aucun fichier reçu'}, status=400)


def list_pdfs(request, user_id):
    """Retourne la liste des fichiers PDF uploadés par un utilisateur spécifique"""
    try:
        files = list_user_pdfs(user_id)
        return JsonResponse({"files": files})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
    
    

def download_pdf(request, file_id):
    try:
        # Conversion de l'ID en ObjectId MongoDB
        file_id = ObjectId(file_id)
        print(f"Récupération du fichier avec ID: {file_id}")
        
        # Connexion à MongoDB
        MONGO_URI = "mongodb://localhost:27017/SubstanceAi"
        client = MongoClient(MONGO_URI)
        db = client.get_database()

        fs = GridFS(db)
        
        # Récupération du fichier
        file = fs.get(file_id)
        if not file:
            return JsonResponse({'error': 'Fichier non trouvé'}, status=404)
        
        # Création de la réponse
        response = FileResponse(
            file,
            content_type='application/pdf',
            as_attachment=True,
            filename=file.filename
        )
        return response
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)




@csrf_exempt
def send_to_rag(request):
    if request.method == 'POST' and request.FILES:
        pdf_file = request.FILES['file']
        question = request.POST.get('question', '')

        try:
            # Sauvegarde temporaire du fichier
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                for chunk in pdf_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name

            # Appel du modèle Hugging Face avec gradio_client
            client = Client("https://ahlamm-rag-api-m.hf.space/--replicas/fd22w/")
            result = client.predict(
                tmp_path,        # 📄 Upload PDF
                question,        # ❓ Pose ta question
                api_name="/predict"
            )

            return JsonResponse({"answer": result})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Méthode non autorisée"}, status=405)

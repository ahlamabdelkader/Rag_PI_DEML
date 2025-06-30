import gridfs
from pymongo import MongoClient

# Connexion à MongoDB
MONGO_URI = "mongodb://localhost:27017/SubstanceAi"
client = MongoClient(MONGO_URI)
db = client.get_database()
users_collection = db["pdfs"]  # Collection MongoDB pour stocker les utilisateurs
fs = gridfs.GridFS(db)

import gridfs
from pymongo import MongoClient

# Connexion à MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["SubstanceAi"]
fs = gridfs.GridFS(db)

def save_pdf_to_mongodb(file, user_id, message):
    """Sauvegarde un fichier PDF dans MongoDB GridFS avec l'ID utilisateur et le message (prompt)"""
    file_id = fs.put(
        file.read(),
        filename=file.name,
        user_id=user_id,
        message=message  # Ajout du message
    )
    return file_id

def list_user_pdfs(user_id):
    files = users_collection.find({"user_id": user_id})
    return [
        {
            "_id": str(file["_id"]),
            "filename": file.get("filename", "inconnu"),
            "length": file.get("length", 0)
        }
        for file in files
    ]


def get_pdf_from_mongodb(file_id):
    """Récupère un fichier PDF stocké dans MongoDB GridFS"""
    return fs.get(file_id)

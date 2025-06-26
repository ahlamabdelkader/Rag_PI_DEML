# PDF RAG API

Une API FastAPI qui reçoit un PDF et une question, puis retourne une réponse générée par un LLM à partir des chunks du document.

## Endpoint

POST /query/

- FormData:
  - `pdf`: fichier PDF
  - `question`: question texte

## Réponse JSON

```json
{
  "answer": "Réponse générée par le modèle",
  "context": [
    {"chunk": "...", "page": 2},
    {"chunk": "...", "page": 4}
  ]
}
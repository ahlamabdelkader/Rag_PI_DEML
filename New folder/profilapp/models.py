from django.db import models

class UserResponse(models.Model):
    user_id = models.CharField(max_length=255)
    question = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Réponse de {self.user_id}"

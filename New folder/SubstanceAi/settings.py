from corsheaders.defaults import default_headers

from pathlib import Path
from pymongo import MongoClient

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-(6=61d!kjg5sv135dd_icnq2q&94-aab1eqai(mxe&19kqb6fw'
DEBUG = True
ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'authapp',
    'rest_framework',
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.google',
    'social_django',
    'profilapp',
    'corsheaders',
]

AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
    'allauth.account.auth_backends.AuthenticationBackend',
    'django.contrib.auth.backends.ModelBackend',  # Si tu veux quand même utiliser Django auth par défaut pour certaines fonctions
    'authapp.backends.MongoDBBackend',
    'social_core.backends.google.GoogleOAuth2',
]
SOCIAL_AUTH_GOOGLE_OAUTH2_KEY = "76497721292-2fmahu68t6r2vaiupdmq6rbbtqsm3jq5.apps.googleusercontent.com"
SOCIAL_AUTH_GOOGLE_OAUTH2_SECRET = "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"
# SOCIAL_AUTH_GOOGLE_OAUTH2_REDIRECT_URI = "http://localhost:8000/complete/google-oauth2/"

SOCIAL_AUTH_GOOGLE_OAUTH2_SCOPE = [
    'email',
    'profile',
]
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = '/'
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
}

SITE_ID = 1

SOCIALACCOUNT_PROVIDERS = {
    'google': {
        'SCOPE': ['profile', 'email'],
        'AUTH_PARAMS': {'access_type': 'online'},
    }
}

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'allauth.account.middleware.AccountMiddleware',  
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',

]

ROOT_URLCONF = 'SubstanceAi.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'SubstanceAi.wsgi.application'

# ⚠️ Désactivation de l'ORM de Django
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.dummy',  # Django ORM désactivé
    }
}

# ✅ Connexion à MongoDB avec PyMongo
MONGO_URI = "mongodb://localhost:27017/SubstanceAi"
client = MongoClient(MONGO_URI)
db = client.get_database()


AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]
CORS_ALLOW_CREDENTIALS = True

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",  
]
CORS_ALLOW_HEADERS = list(default_headers) + [
    'content-type',
    'authorization',
    'x-csrftoken',
    'x-requested-with',
]
CORS_ALLOW_ALL_ORIGINS = True

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

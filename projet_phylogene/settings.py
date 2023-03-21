"""
Django settings for projet_phylogene project.

Generated by 'django-admin startproject' using Django 2.1.7.

For more information on this file, see
https://docs.djangoproject.com/en/2.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.1/ref/settings/
"""

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '@%(6xvusj!)t36!of4%_(jlv!z2v-a_-v+26k7_^x$l0uv1uj6'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']
# ALLOWED_HOSTS = ['http://localhost:8000']
CORS_ORIGIN_ALLOW_ALL = False
CORS_ORIGIN_WHITELIST = (
    'https://*',
)

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'crispy_forms',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # add manually
    'corsheaders',
    'phylogene_app.apps.PhylogeneAppConfig',

]
CRISPY_TEMPLATE_PACK = 'bootstrap4'

#AUTH_USER_MODEL = 'users.CustomUser'

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',

    'corsheaders.middleware.CorsMiddleware',
]

ROOT_URLCONF = 'projet_phylogene.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates'), ],
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

WSGI_APPLICATION = 'projet_phylogene.wsgi.application'

# Database
# https://docs.djangoproject.com/en/2.1/ref/settings/#databases

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.postgresql_psycopg2', # on utilise l'adaptateur postgresql
#         'NAME': 'phylogene', # le nom de notre base de donnees creee precedemment
#         'USER': 'postgres', # attention : remplacez par votre nom d'utilisateur
#         'PASSWORD': 'groscap971',
#         'HOST': 'localhost',
#         'PORT': '5432',
#     }
# }
"""pc home
CURRENT_DIR = '/home/freezer/Documents/PhylEntropy'

pc ipg
CURRENT_DIR = '/home/linuxipg/Documents/PhylEntropy'
"""


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR + '/mydatabase.sqlite3',  # <- The path
        'USER': '',
        'PASSWORD': '',
        'HOST': '',
        'PORT': '8000',
    }
}

DATA_UPLOAD_MAX_MEMORY_SIZE = 5242880

# Password validation
# https://docs.djangoproject.com/en/2.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/2.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.1/howto/static-files/

# STATIC_URL = '/static/'

STATIC_URL = os.path.join(BASE_DIR, '/static/')

LOGIN_REDIRECT_URL = '/'
LOGIN_URL = 'login'

STATICFILES_DIRS = [
    os.path.join(BASE_DIR, '/phylogene_app/static/'),
    os.path.join(BASE_DIR, '/phylogene_app/static/files/'),
]

''' pc home
STATICFILES_DIRS = [
    '/home/freezer/Documents/PhylEntropy/phylogene_app/static',
    '/home/freezer/Documents/PhylEntropy/phylogene_app/static/files',
]
'''

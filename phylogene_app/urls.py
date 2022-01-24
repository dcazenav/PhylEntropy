from django.urls import path
from . import views

urlpatterns=[
    path('', views.intro, name='intro'),
    path('import_data/',views.import_data, name='phylogene_import'),
    path('ajax1/', views.ajax_1, name='phylogene_ajax1'),
    path('run_algo/', views.run_algo, name='phylogene_run_algo'),
    path('test/', views.test, name='phylogene_test'),
    path('base/', views.base, name='base'),
]
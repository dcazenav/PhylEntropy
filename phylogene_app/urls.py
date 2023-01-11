from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
                  path('', views.intro, name='intro'),
                  path('import_data/', views.import_data, name='phylogene_import'),
                  path('ajax1/', views.ajax_1, name='phylogene_ajax1'),
                  path('run_algo/', views.run_algo, name='phylogene_run_algo'),
                  path('base/', views.base, name='base'),
                  path('about/', views.aboutphylentropy, name='aboutphylentropy'),
                  path('links/', views.links, name='links'),
                  path('credits/', views.credits, name='credits'),
                  # path('test/', views.test, name='phylogene_test'),
                  # path('Rtest/', views.run_algo, name='Rtest'),
                  # path('chart/', views.chart, name='chart'),
              ] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

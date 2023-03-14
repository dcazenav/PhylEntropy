from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views

urlpatterns = [
                path('', views.intro, name='intro'),
                path('import_data/', views.import_data, name='phylogene_import'),
                path('ajax1/', views.ajax_1, name='phylogene_ajax1'),
                path('run_algo/', views.run_algo, name='phylogene_run_algo'),
                path('base/', views.base, name='base'),
                path('about/', views.aboutphylentropy, name='aboutphylentropy'),
                path('links/', views.links, name='links'),
                path('credits/', views.credits, name='credits'),
                path("register", views.register, name="register"),
                path('login', auth_views.LoginView.as_view(template_name='users/login.html'), name='login'),
                path('logout', auth_views.LogoutView.as_view(template_name='users/logout.html'), name='logout'),
                # path('test/', views.test, name='phylogene_test'),
                # path('Rtest/', views.run_algo, name='Rtest'),
                # path('chart/', views.chart, name='chart'),
              ] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

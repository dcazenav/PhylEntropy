from django.urls import path, include
from . import views
# from schema_graph.views import Schema
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views

urlpatterns = [
                path('', views.intro, name='intro'),
                path('import_data/', views.import_data, name='phylogene_import'),
                path('import_data_other_tools/', views.import_data_other_tool, name='phylogene_import_other'),
                path('ajax1/', views.ajax_1, name='phylogene_ajax1'),
                path('ajax_other_tools/', views.ajax_other_tools, name='phylogene_ajax_other_tools'),
                path('run_algo/', views.run_algo, name='phylogene_run_algo'),
                path('run_algo_other_tools/', views.run_algo_other_tools, name='phylogene_run_algo_other_tools'),
                path('base/', views.base, name='base'),
                path('about/', views.aboutphylentropy, name='aboutphylentropy'),
                path('links/', views.links, name='links'),
                path('credits/', views.credits, name='credits'),
                path("register", views.register, name="register"),
                path('login', auth_views.LoginView.as_view(template_name='users/login.html'), name='login'),
                path('logout', auth_views.LogoutView.as_view(template_name='users/logout.html'), name='logout'),
                path('profile/<username>', views.profile, name="profile"),
                path('adduserfile', views.adduserfile, name='adduserfile'),
                #path('delete/<int:id>/', views.delete_file, name='delete-file')
                path('django_plotly_dash/', include('django_plotly_dash.urls')),
                path('delete/<int:pk>/', views.delete_file, name='delete-file' ),
                path('other_tools/', views.genomics, name='other_tools'),
                # path("schema/", Schema.as_view()), (à retirer en production)
                # path('listfiles', views.listfiles, name='listfiles'),
                # path('test/', views.test, name='phylogene_test'),
                # path('Rtest/', views.run_algo, name='Rtest'),
                # path('chart/', views.chart, name='chart'),
              ] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

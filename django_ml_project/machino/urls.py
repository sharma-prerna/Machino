from django.urls import path, include
from . import views
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='machino-home'),
    path('knn/',views.knn, name='knn'),
    path('deep/',views.deep, name='deep'),
    path('lin/',views.lin, name='lin'),
    path('log/',views.log, name='log'),
    path('naive/',views.naive, name='naive'),
    path('kmc/',views.kmc, name='kmc'),
]

from django.urls import path
from . import views

urlpatterns = [
    path('navios/', views.NavioList.as_view(), name='navio-list')

]
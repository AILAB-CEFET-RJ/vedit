
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^navios/$', views.NavioList.as_view(), name='navio-list'),

]
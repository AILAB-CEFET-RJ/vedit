from rest_framework import generics
from .models import Navio
from .serializers import NavioSerializer

# Create your views here.
class NavioList(generics.ListCreateAPIView):

    queryset = Navio.objects.all()
    serializer_class = NavioSerializer
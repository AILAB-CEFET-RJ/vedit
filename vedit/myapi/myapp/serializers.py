from rest_framework import serializers
from .models import Navio

class MusicSerializer(serializers.ModelSerializer):

    class Meta:

        model = Navio
        fields = '__all__'
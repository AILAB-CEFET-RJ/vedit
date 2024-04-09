from django.db import models
# Create your models here.


class Navio(models.Model):

    class Meta:

        db_table = 'navio'

    title = models.CharField(max_length=200)
    seconds = models.IntegerField()

    def __str__(self):
        return self.title
# Create your models here.

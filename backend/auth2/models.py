from django.db import models
from django.contrib.auth.models import AbstractUser

class Gender(models.TextChoices):
    MALE = 'M', "Male"
    FEMALE = 'F', "Female"
    OTHER = 'O', "Other"

class Timezone(models.Model):
    name = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.name}"
    
class Language(models.Model):
    code = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=50)

    def __str__(self):
        return self.name
    
class Location(models.Model):
    city = models.CharField(max_length=100)
    country = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.city}, {self.country}"

class User(AbstractUser):
    phone_number = models.CharField(max_length=15,blank=True,null=True)
    date_of_birth = models.DateField(blank=True,null=True)
    avatar = models.ImageField(
        upload_to="avatars/",
        null=True,
        blank=True
    )
    bio = models.TextField(blank=True,null=True)
    gender = models.CharField(max_length=1, choices=Gender.choices, null=True, blank=True)
    timezone = models.ForeignKey(Timezone, on_delete=models.SET_NULL, null=True, blank=True)
    language = models.ForeignKey(Language, on_delete=models.SET_NULL, blank=True, null=True)
    location = models.ForeignKey(Location, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return self.username
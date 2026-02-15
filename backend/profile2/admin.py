from django.contrib import admin
from auth2.models import Language, Timezone, Location

admin.site.register(Language)
admin.site.register(Timezone)
admin.site.register(Location)

# Register your models here.

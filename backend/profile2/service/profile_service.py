from django.contrib.auth import get_user_model
from auth2.models import Language, Location, Timezone
from django.http import QueryDict
User = get_user_model()

class ProfileService:
    def __init__(self):
        pass

    def get_profile(self, user_id):
        user = User.objects.filter(id=user_id).first()
        roles = user.groups.all() if user else []
        if not user:
            return {
                'error': 'User not found'
            }
        roles_map = [role.name for role in roles]
        print(roles_map, roles)
        user_profile = {
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'phone_number': user.phone_number,
            'date_of_birth': user.date_of_birth,
            'bio': user.bio,
            'gender': user.gender,
            'timezone': user.timezone.name if user.timezone else None,
            'language': user.language.name if user.language else None, 
            'location': f"{user.location.city}, {user.location.country}" if user.location else None, 
            'avatar_url': user.avatar.url if user.avatar else None,
            'title': roles_map
        }
        return {
            'message': user_profile
        }
    
    
    def update_profile(self, user_id, data, files=None):
        user = User.objects.filter(id=user_id).first()
        if not user:
            return {"error": "User not found"}

        # --- basic fields ---
        simple_fields = [
            "username",
            "email",
            "first_name",
            "last_name",
            "phone_number",
            "date_of_birth",
            "bio",
            "gender",
        ]

        for field in simple_fields:
            value = data.get(field)

            if value is not None and value != "":
                setattr(user, field, value)

        if data.get("timezone"):
            tz = Timezone.objects.filter(name=data["timezone"]).first()
            if tz:
                user.timezone = tz

        if data.get("language"):
            lang = Language.objects.filter(name=data["language"]).first()
            if lang:
                user.language = lang

        # if data.get("location"):
        #     loc = Location.objects.filter(name=data["location"]).first()
        #     if loc:
        #         user.location = loc

        if files and files.get("avatar"):
            user.avatar = files["avatar"]

        user.save()

        return {"message": "Profile updated successfully"}
    
    def get_language(self,offset=0, limit=10):
        languages = Language.objects.all()[offset:offset+limit]
        language_list = [{'id': lang.id, 'name': lang.name} for lang in languages]
        return {
            'message': language_list
        }
    
    def get_timezone(self,offset=0, limit=10):
        timezones = Timezone.objects.all()[offset:offset+limit]
        timezone_list = [{'id': tz.id, 'name': tz.name} for tz in timezones]
        return {
            'message': timezone_list
        }
    
    def get_location(self,offset=0, limit=10):
        locations = Location.objects.all()[offset:offset+limit]
        location_list = [{'id': loc.id, 'city': loc.city, 'country': loc.country} for loc in locations]
        return {
            'message': location_list
        }
    
    def convert_formdata_to_dict(self, req):
        print("POST:", req.POST)
        form_data = req.POST.dict()

        result = {}
        for key in form_data:
            result[key] = form_data.get(key)

        return result

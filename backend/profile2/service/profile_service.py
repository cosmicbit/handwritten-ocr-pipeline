from django.contrib.auth.models import User

class ProfileService:
    def __init__(self):
        pass

    def get_profile(self, user_id):
        user = User.objects.filter(id=user_id).first()
        if not user:
            return {
                'error': 'User not found'
            }
        user_profile = {
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'phone_number': user.phone_number,
            'date_of_birth': user.date_of_birth
        }
        return {
            'message': user_profile
        }
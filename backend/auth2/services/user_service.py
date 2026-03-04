from ..dtos.user_registeration import UserRegistration
from django.contrib.auth import get_user_model
User = get_user_model()

from ..exceptions.user_already_exist import UserAlreadyExist
from ..dtos.user_login import UserLogin
from ..jwt_utils import create_jwt_token
from django.contrib.auth.models import Group
from auth2.models import Role


class UserService:

    ALLOWED_ROLES = {Role.INSTITUTION, Role.TEACHER, Role.STUDENT}

    def _normalize_role(self, role_value):
        if role_value is None:
            raise ValueError("role is required")

        role = str(role_value).strip().lower()
        if role not in self.ALLOWED_ROLES:
            raise ValueError("role must be one of: institution, teacher, student")
        return role

    def _assign_role_group(self, user, role):
        group, _ = Group.objects.get_or_create(name=role)
        user.groups.clear()
        user.groups.add(group)

    def _user_payload(self, user):
        role = getattr(user, "role", None) or Role.STUDENT
        return {
            "username": user.username,
            "is_superAdmin": user.is_superuser,
            "role": role,
        }

    def build_auth_response(self, user, token, info_message):
        payload = {
            "token": token,
            "user": self._user_payload(user),
        }
        return {
            "message": payload,
            "info": info_message,
        }

    def register(self, data):
        
        userRegisteration = UserRegistration()
        userRegisteration.serialize(data)
        role = self._normalize_role(userRegisteration.role.value)

        if User.objects.filter(username=userRegisteration.username.value).exists():
            raise UserAlreadyExist("user already exists")

        user = User.objects.create_user(username=userRegisteration.username.value,
                                         password=userRegisteration.password.value,
                                         email=userRegisteration.email.value,
                                         role=role,
                                         #phone_number=userRegisteration.phoneNumber
                                         )
        self._assign_role_group(user=user, role=role)
        token = create_jwt_token(user=user)
        return self.build_auth_response(user=user, token=token, info_message="User registered")
    
    def login(self, data):
        userLogin = UserLogin()
        userLogin.serialize(data=data)

    
    def get_group(self, user):
        return user.groups.all()

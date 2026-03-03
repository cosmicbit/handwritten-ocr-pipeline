import json

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TestCase


User = get_user_model()


class AuthRoleTests(TestCase):
    def setUp(self):
        for group_name in ("student", "teacher", "institution"):
            Group.objects.get_or_create(name=group_name)

    def _register(self, username, email, role):
        payload = {
            "email": email,
            "username": username,
            "password": "Pass@12345",
            "password_confirmation": "Pass@12345",
            "role": role,
        }
        return self.client.post(
            "/auth/register",
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_register_succeeds_for_supported_roles(self):
        for role in ("institution", "teacher", "student"):
            response = self._register(
                username=f"{role}_user",
                email=f"{role}@example.com",
                role=role,
            )
            self.assertEqual(response.status_code, 201)

            body = response.json()
            self.assertEqual(body["message"]["user"]["role"], role)
            self.assertTrue(body["message"]["token"])

            user = User.objects.get(username=f"{role}_user")
            self.assertEqual(user.role, role)
            self.assertTrue(user.groups.filter(name=role).exists())

    def test_register_rejects_invalid_role(self):
        response = self._register(
            username="invalid_role_user",
            email="invalid@example.com",
            role="admin",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["error"],
            "role must be one of: institution, teacher, student",
        )

    def test_register_rejects_missing_role(self):
        payload = {
            "email": "missing@example.com",
            "username": "missing_role_user",
            "password": "Pass@12345",
            "password_confirmation": "Pass@12345",
        }
        response = self.client.post(
            "/auth/register",
            data=json.dumps(payload),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "role is required")

    def test_login_response_includes_single_role(self):
        user = User.objects.create_user(
            username="teacher_login",
            email="teacher_login@example.com",
            password="Pass@12345",
            role="teacher",
        )
        user.groups.clear()
        user.groups.add(Group.objects.get(name="teacher"))

        response = self.client.post(
            "/auth/login",
            data=json.dumps({"username": "teacher_login", "password": "Pass@12345"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["message"]["user"]["role"], "teacher")
        self.assertTrue(body["message"]["token"])

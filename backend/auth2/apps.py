from django.apps import AppConfig


class Auth2Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'auth2'

    def ready(self):
        import auth2.signals  # noqa: F401

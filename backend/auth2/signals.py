import logging

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group, Permission
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.db.models.signals import post_migrate


logger = logging.getLogger(__name__)
User = get_user_model()

GROUP_PERMISSION_MAP = {
    "student": [
        "view_user",
        "view_language",
        "view_location",
        "view_timezone",
        "view_group",
        "view_permission",
        "view_notification",
    ],
    "admin": [
        "view_user",
        "add_user",
        "change_user",
        "delete_user",
    ],
}


@receiver(post_save, sender=User)
def add_default_group_on_user_create(sender, instance, created, **kwargs):
    if not created:
        return

    default_group = Group.objects.filter(name="student").first()
    if not default_group:
        logger.warning("Default group 'student' not found for user_id=%s", instance.id)
        return

    instance.groups.add(default_group)



@receiver(post_migrate)
def ensure_default_groups(sender, **kwargs):
    # limit to your app migrations
    if sender.name != "auth2":
        return
    Group.objects.get_or_create(name="student")


@receiver(post_save, sender=Group)
def assign_permissions_on_group_create(sender, instance, created, **kwargs):
    if not created:
        return

    codenames = GROUP_PERMISSION_MAP.get(instance.name, [])
    if not codenames:
        return

    permissions = Permission.objects.filter(codename__in=codenames)
    found_codenames = set(permissions.values_list("codename", flat=True))
    missing_codenames = sorted(set(codenames) - found_codenames)

    if missing_codenames:
        logger.warning(
            "Missing permissions for group '%s': %s",
            instance.name,
            ", ".join(missing_codenames),
        )

    instance.permissions.add(*permissions)

import logging

from core.models import Department, Institution, Student, Teacher
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group, Permission
from django.db.models.signals import post_migrate, post_save
from django.dispatch import receiver

logger = logging.getLogger(__name__)
User = get_user_model()

ROLE_GROUPS = ("student", "teacher", "institution")

GROUP_PERMISSION_MAP = {
    "student": [
        "view_user",
        "view_language",
        "view_location",
        "view_timezone",
        "view_group",
        "view_permission",
        "view_notification",
        "view_notificationtype",
    ],
    "teacher": [
        "view_user",
        "view_language",
        "view_location",
        "view_timezone",
        "view_group",
        "view_permission",
        "view_notification",
        "add_notification",
        "change_notification",
        "view_notificationtype",
    ],
    "institution": [
        "view_user",
        "add_user",
        "change_user",
        "view_language",
        "view_location",
        "view_timezone",
        "view_group",
        "view_permission",
        "view_notification",
        "add_notification",
        "change_notification",
        "view_notificationtype",
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

    role_name = getattr(instance, "role", None) or "student"
    role_name = str(role_name).lower()
    if role_name not in ROLE_GROUPS:
        role_name = "student"

    if instance.groups.exists():
        return

    default_group = Group.objects.filter(name=role_name).first()
    if not default_group:
        logger.warning("Default group '%s' not found for user_id=%s", role_name, instance.id)
        return

    instance.groups.add(default_group)


@receiver(post_save, sender=User)
def ensure_role_profile_exists(sender, instance, created, **kwargs):
    ensure_role_profile_for_user(instance)


def ensure_role_profile_for_user(user):
    instance = user
    role_name = str(getattr(instance, "role", "") or "").lower().strip()
    if not role_name:
        first_group = instance.groups.first()
        role_name = str(first_group.name).lower().strip() if first_group else "student"
    if role_name not in {"student", "teacher", "institution", "admin"}:
        role_name = "student"

    if role_name == "teacher":
        if Teacher.objects.filter(user=instance).exists():
            return
        department, _ = Department.objects.get_or_create(name="General")
        Teacher.objects.create(user=instance, department=department)
        return
    
    if role_name == "institution":
        if Institution.objects.filter(user=instance).exists():
            return
        Institution.objects.create(
            user=instance,
            name=instance.username or instance.email or f"institution-{instance.id}",
        )
        return
    
    if role_name == "admin":
        return

    department, _ = Department.objects.get_or_create(name="General")
    if Student.objects.filter(user=instance).exists():
        return
    Student.objects.create(user=instance, department=department)


@receiver(post_migrate)
def ensure_default_groups(sender, **kwargs):
    # limit to your app migrations
    if sender.name != "auth2":
        return
    for group_name in ROLE_GROUPS:
        group, _ = Group.objects.get_or_create(name=group_name)
        codenames = GROUP_PERMISSION_MAP.get(group_name, [])
        permissions = Permission.objects.filter(codename__in=codenames)
        group.permissions.add(*permissions)


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

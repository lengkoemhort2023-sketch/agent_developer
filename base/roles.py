from __future__ import annotations

import logging

from django.conf import settings
from django.contrib.auth.models import Group, Permission


logger = logging.getLogger(__name__)


def ensure_default_groups() -> None:
    """Create the default Django groups and attach their permissions."""

    admin_group_name = getattr(settings, "AUTH_LDAP_DJANGO_ADMIN_GROUP", "Admin")
    user_group_name = getattr(settings, "AUTH_LDAP_DJANGO_USER_GROUP", "User")

    role_names = [
        "SuperAdmin",
        admin_group_name,
        "UserDownload",
        user_group_name,
    ]

    roles = []
    for role_name in role_names:
        if role_name not in roles:
            roles.append(role_name)

    for role_name in roles:
        group, created = Group.objects.get_or_create(name=role_name)
        if created:
            logger.info("Created default Django group '%s'", role_name)

    all_permissions = Permission.objects.all()
    Group.objects.get(name="SuperAdmin").permissions.set(all_permissions)
    Group.objects.get(name=admin_group_name).permissions.set(all_permissions)

    user_download_permissions = Permission.objects.filter(
        codename__in=[
            "can_view_document",
            "can_search_document",
            "can_download_document",
        ]
    )
    Group.objects.get(name="UserDownload").permissions.set(user_download_permissions)

    user_permissions = Permission.objects.filter(
        codename__in=[
            "can_search_document",
            "can_view_document",
        ]
    )
    Group.objects.get(name=user_group_name).permissions.set(user_permissions)

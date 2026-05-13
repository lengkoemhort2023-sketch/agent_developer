from __future__ import annotations

import logging
from django.contrib.auth.models import Group
from django.utils import timezone

from base.roles import ensure_default_groups
from user.models import LDAPGroupRoleMapping


logger = logging.getLogger(__name__)

MANAGED_ROLE_NAMES = {
    LDAPGroupRoleMapping.ROLE_SUPERADMIN,
    LDAPGroupRoleMapping.ROLE_ADMIN,
    LDAPGroupRoleMapping.ROLE_USER,
    LDAPGroupRoleMapping.ROLE_USERDOWNLOAD,
}


def extract_ldap_group_names(user) -> list[str]:
    """Extract LDAP group names from the authenticated user object."""

    ldap_group_names: list[str] = []
    if hasattr(user, "ldap_user") and user.ldap_user:
        try:
            if hasattr(user.ldap_user, "group_names"):
                ldap_group_names = list(user.ldap_user.group_names)
            elif hasattr(user.ldap_user, "group_dns"):
                for dn in user.ldap_user.group_dns:
                    cn_part = dn.split(",")[0] if "," in dn else dn
                    if cn_part.startswith("CN="):
                        ldap_group_names.append(cn_part[3:])
        except Exception as exc:
            logger.warning("Could not extract LDAP groups from ldap_user attribute: %s", exc)

    # Fallback only when LDAP metadata was not available. Avoid managed Django role
    # groups contaminating the LDAP snapshot.
    if not ldap_group_names:
        current_groups = list(user.groups.values_list("name", flat=True))
        ldap_group_names = [group for group in current_groups if group not in MANAGED_ROLE_NAMES]

    return sorted({group_name for group_name in ldap_group_names if group_name})


def upsert_observed_ldap_groups(group_names: list[str]) -> None:
    """Store observed LDAP groups so admins can map them to application roles."""

    now = timezone.now()
    for group_name in group_names:
        mapping = LDAPGroupRoleMapping.objects.filter(ldap_group_name__iexact=group_name).first()
        if mapping is None:
            mapping = LDAPGroupRoleMapping.objects.create(
                ldap_group_name=group_name,
                last_seen_at=now,
            )
            logger.info("Discovered new LDAP group '%s'", group_name)
            continue

        if mapping.last_seen_at != now:
            mapping.last_seen_at = now
            mapping.save(update_fields=["last_seen_at", "updated_at"])


def resolve_mapped_roles(group_names: list[str]) -> list[str]:
    """Resolve application roles from active LDAP group mappings."""

    mapping_by_name = {
        mapping.ldap_group_name.lower(): mapping
        for mapping in LDAPGroupRoleMapping.objects.filter(is_active=True)
    }

    roles: list[str] = []
    for group_name in group_names:
        mapping = mapping_by_name.get(group_name.lower())
        if mapping and mapping.assigned_role and mapping.assigned_role not in roles:
            roles.append(mapping.assigned_role)

    return roles


def fallback_roles_from_settings(group_names: list[str]) -> list[str]:
    """Default unmapped LDAP users to the standard User role."""

    return [LDAPGroupRoleMapping.ROLE_USER]


def apply_managed_roles(user, roles: list[str]) -> None:
    """Replace the user's managed application roles with the resolved set."""

    ensure_default_groups()

    managed_groups = {
        role_name: Group.objects.get_or_create(name=role_name)[0]
        for role_name in MANAGED_ROLE_NAMES
    }

    for group in managed_groups.values():
        user.groups.remove(group)

    for role_name in roles:
        group = managed_groups.get(role_name)
        if group:
            user.groups.add(group)
            logger.info("Assigned Django group '%s' to LDAP user %s", role_name, user.username)

    user.is_superuser = LDAPGroupRoleMapping.ROLE_SUPERADMIN in roles
    user.is_staff = user.is_superuser or LDAPGroupRoleMapping.ROLE_ADMIN in roles


def sync_ldap_user_groups(user) -> list[str]:
    """Map LDAP group membership to Django roles and store a snapshot."""

    all_ldap_groups = extract_ldap_group_names(user)
    if all_ldap_groups:
        logger.info("LDAP user %s groups: %s", user.username, all_ldap_groups)

    upsert_observed_ldap_groups(all_ldap_groups)

    resolved_roles = resolve_mapped_roles(all_ldap_groups)
    if not resolved_roles:
        resolved_roles = fallback_roles_from_settings(all_ldap_groups)
        logger.info("Using fallback LDAP role resolution for user %s: %s", user.username, resolved_roles)

    apply_managed_roles(user, resolved_roles)

    user.last_ldap_groups = all_ldap_groups
    user.last_ldap_login_at = timezone.now()
    return all_ldap_groups

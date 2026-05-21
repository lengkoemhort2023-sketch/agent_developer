from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver
from django.contrib.auth.models import Group
from django.conf import settings
import logging

from user.group_sync import sync_ldap_user_groups

logger = logging.getLogger(__name__)

@receiver(user_logged_in)
def update_ldap_flag_and_assign_groups(sender, user, request, **kwargs):
    """
    Signal handler that:
    1. Updates the is_ldap_user flag based on authentication method
    2. Automatically assigns Django permission groups based on LDAP group membership
       - If LDAP user is in "admin" group → assign Django "Admin" group
       - If LDAP user is in any other group → assign Django "User" group
    """
    # Update LDAP flag
    if hasattr(user, 'ldap_user') or getattr(user, 'is_ldap_user', False):
        user.is_ldap_user = True
    else:
        user.is_ldap_user = False
    
    # Auto-assign Django groups based on LDAP group membership
    if user.is_ldap_user:
        try:
            sync_ldap_user_groups(user)
        except Exception as e:
            logger.error(f"Error assigning groups to LDAP user {user.username}: {str(e)}", exc_info=True)
            # Fallback: assign user group on error
            try:
                user_group, _ = Group.objects.get_or_create(name=getattr(settings, 'AUTH_LDAP_DJANGO_USER_GROUP', 'User'))
                user.groups.add(user_group)
                user.is_staff = False
                user.is_superuser = False
                logger.info(f"Fallback: Assigned default '{user_group.name}' group to user {user.username}")
            except Exception as fallback_error:
                logger.error(f"Fallback group assignment also failed: {str(fallback_error)}")
    
    user.save()








# LDAP Group to Django Permission Group Mapping

## Overview
This document explains how LDAP groups are automatically mapped to Django permission groups when users log in via LDAP authentication.

## Configuration

### Settings Configuration (`app/core/settings.py`)

The following settings control the LDAP group mapping:

```python
# LDAP admin group name (case-insensitive matching)
AUTH_LDAP_ADMIN_GROUP_NAME = os.environ.get("AUTH_LDAP_ADMIN_GROUP_NAME", "admin")

# Django groups to assign (must exist in Django)
AUTH_LDAP_DJANGO_ADMIN_GROUP = os.environ.get("AUTH_LDAP_DJANGO_ADMIN_GROUP", "Admin")
AUTH_LDAP_DJANGO_USER_GROUP = os.environ.get("AUTH_LDAP_DJANGO_USER_GROUP", "User")
```

### Environment Variables

You can configure these via environment variables in your `.env` file:

```bash
# LDAP admin group name (the name of the LDAP group that should map to Django Admin)
AUTH_LDAP_ADMIN_GROUP_NAME=admin

# Django permission group names (must match groups created in Django)
AUTH_LDAP_DJANGO_ADMIN_GROUP=Admin
AUTH_LDAP_DJANGO_USER_GROUP=User
```

## How It Works

### Automatic Group Assignment

When a user logs in via LDAP:

1. **LDAP Authentication**: User authenticates against LDAP/Active Directory
2. **Group Detection**: The system checks which LDAP groups the user belongs to
3. **Group Mapping**:
   - If user is in LDAP group **"admin"** → Assigns Django **"Admin"** permission group
   - If user is in any other LDAP group → Assigns Django **"User"** permission group
4. **Cleanup**: Removes any mirrored LDAP groups (since `AUTH_LDAP_MIRROR_GROUPS = True`)

### Implementation Details

The mapping is handled by the `update_ldap_flag_and_assign_groups` signal handler in `user/signals.py`, which:

- Runs automatically on every user login (`user_logged_in` signal)
- Checks LDAP group membership (case-insensitive)
- Assigns appropriate Django permission groups
- Logs all group assignments for debugging

## Required Django Groups

Make sure these Django permission groups exist before using LDAP authentication:

- **Admin**: For users in LDAP "admin" group
- **User**: For users in other LDAP groups

### Creating Groups

Run the management command to create all required groups:

```bash
python manage.py create_roles_permissions
```

This command creates:
- SuperAdmin
- Admin
- UserDownload
- User

## Testing

### Test LDAP Admin User
1. Ensure a user exists in LDAP with group membership "admin"
2. Log in with that user's credentials
3. Verify the user is assigned Django "Admin" group
4. Check logs for: `Assigned Django group 'Admin' to LDAP user <username>`

### Test LDAP Regular User
1. Ensure a user exists in LDAP with group membership other than "admin"
2. Log in with that user's credentials
3. Verify the user is assigned Django "User" group
4. Check logs for: `Assigned Django group 'User' to LDAP user <username>`

## Troubleshooting

### User Not Getting Assigned Groups

1. **Check LDAP Group Names**: Verify the LDAP group name matches `AUTH_LDAP_ADMIN_GROUP_NAME` (case-insensitive)
2. **Check Django Groups Exist**: Run `python manage.py create_roles_permissions` to ensure groups exist
3. **Check Logs**: Look for error messages in application logs
4. **Verify LDAP Connection**: Ensure `AUTH_LDAP_MIRROR_GROUPS = True` is set in settings

### Logs Location

Check application logs for group assignment messages:
- Success: `Assigned Django group 'Admin' to LDAP user <username>`
- Errors: `Error assigning groups to LDAP user <username>: <error>`

## Customization

### Changing LDAP Admin Group Name

If your LDAP admin group has a different name (e.g., "administrators"), update:

```python
# In settings.py or .env
AUTH_LDAP_ADMIN_GROUP_NAME = "administrators"
```

### Changing Django Group Names

If you want to use different Django group names:

```python
# In settings.py or .env
AUTH_LDAP_DJANGO_ADMIN_GROUP = "Administrators"
AUTH_LDAP_DJANGO_USER_GROUP = "RegularUsers"
```

Make sure these groups exist in Django before changing the configuration.

## Notes

- Group assignment happens **on every login** to ensure groups stay synchronized
- The system removes mirrored LDAP groups to keep only permission groups
- Case-insensitive matching is used for LDAP group names
- If no LDAP groups are found, the user is assigned the default "User" group

from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager, PermissionsMixin
import uuid
from django.core.validators import RegexValidator
username_validator = RegexValidator(
    regex=r'^[a-zA-Z0-9_.]+$',
    message="Username may only contain letters, numbers, underscores, and dots."
)
class CustomUserManager(BaseUserManager):
    def create_user(self, username, email, password=None, **extra_fields):
        if not username:
            raise ValueError('The Username field must be set')
        if not email:
            raise ValueError('The Email field must be set')
        user = self.model(username=username, email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self.create_user(username, email, password, **extra_fields)

class User(AbstractUser, PermissionsMixin):

    LANGUAGE_CHOICES = [
        ('en', 'English'),
        ('km', 'ខ្មែរ'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    
    username = models.CharField(max_length=128, unique=True, validators=[username_validator])
    email = models.EmailField(max_length=255, unique=True)
    first_name = models.CharField(max_length=25)
    last_name = models.CharField(max_length=25)
    employee_id = models.CharField(max_length=50)
    language = models.CharField(max_length=2, choices=LANGUAGE_CHOICES, default='en')
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    objects = CustomUserManager()
    is_ldap_user = models.BooleanField(default=False)
    last_ldap_groups = models.JSONField(default=list, blank=True)
    last_ldap_login_at = models.DateTimeField(blank=True, null=True)
    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['first_name', 'last_name', 'email', 'employee_id']
    class Meta:
        swappable = "AUTH_USER_MODEL"
        db_table = "users"
        verbose_name = "User"
        verbose_name_plural = "Users"
        app_label = 'user'

    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.username})"


class LDAPGroupRoleMapping(models.Model):
    ROLE_SUPERADMIN = "SuperAdmin"
    ROLE_ADMIN = "Admin"
    ROLE_USER = "User"
    ROLE_USERDOWNLOAD = "UserDownload"
    ROLE_NONE = ""

    ROLE_CHOICES = [
        (ROLE_NONE, "No automatic role"),
        (ROLE_SUPERADMIN, "SuperAdmin"),
        (ROLE_ADMIN, "Admin"),
        (ROLE_USER, "User"),
        (ROLE_USERDOWNLOAD, "UserDownload"),
    ]

    ldap_group_name = models.CharField(max_length=255, unique=True, db_index=True)
    assigned_role = models.CharField(max_length=32, choices=ROLE_CHOICES, blank=True, default=ROLE_NONE)
    is_active = models.BooleanField(default=True)
    last_seen_at = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "ldap_group_role_mappings"
        verbose_name = "LDAP Group Mapping"
        verbose_name_plural = "LDAP Group Mappings"
        ordering = ["ldap_group_name"]
        app_label = "user"

    def __str__(self):
        return self.ldap_group_name

from rest_framework import serializers
from .models import User

class UserSerializer(serializers.ModelSerializer):
    """Serializer for User model"""
    permissions = serializers.SerializerMethodField()
    groups = serializers.SerializerMethodField() 
    class Meta:
        model = User
        fields = ['id', 'first_name', 'last_name', 'username', 'email', 'is_superuser', 'is_ldap_user', 'permissions', 'groups', 'last_ldap_groups', 'last_ldap_login_at', 'language', 'avatar', 'employee_id']
    def get_permissions(self, obj):

        return list(obj.get_all_permissions())
    
    def get_groups(self, obj):

        return list(obj.groups.values_list('name', flat=True))


class UserProfileUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating user profile"""
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email', 'language', 'avatar']
        extra_kwargs = {
            'email': {'required': False},
        }

    def validate_email(self, value):
        user = self.instance
        if User.objects.filter(email=value).exclude(id=user.id).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return value


class LoginSerializer(serializers.Serializer):
    username_or_email = serializers.CharField(
        max_length=150,
        required=False,
        allow_blank=True,
        help_text="Username or email for authentication"
    )
    email = serializers.EmailField(
        max_length=254,
        required=False,
        allow_blank=True,
        help_text="Email for authentication"
    )
    password = serializers.CharField(
        max_length=128,
        write_only=True,
        help_text="Password for authentication"
    )

    def validate(self, attrs):
        username_or_email = attrs.get('username_or_email', '').strip()
        password = attrs.get('password')

        if not password:
            raise serializers.ValidationError("Password is required.")

        if not username_or_email:
            raise serializers.ValidationError("Either username or email must be provided.")

        return attrs









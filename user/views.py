from base.management.commands.load_dummy_data import User
from .serializers import UserSerializer, UserProfileUpdateSerializer
from base.utils.api_response import *
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework import status
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import LoginSerializer
from base.utils import error_response, success_response
from rest_framework_simplejwt.views import TokenRefreshView
from rest_framework_simplejwt.exceptions import TokenError
from chat.models import ChatSession
from user.group_sync import sync_ldap_user_groups
import ldap
from datetime import datetime, timezone

class LdapOrBackendLoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if not serializer.is_valid():
            return error_response(serializer.errors, status_code=400)

        username_or_email = serializer.validated_data.get("username_or_email")
        password = serializer.validated_data.get("password")
        
        # Try Django user first (not LDAP)
        user_backend = User.objects.filter(username=username_or_email, is_ldap_user=False).first()
        if not user_backend and '@' in username_or_email:
            user_backend = User.objects.filter(email=username_or_email, is_ldap_user=False).first()

        user = None
        ldap_authenticated = False
        if user_backend:
            user = authenticate(username=user_backend.username, password=password)

        # If not found, fallback to LDAP
        if user is None:
            user = authenticate(username=username_or_email, password=password)
            if user is not None:
                ldap_authenticated = True
        if user is None:
            user = authenticate(email=username_or_email, password=password)
            if user is not None:
                ldap_authenticated = True

        if user is not None:
            # Always update is_ldap_user flag based on how the user was authenticated
            user.is_ldap_user = ldap_authenticated
            if ldap_authenticated:
                sync_ldap_user_groups(user)
            # If user is SuperAdmin and authenticated via LDAP, save password hash to Django DB
            superadmin_group_name = "superadmin"
            user_groups = [g.lower() for g in user.groups.values_list("name", flat=True)]
            if superadmin_group_name.lower() in user_groups:
                if not user.is_superuser:
                    user.is_superuser = True
            user.save()

            # Check if user has any chat sessions, create one if not
            session = ChatSession.get_last_empty(user=user)
            if not session:
                session = ChatSession.objects.create(user=user)

            refresh = RefreshToken.for_user(user)
            access_token = refresh.access_token
            expires = datetime.fromtimestamp(access_token["exp"], tz=timezone.utc).isoformat()
            user_info = UserSerializer(user).data

            if hasattr(user, "ldap_user") and "employeeID" in user.ldap_user.attrs:
                user_info["employee_id"] = user.ldap_user.attrs["employeeID"][0]

            return success_response(
                message="Login successful",
                data={
                    "access_token": str(access_token),
                    "refresh_token": str(refresh),
                    "expires": expires,
                    "user": user_info,
                    "current_session_id": str(session.id),
                },
            )
        return error_response(message="Your password is incorrect or this account doesn't exist. Please check and try again", status_code=status.HTTP_401_UNAUTHORIZED)

class TokenRefreshView(TokenRefreshView):
    permission_classes = [AllowAny]
    def post(self, request, *args, **kwargs):
        try:
            refresh_token = request.data.get("refresh_token")
            serializer = self.get_serializer(data={"refresh": refresh_token})
            serializer.is_valid(raise_exception=True)
            data = {
                "access_token": serializer.validated_data.get("access"),
                "refresh_token": serializer.validated_data.get("refresh"),
            }
            return success_response(
                message="Refresh token",
                data=data, status_code=status.HTTP_200_OK
            )
        except TokenError:
            return error_response(
                message="Token is blacklisted or invalid",
                status_code=status.HTTP_401_UNAUTHORIZED
            )
class WhoAmIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        data = UserSerializer(user).data
        return success_response(
            message="Current user retrieved successfully",
            data=data
        )

@api_view(['POST'])
def logout(request):
    refresh_token = request.data.get("refresh_token")
    if refresh_token:
        try:
            token = RefreshToken(refresh_token)
            token.blacklist()
            return success_response(message="User logged out successfully")
        except Exception:
            return error_response(message="Invalid token", status_code=status.HTTP_400_BAD_REQUEST)


class ProfileView(APIView):
    """Get or update user profile"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Get current user profile"""
        user = request.user
        serializer = UserSerializer(user)
        return success_response(
            message="Profile retrieved successfully",
            data=serializer.data
        )

    def patch(self, request):
        """Update user profile"""
        user = request.user
        serializer = UserProfileUpdateSerializer(user, data=request.data, partial=True)
        
        if serializer.is_valid():
            serializer.save()
            # Return updated user data
            user_serializer = UserSerializer(user)
            return success_response(
                message="Profile updated successfully",
                data=user_serializer.data
            )
        return error_response(
            message="Validation error",
            status_code=status.HTTP_400_BAD_REQUEST
        )


class UpdateAvatarView(APIView):
    """Update user avatar"""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        """Upload new avatar"""
        user = request.user
        
        if 'avatar' not in request.FILES:
            return error_response(
                message="No avatar file provided",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        avatar = request.FILES['avatar']
        
        # Validate file type
        allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
        if avatar.content_type not in allowed_types:
            return error_response(
                message="Invalid file type. Allowed: JPEG, PNG, GIF, WebP",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate file size (max 2MB)
        if avatar.size > 2 * 1024 * 1024:
            return error_response(
                message="File too large. Maximum size is 2MB",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        user.avatar = avatar
        user.save()
        
        user_serializer = UserSerializer(user)
        return success_response(
            message="Avatar updated successfully",
            data=user_serializer.data
        )







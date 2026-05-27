from .views import (
    download_file,
    ldap_health_check,
    qdrant_health_check,
    get_signed_download_url,
    protected_media,
    docx_extracted_media,
    metrics_view,
)
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
import user.views as user_views
from django.conf.urls.static import static
from django.conf import settings
from django.http import HttpResponse
from chat.views import submit_feedback

BASE_PREFIX = 'api/'

urlpatterns = [
    path('admin', RedirectView.as_view(pattern_name='admin:index', permanent=True)),
    path('admin/', admin.site.urls),

    # REGISTER URL HERE
    path(BASE_PREFIX, include([
        path('departments/', include('department.urls')),
        path('document-types/', include('docs_type.urls')),
        path('documents/', include('document.urls')),
        path('chats/', include('chat.urls')),

        # USER AUTHENTICATION
        path('whoami/', user_views.WhoAmIView.as_view(), name='whoami'),
        path('refresh/', user_views.TokenRefreshView.as_view(), name='token_refresh'),
        path('login/', user_views.LdapOrBackendLoginView.as_view(), name='login'),
        path('logout/', user_views.logout, name='logout'),

        # USER PROFILE
        path('profile/', user_views.ProfileView.as_view(), name='profile'),
        path('profile/avatar/', user_views.UpdateAvatarView.as_view(), name='update_avatar'),

        # HEALTHCHECK
        path('ldap/health/', ldap_health_check, name='ldap_health_check'),
        path('qdrant/health/', qdrant_health_check, name='qdrant_health_check'),
        path('get-signed-download-url/', get_signed_download_url, name='get_signed_download_url'),
        path('metrics/', metrics_view, name='metrics'),
    ])),
    # Add download URL pattern
    path('download/', download_file, name='download_file'),
    path('feedback/', submit_feedback, name='submit_feedback_root'),
    path('media/protected/<path:path>', protected_media, name='protected_media'),
    path('media/<path:path>', protected_media, name='media_file'),
    path('docx_extracted_images/<path:path>', docx_extracted_media, name='docx_extracted_image_file'),
    # Add root URL pattern to handle requests to /
    path('', lambda request: HttpResponse("AMK Agent API is running", content_type="text/plain"), name='root'),
    
    # Health check endpoint for Docker
    path('health/', lambda request: HttpResponse("OK", content_type="text/plain"), name='health_check'),

    # Prometheus metrics endpoint — scraped by Prometheus at /metrics
    path('', include('django_prometheus.urls')),
]

# Serve media files in development with CORS headers
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT, view=protected_media)
    urlpatterns += static('/favicon.ico', document_root=str(settings.STATIC_ROOT) + '/favicon.ico')
admin.site.site_title = "AMK Agent site admin (DEV)"
admin.site.site_header = "AMK Agent administration"
admin.site.index_title = "Site administration"

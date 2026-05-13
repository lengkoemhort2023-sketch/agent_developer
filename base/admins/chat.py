from django.contrib import admin
from chat.models import ChatSession

try:
    admin.site.register(ChatSession)
except admin.sites.AlreadyRegistered:
    pass








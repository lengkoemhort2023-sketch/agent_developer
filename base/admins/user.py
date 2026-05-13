from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from user.models import LDAPGroupRoleMapping, User

class UserAdmin(BaseUserAdmin):
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'first_name', 'last_name', 'password1', 'password2', 'employee_id', 'is_ldap_user'),
        }),
    )
    list_display = BaseUserAdmin.list_display + ('is_ldap_user', 'last_ldap_login_at', 'last_ldap_groups_preview')
    fieldsets = BaseUserAdmin.fieldsets + (
        ('LDAP', {'fields': ('is_ldap_user', 'last_ldap_login_at', 'last_ldap_groups_display')}),
    )
    readonly_fields = ('is_ldap_user', 'last_ldap_login_at', 'last_ldap_groups_display')

    @admin.display(description='Last LDAP groups')
    def last_ldap_groups_preview(self, obj):
        groups = obj.last_ldap_groups or []
        preview = ", ".join(groups)
        return preview[:60] + "..." if len(preview) > 60 else preview

    @admin.display(description='Last LDAP groups')
    def last_ldap_groups_display(self, obj):
        groups = obj.last_ldap_groups or []
        return ", ".join(groups) if groups else "-"

    def get_fieldsets(self, request, obj=None):
        fieldsets = super().get_fieldsets(request, obj)
        if obj and getattr(obj, 'is_ldap_user', False):
            new_fieldsets = []
            for name, opts in fieldsets:
                fields = opts.get('fields', ())
                if 'password' in fields:
                    fields = tuple(f for f in fields if f != 'password')
                new_fieldsets.append((name, {**opts, 'fields': fields}))
            return new_fieldsets
        return fieldsets

    def get_form(self, request, obj=None, **kwargs):
        form = super().get_form(request, obj, **kwargs)
        if obj and getattr(obj, 'is_ldap_user', False):
            if 'password' in form.base_fields:
                form.base_fields['password'].disabled = True
        return form

admin.site.register(User, UserAdmin)


@admin.register(LDAPGroupRoleMapping)
class LDAPGroupRoleMappingAdmin(admin.ModelAdmin):
    list_display = ("ldap_group_name", "assigned_role", "is_active", "last_seen_at", "updated_at")
    list_filter = ("assigned_role", "is_active")
    search_fields = ("ldap_group_name",)
    ordering = ("ldap_group_name",)
    readonly_fields = ("last_seen_at", "created_at", "updated_at")
    fieldsets = (
        (None, {"fields": ("ldap_group_name", "assigned_role", "is_active")}),
        ("Tracking", {"fields": ("last_seen_at", "created_at", "updated_at")}),
    )






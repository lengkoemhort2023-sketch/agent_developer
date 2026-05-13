from django.core.management.base import BaseCommand
from user.models import LDAPGroupRoleMapping


class Command(BaseCommand):
    help = 'Setup default LDAP group to role mappings for chatbot groups'

    def handle(self, *args, **options):
        created_count = 0
        
        # Define the default mappings
        mappings = [
            {
                'ldap_group_name': 'chatbot_admin',
                'assigned_role': LDAPGroupRoleMapping.ROLE_ADMIN,
                'description': 'Administrator group'
            },
            {
                'ldap_group_name': 'chatbot_userdownload',
                'assigned_role': LDAPGroupRoleMapping.ROLE_USERDOWNLOAD,
                'description': 'User download access group'
            },
        ]
        
        for mapping_config in mappings:
            ldap_group_name = mapping_config['ldap_group_name']
            assigned_role = mapping_config['assigned_role']
            description = mapping_config['description']
            
            mapping, created = LDAPGroupRoleMapping.objects.get_or_create(
                ldap_group_name=ldap_group_name,
                defaults={
                    'assigned_role': assigned_role,
                    'is_active': True,
                }
            )
            
            if created:
                self.stdout.write(
                    self.style.SUCCESS(
                        f'✓ Created LDAP mapping: {ldap_group_name} → {assigned_role} ({description})'
                    )
                )
                created_count += 1
            else:
                # Update existing mapping if needed
                if mapping.assigned_role != assigned_role or not mapping.is_active:
                    mapping.assigned_role = assigned_role
                    mapping.is_active = True
                    mapping.save()
                    self.stdout.write(
                        self.style.WARNING(
                            f'⟳ Updated existing mapping: {ldap_group_name} → {assigned_role} ({description})'
                        )
                    )
                else:
                    self.stdout.write(
                        self.style.SUCCESS(
                            f'✓ LDAP mapping already exists: {ldap_group_name} → {assigned_role} ({description})'
                        )
                    )
        
        if created_count > 0:
            self.stdout.write(
                self.style.SUCCESS(f'\n✓ Successfully created {created_count} LDAP group mapping(s).')
            )
        
        self.stdout.write(
            self.style.SUCCESS(
                '\n✓ LDAP group mappings setup complete.\n'
                'Users in chatbot_admin will get Admin role\n'
                'Users in chatbot_userdownload will get UserDownload role\n'
                'All other users will get User role by default'
            )
        )

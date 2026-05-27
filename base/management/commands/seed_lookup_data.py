from django.core.management.base import BaseCommand
from department.models import Department
from docs_type.models import DocumentType


DEPARTMENTS = [
    "12. Core Banking System",
    "13. Contact Center",
    "14. Credit Business",
    "16. Executive",
    "17. Finance",
    "18. Human Resources",
    "19. Internal Audit",
    "20. IT Infrastructure & Operation",
    "21. Marketing and Communication",
    "23. Product Development",
    "24. Admin and Procurement",
    "25. Risk Management",
    "26. Research",
    "27. Training and Development",
    "28. Treasury",
    "29. IT Security",
    "30. Bancassurance",
    "31. Management Information System",
    "34. Legal and Compliance",
    "35. Deposit and Service",
    "36. Business",
    "37. IT Project Management",
    "38. Software Research and Development",
    "39. Credit Underwriting",
    "40. Operations",
    "41. Digital Banking & Card Payment",
    "44. Business Development",
    "45. Supply Chain Financing Business",
    "46. Credit Control",
    "48. Research and Development",
    "49. Agent and Digital Banking",
    "50. Business Intelligent",
    "51. Legal",
    "52. Compliance",
]

DOCUMENT_TYPES = [
    "Addendum to Policy",
    "Policy",
    "Memo",
    "Interim Memo",
    "Guidelines",
    "Notification",
    "Amendment to SOP",
    "Standard Operating Procedure",
    "User Guide",
    "Announcement",
    "Framework",
    "Amendment to Policy",
    "Addendum to SOP",
]


class Command(BaseCommand):
    help = "Seed departments and document types"

    def handle(self, *args, **options):
        created_depts = 0
        for name in DEPARTMENTS:
            _, created = Department.objects.get_or_create(name=name)
            if created:
                created_depts += 1

        created_types = 0
        for name in DOCUMENT_TYPES:
            _, created = DocumentType.objects.get_or_create(presentation=name)
            if created:
                created_types += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Created {created_depts} departments and {created_types} document types"
            )
        )

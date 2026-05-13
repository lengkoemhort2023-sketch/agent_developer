import os
import tempfile
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group, Permission
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, override_settings
from rest_framework import status
from rest_framework.test import APITestCase
from reportlab.pdfgen import canvas

from base.application.document_indexing import create_document_indexing_job
from base.roles import ensure_default_groups
from department.models import Department
from docs_type.models import DocumentType
from document.models import Document, DocumentIndexingJob
from document.serializers import DocumentListSerializer, DocumentUpdateSerializer
from document.tasks import index_document_indexing_job, process_unprocessed_documents
from document.utils import ensure_pdf_version_exists


class DocumentPdfCacheTests(SimpleTestCase):
    def test_existing_cached_pdf_is_reused_without_reconversion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = os.path.join(temp_dir, "documents")
            os.makedirs(documents_dir, exist_ok=True)

            source_path = os.path.join(documents_dir, "policy.docx")
            cached_pdf_path = os.path.splitext(source_path)[0] + "_view.pdf"

            with open(source_path, "wb") as source_file:
                source_file.write(b"docx placeholder")

            with open(cached_pdf_path, "wb") as cached_pdf:
                cached_pdf.write(b"%PDF-1.4\ncached preview\n")

            document = SimpleNamespace(id="doc-1", full_file_path=source_path)

            with patch("document.utils.convert_to_pdf") as mock_convert:
                self.assertTrue(ensure_pdf_version_exists(document))
                mock_convert.assert_not_called()

    def test_force_refresh_rebuilds_existing_cached_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = os.path.join(temp_dir, "documents")
            os.makedirs(documents_dir, exist_ok=True)

            source_path = os.path.join(documents_dir, "policy.docx")
            cached_pdf_path = os.path.splitext(source_path)[0] + "_view.pdf"

            with open(source_path, "wb") as source_file:
                source_file.write(b"docx placeholder")

            with open(cached_pdf_path, "wb") as cached_pdf:
                cached_pdf.write(b"%PDF-1.4\nstale cached preview\n")

            document = SimpleNamespace(id="doc-2", full_file_path=source_path)

            def _write_refreshed_pdf(_input_path, output_path):
                with open(output_path, "wb") as refreshed_pdf:
                    refreshed_pdf.write(b"%PDF-1.4\nrefreshed preview\n")

            with patch("document.utils.convert_to_pdf", side_effect=_write_refreshed_pdf) as mock_convert:
                self.assertTrue(ensure_pdf_version_exists(document, force_refresh=True))
                mock_convert.assert_called_once_with(source_path, cached_pdf_path)
                self.assertTrue(os.path.exists(cached_pdf_path))


class UploadNewVersionTests(APITestCase):
    def setUp(self):
        self.temp_media_dir = tempfile.TemporaryDirectory()
        media_override = override_settings(MEDIA_ROOT=self.temp_media_dir.name)
        media_override.enable()
        self.addCleanup(media_override.disable)
        self.addCleanup(self.temp_media_dir.cleanup)

        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="version_uploader",
            email="version_uploader@example.com",
            password="testpass123",
            first_name="Version",
            last_name="Uploader",
            employee_id="EMP-001",
        )
        upload_permission = Permission.objects.get(codename="can_upload_document")
        self.user.user_permissions.add(upload_permission)
        self.client.force_authenticate(self.user)

        self.department = Department.objects.create(name="Operations")
        self.doc_type = DocumentType.objects.create(
            name="policy",
            presentation="Policy",
        )
        self.document = Document.objects.create(
            title="Travel Policy",
            description="Original version",
            department=self.department,
            type=self.doc_type,
            version=Decimal("1.0"),
            total_versions=1,
            original_filename="travel-policy-v1.pdf",
            stored_filename="travel-policy-v1.pdf",
            file_path="documents/travel-policy-v1.pdf",
            file_size=18,
            file_format="pdf",
            publisher_id=str(self.user.id),
            publisher_name="Version Uploader",
            is_latest_version=True,
        )

    @patch(
        "document.views.enqueue_document_indexing",
        return_value=SimpleNamespace(job=SimpleNamespace(id="job-1")),
    )
    def test_upload_new_version_accepts_post_and_creates_version(self, _mock_enqueue):
        uploaded_file = SimpleUploadedFile(
            "travel-policy-v2.pdf",
            b"%PDF-1.4\nupdated policy\n",
            content_type="application/pdf",
        )
        response = self.client.post(
            f"/api/documents/{self.document.id}/upload-version/",
            {"file": uploaded_file, "title": "Travel Policy Updated"},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.document.refresh_from_db()

        self.assertFalse(self.document.is_latest_version)
        self.assertEqual(self.document.total_versions, 2)

        new_version = Document.objects.exclude(id=self.document.id).get()
        self.assertEqual(new_version.title, "Travel Policy Updated")
        self.assertEqual(new_version.version, Decimal("2"))
        self.assertTrue(new_version.is_latest_version)
        self.assertEqual(new_version.parent_document_id, self.document.id)
        self.assertEqual(new_version.file_format, "pdf")
        self.assertEqual(response.data["body"]["id"], str(new_version.id))

    @patch(
        "document.views.enqueue_document_indexing",
        return_value=SimpleNamespace(job=SimpleNamespace(id="job-2")),
    )
    def test_upload_new_version_persists_submitted_effective_and_expiry_dates(self, _mock_enqueue):
        uploaded_file = SimpleUploadedFile(
            "travel-policy-v3.pdf",
            b"%PDF-1.4\nupdated policy with new dates\n",
            content_type="application/pdf",
        )
        response = self.client.post(
            f"/api/documents/{self.document.id}/upload-version/",
            {
                "file": uploaded_file,
                "title": "Travel Policy Updated",
                "effective_date": "2026-04-01",
                "expiry_date": "2026-12-31",
            },
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        new_version = Document.objects.exclude(id=self.document.id).get()
        self.assertEqual(new_version.effective_date.isoformat(), "2026-04-01")
        self.assertEqual(new_version.expiry_date.isoformat(), "2026-12-31")

    @patch(
        "document.views.enqueue_document_indexing",
        return_value=SimpleNamespace(job=SimpleNamespace(id="job-3")),
    )
    def test_upload_document_rejects_expiry_date_before_effective_date(self, _mock_enqueue):
        payload = {
            "file_name": "invalid-dates.pdf",
            "file_data": "JVBERi0xLjQK",
            "title": "Invalid Dates",
            "department_id": str(self.department.id),
            "type_id": str(self.doc_type.id),
            "effective_date": "2026-12-31",
            "expiry_date": "2026-03-21",
        }

        response = self.client.post(
            "/api/documents/upload/",
            data=json.dumps(payload),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn(
            "Expiry date must be on or after effective date.",
            response.data["result_message"],
        )

    @patch("document.tasks.finalize_document_indexing_job.delay")
    @patch("document.tasks._run_vector_processing")
    @patch(
        "document.tasks.deactivate_document_from_rag",
        return_value={"success": True},
    )
    def test_indexing_new_version_deactivates_previous_family_vectors(
        self,
        mock_deactivate,
        mock_run_vector_processing,
        _mock_finalize,
    ):
        os.makedirs(os.path.join(self.temp_media_dir.name, "documents"), exist_ok=True)
        new_file_path = os.path.join("documents", "travel-policy-renamed.pdf")
        with open(os.path.join(self.temp_media_dir.name, new_file_path), "wb") as file_handle:
            file_handle.write(b"%PDF-1.4\nlatest version\n")

        self.document.is_latest_version = False
        self.document.is_vector_processed = True
        self.document.save(update_fields=["is_latest_version", "is_vector_processed"])

        new_version = Document.objects.create(
            title="Travel Policy",
            description="Updated version",
            department=self.department,
            type=self.doc_type,
            version=Decimal("2.0"),
            total_versions=2,
            parent_document=self.document,
            original_filename="travel-policy-renamed.pdf",
            stored_filename="travel-policy-renamed.pdf",
            file_path=new_file_path,
            file_size=22,
            file_format="pdf",
            publisher_id=str(self.user.id),
            publisher_name="Version Uploader",
            is_latest_version=True,
        )

        job = create_document_indexing_job(new_version)
        index_document_indexing_job(str(job.id))

        mock_deactivate.assert_called_once_with(
            file_id=str(self.document.id),
            file_type=self.doc_type.name,
        )
        mock_run_vector_processing.assert_called_once()

    def test_document_list_serializer_surfaces_vector_cleanup_stage(self):
        job = create_document_indexing_job(self.document)
        job.status = DocumentIndexingJob.STATUS_RUNNING
        job.stage = DocumentIndexingJob.STAGE_DEACTIVATING_PREVIOUS_VERSIONS
        job.save(update_fields=["status", "stage", "updated_at"])

        serialized = DocumentListSerializer(self.document).data

        self.assertEqual(serialized["status"], "Updating Index")
        self.assertEqual(
            serialized["indexing_stage"],
            DocumentIndexingJob.STAGE_DEACTIVATING_PREVIOUS_VERSIONS,
        )
        self.assertEqual(
            serialized["indexing_message"],
            "Deactivating previous version vectors before indexing the new version.",
        )


class UploadDocumentLongTitleTests(APITestCase):
    def setUp(self):
        self.temp_media_dir = tempfile.TemporaryDirectory()
        media_override = override_settings(MEDIA_ROOT=self.temp_media_dir.name)
        media_override.enable()
        self.addCleanup(media_override.disable)
        self.addCleanup(self.temp_media_dir.cleanup)

        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="long_title_uploader",
            email="long_title_uploader@example.com",
            password="testpass123",
            first_name="Long",
            last_name="Title",
            employee_id="EMP-002",
        )
        upload_permission = Permission.objects.get(codename="can_upload_document")
        self.user.user_permissions.add(upload_permission)
        self.client.force_authenticate(self.user)

        self.department = Department.objects.create(name="Compliance")
        self.doc_type = DocumentType.objects.create(
            name="guideline",
            presentation="Guideline",
        )

    @patch(
        "document.views.enqueue_document_indexing",
        return_value=SimpleNamespace(job=SimpleNamespace(id="job-long-title")),
    )
    def test_upload_document_accepts_title_longer_than_previous_limit(self, _mock_enqueue):
        long_title = "A" * 1500
        payload = {
            "file_name": "long-title.pdf",
            "file_data": "JVBERi0xLjQK",
            "title": long_title,
            "department_id": str(self.department.id),
            "type_id": str(self.doc_type.id),
            "effective_date": "2026-03-21",
            "expiry_date": "2026-12-31",
        }

        response = self.client.post(
            "/api/documents/upload/",
            data=json.dumps(payload),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        document = Document.objects.get(id=response.data["body"]["id"])
        self.assertEqual(document.title, long_title)
        self.assertEqual(document.effective_date.isoformat(), "2026-03-21")
        self.assertEqual(document.expiry_date.isoformat(), "2026-12-31")


class DocumentSuggestionEndpointTests(APITestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="document_suggestion_user",
            email="document_suggestion_user@example.com",
            password="testpass123",
            first_name="Document",
            last_name="Suggestion",
            employee_id="EMP-003",
        )
        view_permission = Permission.objects.get(codename="can_view_document")
        self.user.user_permissions.add(view_permission)
        self.client.force_authenticate(self.user)

        self.department = Department.objects.create(name="Risk")
        self.doc_type = DocumentType.objects.create(
            name="manual",
            presentation="Manual",
        )

    def test_document_suggestions_return_only_active_latest_documents(self):
        visible_document = Document.objects.create(
            title="AML Handbook",
            description="Visible latest document",
            department=self.department,
            type=self.doc_type,
            version=Decimal("2.0"),
            total_versions=2,
            original_filename="aml-handbook-v2.pdf",
            stored_filename="aml-handbook-v2.pdf",
            file_path="documents/aml-handbook-v2.pdf",
            file_size=42,
            file_format="pdf",
            publisher_id=str(self.user.id),
            publisher_name="Document Suggestion",
            is_latest_version=True,
            is_active=True,
        )

        Document.objects.create(
            title="AML Handbook Old",
            description="Old version should be excluded",
            department=self.department,
            type=self.doc_type,
            version=Decimal("1.0"),
            total_versions=2,
            original_filename="aml-handbook-v1.pdf",
            stored_filename="aml-handbook-v1.pdf",
            file_path="documents/aml-handbook-v1.pdf",
            file_size=41,
            file_format="pdf",
            publisher_id=str(self.user.id),
            publisher_name="Document Suggestion",
            is_latest_version=False,
            is_active=True,
        )

        Document.objects.create(
            title="Archived Policy",
            description="Inactive document should be excluded",
            department=self.department,
            type=self.doc_type,
            version=Decimal("1.0"),
            total_versions=1,
            original_filename="archived-policy.pdf",
            stored_filename="archived-policy.pdf",
            file_path="documents/archived-policy.pdf",
            file_size=40,
            file_format="pdf",
            publisher_id=str(self.user.id),
            publisher_name="Document Suggestion",
            is_latest_version=True,
            is_active=False,
        )

        response = self.client.get("/api/documents/suggestions/")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data["body"],
            [
                {
                    "id": str(visible_document.id),
                    "title": "AML Handbook",
                    "document_type": "manual",
                    "file_name": "aml-handbook-v2.pdf",
                }
            ],
        )


class DocumentUpdateDateValidationTests(APITestCase):
    def setUp(self):
        self.department = Department.objects.create(name="Quality")
        self.doc_type = DocumentType.objects.create(name="procedure", presentation="Procedure")
        self.document = Document.objects.create(
            title="Date Validation",
            description="Document",
            department=self.department,
            type=self.doc_type,
            version=Decimal("1.0"),
            total_versions=1,
            original_filename="date-validation.pdf",
            stored_filename="date-validation.pdf",
            file_path="documents/date-validation.pdf",
            file_size=10,
            file_format="pdf",
            effective_date="2026-03-21",
            expiry_date="2026-12-31",
        )

    def test_update_document_serializer_rejects_expiry_before_effective(self):
        serializer = DocumentUpdateSerializer(
            self.document,
            data={
                "effective_date": "2026-12-31",
                "expiry_date": "2026-03-21",
            },
            partial=True,
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("expiry_date", serializer.errors)


class DocumentUserRolePreviewTests(APITestCase):
    def setUp(self):
        self.temp_media_dir = tempfile.TemporaryDirectory()
        media_override = override_settings(MEDIA_ROOT=self.temp_media_dir.name)
        media_override.enable()
        self.addCleanup(media_override.disable)
        self.addCleanup(self.temp_media_dir.cleanup)
        ensure_default_groups()

        user_model = get_user_model()
        self.view_user = user_model.objects.create_user(
            username="view_only_user",
            email="view_only_user@example.com",
            password="testpass123",
            first_name="View",
            last_name="Only",
            employee_id="EMP-010",
        )
        self.download_user = user_model.objects.create_user(
            username="download_user",
            email="download_user@example.com",
            password="testpass123",
            first_name="Download",
            last_name="User",
            employee_id="EMP-011",
        )
        self.admin_user = user_model.objects.create_user(
            username="admin_user",
            email="admin_user@example.com",
            password="testpass123",
            first_name="Admin",
            last_name="User",
            employee_id="EMP-012",
        )
        self.superadmin_user = user_model.objects.create_user(
            username="superadmin_user",
            email="superadmin_user@example.com",
            password="testpass123",
            first_name="Super",
            last_name="Admin",
            employee_id="EMP-013",
        )

        view_permission = Permission.objects.get(codename="can_view_document")
        download_permission = Permission.objects.get(codename="can_download_document")
        self.view_user.user_permissions.add(view_permission)
        self.download_user.user_permissions.add(view_permission, download_permission)

        user_group_name = getattr(settings, "AUTH_LDAP_DJANGO_USER_GROUP", "User")
        admin_group_name = getattr(settings, "AUTH_LDAP_DJANGO_ADMIN_GROUP", "Admin")
        user_group, _ = Group.objects.get_or_create(name=user_group_name)
        user_download_group, _ = Group.objects.get_or_create(name="UserDownload")
        admin_group, _ = Group.objects.get_or_create(name=admin_group_name)
        superadmin_group, _ = Group.objects.get_or_create(name="SuperAdmin")
        self.view_user.groups.add(user_group)
        self.download_user.groups.add(user_download_group)
        self.admin_user.groups.add(admin_group)
        self.superadmin_user.groups.add(superadmin_group)

        self.department = Department.objects.create(name="Compliance")
        self.doc_type = DocumentType.objects.create(
            name="guideline",
            presentation="Guideline",
        )

        os.makedirs(os.path.join(self.temp_media_dir.name, "documents"), exist_ok=True)
        relative_path = "documents/user-role-preview.pdf"
        absolute_path = os.path.join(self.temp_media_dir.name, relative_path)
        pdf_canvas = canvas.Canvas(absolute_path)
        pdf_canvas.setTitle("User Role Preview")
        pdf_canvas.drawString(72, 720, "User role preview test document")
        pdf_canvas.showPage()
        pdf_canvas.save()

        self.document = Document.objects.create(
            title="User Role Preview",
            description="Preview-only document",
            department=self.department,
            type=self.doc_type,
            version=Decimal("1.0"),
            total_versions=1,
            original_filename="user-role-preview.pdf",
            stored_filename="user-role-preview.pdf",
            file_path=relative_path,
            file_size=os.path.getsize(absolute_path),
            file_format="pdf",
            publisher_id=str(self.view_user.id),
            publisher_name="View Only",
            is_latest_version=True,
            is_active=True,
        )

        office_relative_path = "documents/userdownload-office.docx"
        office_absolute_path = os.path.join(self.temp_media_dir.name, office_relative_path)
        with open(office_absolute_path, "wb") as office_file:
            office_file.write(b"office placeholder")

        office_pdf_path = os.path.splitext(office_absolute_path)[0] + "_view.pdf"
        office_pdf_canvas = canvas.Canvas(office_pdf_path)
        office_pdf_canvas.setTitle("UserDownload Office PDF")
        office_pdf_canvas.drawString(72, 720, "UserDownload office document cached PDF")
        office_pdf_canvas.showPage()
        office_pdf_canvas.save()

        self.office_document = Document.objects.create(
            title="UserDownload Office Document",
            description="Office document with cached PDF preview",
            department=self.department,
            type=self.doc_type,
            version=Decimal("1.0"),
            total_versions=1,
            original_filename="userdownload-office.docx",
            stored_filename="userdownload-office.docx",
            file_path=office_relative_path,
            file_size=os.path.getsize(office_absolute_path),
            file_format="docx",
            publisher_id=str(self.download_user.id),
            publisher_name="Downloader",
            is_latest_version=True,
            is_active=True,
        )

    def test_user_role_download_route_returns_secure_preview_html(self):
        self.client.force_authenticate(self.view_user)

        response = self.client.get(f"/api/documents/{self.document.id}/download/")
        response.render()

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response["Content-Type"].startswith("text/html"))
        self.assertIn("Secure Preview", response.content.decode("utf-8"))

    def test_user_role_pdf_view_route_returns_secure_preview_html(self):
        self.client.force_authenticate(self.view_user)

        response = self.client.get(f"/api/documents/v/{self.document.id}/")
        response.render()

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response["Content-Type"].startswith("text/html"))
        self.assertIn("View-only mode is active", response.content.decode("utf-8"))

    def test_user_role_download_token_hides_download_url(self):
        self.client.force_authenticate(self.view_user)

        response = self.client.get(f"/api/documents/{self.document.id}/download-token/")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data["preview_only"])
        self.assertIsNone(response.data["download_url"])
        self.assertEqual(
            response.data["view_url"],
            f"/api/documents/{self.document.id}/preview/?token={response.data['token']}",
        )

    def test_user_role_cannot_access_original_media_path(self):
        self.client.force_authenticate(self.view_user)

        response = self.client.get("/media/documents/user-role-preview.pdf")

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_other_roles_keep_original_file_access(self):
        role_users = {
            "UserDownload": self.download_user,
            getattr(settings, "AUTH_LDAP_DJANGO_ADMIN_GROUP", "Admin"): self.admin_user,
            "SuperAdmin": self.superadmin_user,
        }

        for role_name, user in role_users.items():
            with self.subTest(role=role_name):
                self.client.force_authenticate(user)
                token_response = self.client.get(f"/api/documents/{self.document.id}/download-token/")

                self.assertEqual(token_response.status_code, status.HTTP_200_OK)
                self.assertFalse(token_response.data["preview_only"])
                self.assertIsNotNone(token_response.data["download_url"])

                response = self.client.get(f"/api/documents/{self.document.id}/download/?view=true")

                self.assertEqual(response.status_code, status.HTTP_200_OK)
                self.assertEqual(response["Content-Type"], "application/pdf")
                self.assertIn("inline;", response["Content-Disposition"])

    def test_userdownload_office_download_token_serves_pdf_attachment(self):
        self.client.force_authenticate(self.download_user)

        token_response = self.client.get(f"/api/documents/{self.office_document.id}/download-token/")

        self.assertEqual(token_response.status_code, status.HTTP_200_OK)
        self.assertFalse(token_response.data["preview_only"])
        self.assertIsNotNone(token_response.data["download_url"])

        response = self.client.get(token_response.data["download_url"])

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertIn("attachment;", response["Content-Disposition"])
        self.assertIn("userdownload-office.pdf", response["Content-Disposition"])

    def test_admin_office_download_token_keeps_original_file_download(self):
        self.client.force_authenticate(self.admin_user)

        token_response = self.client.get(f"/api/documents/{self.office_document.id}/download-token/")

        self.assertEqual(token_response.status_code, status.HTTP_200_OK)
        self.assertFalse(token_response.data["preview_only"])
        self.assertIsNotNone(token_response.data["download_url"])

        response = self.client.get(token_response.data["download_url"])

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response["Content-Type"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        self.assertIn("attachment;", response["Content-Disposition"])
        self.assertIn("userdownload-office.docx", response["Content-Disposition"])


class DocumentIndexingQueueGuardTests(APITestCase):
    def setUp(self):
        self.department = Department.objects.create(name="Ops")
        self.doc_type = DocumentType.objects.create(name="policy", presentation="Policy")
        self.document = Document.objects.create(
            title="Queue Guard Document",
            description="",
            department=self.department,
            type=self.doc_type,
            version=Decimal("1.0"),
            total_versions=1,
            original_filename="queue-guard.docx",
            stored_filename="queue-guard.docx",
            file_path="documents/queue-guard.docx",
            file_size=10,
            file_format="docx",
            is_active=True,
            is_latest_version=True,
            is_vector_processed=False,
        )

    @patch("document.tasks.extract_document_indexing_job.delay")
    def test_process_unprocessed_documents_skips_when_running_or_queued_job_exists(
        self,
        mock_extract_delay,
    ):
        DocumentIndexingJob.objects.create(
            id=uuid4(),
            document=self.document,
            status=DocumentIndexingJob.STATUS_RUNNING,
            stage=DocumentIndexingJob.STAGE_INDEXING,
            source_path="/tmp/source.docx",
            attempts=1,
            error_message="",
            started_at=None,
            finished_at=None,
        )

        process_unprocessed_documents()

        self.assertEqual(
            DocumentIndexingJob.objects.filter(document=self.document).count(),
            1,
        )
        mock_extract_delay.assert_not_called()

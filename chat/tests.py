import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.http import JsonResponse, HttpResponse
from django.test import RequestFactory, SimpleTestCase

from base.services.audio import setup as audio_setup
from base.services.audio.setup import allowed_file
from chat.models import ChatInput
from app.core.views import protected_media, docx_extracted_media
from chat.serializers import ChatInputDetailSerializer
from chat.utils import audio_processing
from chat.views import ensure_voice_file_extension


class VoiceUploadTests(SimpleTestCase):
    def test_allowed_file_accepts_common_browser_audio_formats(self):
        supported_files = [
            "sample.webm",
            "sample.mp4",
            "sample.m4a",
            "sample.mp3",
            "sample.ogg",
            "sample.wav",
            "upload_without_extension",
        ]

        for file_name in supported_files:
            with self.subTest(file_name=file_name):
                self.assertTrue(allowed_file(file_name))

    @patch("chat.utils.audio_prompt_response", return_value="transcribed text")
    def test_audio_processing_passes_uploaded_file_directly_to_transcriber(self, mocked_transcriber):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            temp_file.write(b"fake audio bytes")
            temp_path = temp_file.name

        try:
            result = audio_processing(temp_path)
            self.assertEqual(result, "transcribed text")
            mocked_transcriber.assert_called_once_with(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch("chat.utils.audio_prompt_response", return_value="")
    def test_audio_processing_returns_bad_request_when_no_speech_detected(self, mocked_transcriber):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            temp_file.write(b"fake audio bytes")
            temp_path = temp_file.name

        try:
            result = audio_processing(temp_path)
            self.assertIsInstance(result, JsonResponse)
            self.assertEqual(result.status_code, 400)
            self.assertIn("No speech detected", result.content.decode("utf-8"))
            mocked_transcriber.assert_called_once_with(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_ensure_voice_file_extension_appends_browser_audio_suffix(self):
        uploaded_file = SimpleUploadedFile("blob", b"fake audio bytes", content_type="audio/webm")

        ensure_voice_file_extension(uploaded_file)

        self.assertEqual(uploaded_file.name, "blob.webm")


class ChatInputDetailSerializerTests(SimpleTestCase):
    def test_voice_file_uses_protected_media_url(self):
        chat_input = ChatInput(input_type="voice")
        chat_input.voice_file.name = "voice_inputs/blob"

        data = ChatInputDetailSerializer(chat_input).data

        self.assertEqual(data["voice_file"], "/media/protected/voice_inputs/blob")



class ProtectedMediaViewTests(SimpleTestCase):
    def test_protected_media_serves_voice_input_from_protected_prefix(self):
        request = RequestFactory().get("/media/protected/voice_inputs/blob")

        with tempfile.TemporaryDirectory() as tmpdir:
            media_root = Path(tmpdir)
            voice_file = media_root / "voice_inputs" / "blob"
            voice_file.parent.mkdir(parents=True, exist_ok=True)
            voice_file.write_bytes(b"fake audio bytes")

            with self.settings(MEDIA_ROOT=str(media_root)):
                response = protected_media(request, "protected/voice_inputs/blob")

        self.assertEqual(response["Content-Type"], "audio/webm")
        self.assertEqual(response["X-Accel-Redirect"], "/media/protected/voice_inputs/blob")


class DocxExtractedMediaViewTests(SimpleTestCase):
    def test_docx_extracted_media_attempts_restore_for_missing_file(self):
        request = RequestFactory().get("/docx_extracted_images/sample-doc/image-006.png")

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                self.settings(MEDIA_ROOT=str(tmpdir)),
                patch("app.core.views._restore_missing_docx_inline_image", return_value=True) as mocked_restore,
                patch("app.core.views.protected_media", return_value=HttpResponse("ok")) as mocked_media,
            ):
                response = docx_extracted_media(request, "sample-doc/image-006.png")

        self.assertEqual(response.status_code, 200)
        mocked_restore.assert_called_once_with("docx_extracted_images/sample-doc/image-006.png")
        mocked_media.assert_called_once_with(
            request,
            "docx_extracted_images/sample-doc/image-006.png",
            document_root=None,
        )

    def test_docx_extracted_media_skips_restore_for_existing_file(self):
        request = RequestFactory().get("/docx_extracted_images/sample-doc/image-006.png")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "docx_extracted_images" / "sample-doc" / "image-006.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"image bytes")

            with (
                self.settings(MEDIA_ROOT=str(tmpdir)),
                patch("app.core.views._restore_missing_docx_inline_image") as mocked_restore,
                patch("app.core.views.protected_media", return_value=HttpResponse("ok")) as mocked_media,
            ):
                response = docx_extracted_media(request, "sample-doc/image-006.png")

        self.assertEqual(response.status_code, 200)
        mocked_restore.assert_not_called()
        mocked_media.assert_called_once_with(
            request,
            "docx_extracted_images/sample-doc/image-006.png",
            document_root=None,
        )



class AudioTranscriptionFallbackTests(SimpleTestCase):
    @patch("base.services.audio.setup.os.remove")
    @patch("base.services.audio.setup.os.path.exists", return_value=True)
    @patch("base.services.audio.setup.resample_audio", return_value="/tmp/resampled.wav")
    @patch("base.services.audio.setup.get_transcription_model")
    def test_audio_prompt_response_retries_without_vad_when_first_pass_is_empty(
        self,
        mocked_get_model,
        mocked_resample_audio,
        mocked_exists,
        mocked_remove,
    ):
        class Segment:
            def __init__(self, text):
                self.text = text

        mocked_model = mocked_get_model.return_value
        mocked_model.transcribe.side_effect = [
            ([], object()),
            ([Segment("hello world")], object()),
        ]

        transcription = audio_setup.audio_prompt_response("/tmp/input.webm")

        self.assertEqual(transcription, "hello world")
        self.assertEqual(mocked_model.transcribe.call_count, 2)
        self.assertTrue(mocked_model.transcribe.call_args_list[0].kwargs["vad_filter"])
        self.assertFalse(mocked_model.transcribe.call_args_list[1].kwargs["vad_filter"])
        mocked_resample_audio.assert_called_once_with("/tmp/input.webm")
        mocked_remove.assert_called_once_with("/tmp/resampled.wav")


class WhisperModelPathResolutionTests(SimpleTestCase):
    def test_get_whisper_model_base_path_prefers_project_root_models(self):
        with (
            patch.dict(audio_setup.os.environ, {}, clear=True),
            self.settings(BASE_DIR="/usr/src/app/app", PROJECT_ROOT="/usr/src/app"),
            patch("base.services.audio.setup.os.path.exists") as mocked_exists,
        ):
            mocked_exists.side_effect = (
                lambda path: path == "/usr/src/app/models/whisper-base-khmer-ct2"
            )
            resolved_path = audio_setup.get_whisper_model_base_path()

        self.assertEqual(resolved_path, "/usr/src/app/models/whisper-base-khmer-ct2")

    def test_get_whisper_model_base_path_honors_env_override(self):
        with patch.dict(
            audio_setup.os.environ,
            {"WHISPER_MODEL_PATH": "/custom/whisper-model"},
            clear=True,
        ):
            resolved_path = audio_setup.get_whisper_model_base_path()

        self.assertEqual(resolved_path, "/custom/whisper-model")

    def test_find_local_whisper_tokenizer_path_prefers_project_root_cache(self):
        with (
            patch.dict(audio_setup.os.environ, {}, clear=True),
            self.settings(BASE_DIR="/usr/src/app/app", PROJECT_ROOT="/usr/src/app"),
            patch("base.services.audio.setup.os.path.isdir") as mocked_isdir,
            patch("base.services.audio.setup.os.listdir", return_value=["snapshot-a"]),
            patch("base.services.audio.setup.os.path.isfile") as mocked_isfile,
            patch("base.services.audio.setup.os.path.getsize", return_value=1024),
        ):
            mocked_isdir.side_effect = (
                lambda path: path == "/usr/src/app/models/hub/models--openai--whisper-tiny/snapshots"
            )
            mocked_isfile.side_effect = (
                lambda path: path == "/usr/src/app/models/hub/models--openai--whisper-tiny/snapshots/snapshot-a/tokenizer.json"
            )
            resolved_path = audio_setup.find_local_whisper_tokenizer_path()

        self.assertEqual(
            resolved_path,
            "/usr/src/app/models/hub/models--openai--whisper-tiny/snapshots/snapshot-a/tokenizer.json",
        )

    def test_ensure_local_whisper_tokenizer_copies_cached_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_path = tmp_path / "model"
            model_path.mkdir()
            source_path = tmp_path / "cache" / "tokenizer.json"
            source_path.parent.mkdir()
            source_path.write_text('{"version": "1.0"}')

            with patch(
                "base.services.audio.setup.find_local_whisper_tokenizer_path",
                return_value=str(source_path),
            ):
                tokenizer_path = audio_setup.ensure_local_whisper_tokenizer(str(model_path))

            copied_path = model_path / "tokenizer.json"
            self.assertEqual(tokenizer_path, str(copied_path))
            self.assertTrue(copied_path.exists())
            self.assertEqual(copied_path.read_text(), source_path.read_text())

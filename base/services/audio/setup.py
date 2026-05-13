import os
import shutil

import librosa
import soundfile as sf
from django.conf import settings
from faster_whisper import WhisperModel

ALLOWED_EXTENSIONS = {"wav", "webm", "mp4", "m4a", "mp3", "ogg"}

# Global model instance (load once)
_whisper_model = None
_model_load_error = None  # Cache model loading errors


def get_whisper_model_base_path():
    configured_path = os.environ.get("WHISPER_MODEL_PATH", "").strip()
    if configured_path:
        return configured_path

    candidate_paths = []
    project_root = getattr(settings, "PROJECT_ROOT", None)
    if project_root:
        candidate_paths.append(
            os.path.join(str(project_root), "models", "whisper-base-khmer-ct2")
        )
    candidate_paths.append(
        os.path.join(str(settings.BASE_DIR), "models", "whisper-base-khmer-ct2")
    )

    seen = set()
    deduped_paths = []
    for path in candidate_paths:
        normalized = os.path.normpath(path)
        if normalized not in seen:
            deduped_paths.append(normalized)
            seen.add(normalized)

    for path in deduped_paths:
        if os.path.exists(path):
            return path

    return deduped_paths[0]


def resolve_whisper_model_path():
    base_model_path = get_whisper_model_base_path()

    # Find the actual model in snapshots directory (DVC structure)
    snapshot_ref_file = os.path.join(base_model_path, "refs", "main")
    if os.path.exists(snapshot_ref_file):
        with open(snapshot_ref_file, "r") as f:
            snapshot_hash = f.read().strip()
        return os.path.join(base_model_path, "snapshots", snapshot_hash)

    # Fallback: look for any directory with model.bin
    snapshots_dir = os.path.join(base_model_path, "snapshots")
    if os.path.exists(snapshots_dir):
        for snapshot in os.listdir(snapshots_dir):
            candidate = os.path.join(snapshots_dir, snapshot)
            if os.path.isdir(candidate) and os.path.exists(
                os.path.join(candidate, "model.bin")
            ):
                return candidate

    return base_model_path


def is_valid_tokenizer_file(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def find_local_whisper_tokenizer_path():
    configured_path = os.environ.get("WHISPER_TOKENIZER_PATH", "").strip()
    if configured_path and is_valid_tokenizer_file(configured_path):
        return configured_path

    candidate_roots = []
    project_root = getattr(settings, "PROJECT_ROOT", None)
    if project_root:
        candidate_roots.append(os.path.join(str(project_root), "models", "hub"))
    candidate_roots.append(os.path.join(str(settings.BASE_DIR), "models", "hub"))

    seen = set()
    for root in candidate_roots:
        normalized_root = os.path.normpath(root)
        if normalized_root in seen:
            continue
        seen.add(normalized_root)

        for model_id in ("models--openai--whisper-tiny", "models--openai--whisper-tiny.en"):
            snapshots_dir = os.path.join(normalized_root, model_id, "snapshots")
            if not os.path.isdir(snapshots_dir):
                continue

            for snapshot in sorted(os.listdir(snapshots_dir)):
                candidate = os.path.join(snapshots_dir, snapshot, "tokenizer.json")
                if is_valid_tokenizer_file(candidate):
                    return candidate

    return None


def ensure_local_whisper_tokenizer(model_path):
    tokenizer_path = os.path.join(model_path, "tokenizer.json")
    if is_valid_tokenizer_file(tokenizer_path):
        return tokenizer_path
    if os.path.isfile(tokenizer_path):
        os.remove(tokenizer_path)

    source_tokenizer_path = find_local_whisper_tokenizer_path()
    if not source_tokenizer_path:
        return None

    shutil.copyfile(source_tokenizer_path, tokenizer_path)
    print(f"Copied Whisper tokenizer from {source_tokenizer_path} to {tokenizer_path}")
    return tokenizer_path


def check_model_integrity(model_path):
    """
    Check if the Whisper model files exist and appear valid
    Returns: (is_valid, error_message)
    """
    if not os.path.exists(model_path):
        return False, f"Model directory not found: {model_path}"

    # Check for required model files
    required_files = {
        "model.bin": {"min_size": 100_000_000},  # At least 100MB
        "config.json": {"min_size": 100},  # At least 100 bytes
        "vocabulary.json": {"min_size": 1000},  # At least 1KB
    }

    for filename, requirements in required_files.items():
        file_path = os.path.join(model_path, filename)

        if not os.path.exists(file_path):
            return False, f"Required model file missing: {filename}"

        file_size = os.path.getsize(file_path)
        if file_size < requirements["min_size"]:
            return False, (
                f"Model file '{filename}' appears incomplete or corrupted. "
                f"Size: {file_size:,} bytes (expected at least {requirements['min_size']:,} bytes). "
                "Please re-download the model."
            )

    # All checks passed
    return True, None


def get_transcription_model():
    global _whisper_model, _model_load_error

    # If we previously failed to load, don't retry every time
    if _model_load_error is not None:
        raise Exception(_model_load_error)

    if _whisper_model is None:
        model_path = resolve_whisper_model_path()

        print(f"Loading Whisper model from: {model_path}")

        # Check model integrity first
        is_valid, error_msg = check_model_integrity(model_path)
        if not is_valid:
            _model_load_error = (
                f"Voice transcription unavailable: {error_msg} "
                "Please contact support to reinstall the model."
            )
            print(f"✗ Model integrity check failed: {error_msg}")
            raise Exception(_model_load_error)

        print("✓ Model integrity check passed")

        tokenizer_path = ensure_local_whisper_tokenizer(model_path)
        if tokenizer_path is None:
            error_msg = (
                "Whisper tokenizer is missing from the model snapshot and no local fallback tokenizer "
                "cache was found."
            )
            _model_load_error = error_msg
            print(f"✗ {error_msg}")
            raise Exception(error_msg)

        # Check GPU availability first
        gpu_available = False
        try:
            import torch

            gpu_available = torch.cuda.is_available()
            if gpu_available:
                print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠ No GPU detected, will use CPU")
        except ImportError:
            print("⚠ PyTorch not available, will attempt GPU anyway")
            gpu_available = None  # Unknown, let whisper try

        # Try GPU first if available, fallback to CPU
        if gpu_available or gpu_available is None:
            try:
                print("Attempting to load Whisper model on GPU...")
                _whisper_model = WhisperModel(
                    model_path,
                    device="cuda",
                    compute_type="float16",
                    local_files_only=True,
                )
                print("✓ Whisper model loaded on GPU successfully")
            except Exception as e:
                print(f"⚠ GPU loading failed ({e}), falling back to CPU...")
                try:
                    _whisper_model = WhisperModel(
                        model_path,
                        device="cpu",
                        compute_type="int8",
                        local_files_only=True,
                    )
                    print("✓ Whisper model loaded on CPU")
                except Exception as cpu_error:
                    error_msg = (
                        "Failed to load Whisper model on both GPU and CPU. "
                        f"The model files may be corrupted. Error: {str(cpu_error)}"
                    )
                    _model_load_error = error_msg
                    print(f"✗ {error_msg}")
                    raise Exception(error_msg)
        else:
            # GPU not available, load directly on CPU
            print("Loading Whisper model on CPU...")
            try:
                _whisper_model = WhisperModel(
                    model_path,
                    device="cpu",
                    compute_type="int8",
                    local_files_only=True,
                )
                print("✓ Whisper model loaded on CPU")
            except Exception as e:
                error_msg = (
                    "Failed to load Whisper model on CPU. "
                    f"The model files may be corrupted. Error: {str(e)}"
                )
                _model_load_error = error_msg
                print(f"✗ {error_msg}")
                raise Exception(error_msg)
    return _whisper_model


def allowed_file(filename):
    """Check if file has allowed extension or content type"""
    # If no extension, assume it's valid (Django stripped it)
    if "." not in filename:
        return True

    ext = filename.rsplit(".", 1)[1].lower()

    # Allow Django's temporary upload files (.upload extension)
    if ext == "upload":
        return True

    return ext in ALLOWED_EXTENSIONS


def resample_audio(audio):
    try:
        waveform, sample_rate = librosa.load(audio, sr=None)
        if waveform is None or sample_rate is None:
            raise ValueError("Failed to load audio file")

        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            waveform = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=target_sample_rate
            )
            sample_rate = target_sample_rate

        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        resampled_path = os.path.join(
            settings.MEDIA_ROOT,
            f"resampled_{os.path.splitext(os.path.basename(audio))[0]}.wav",
        )
        sf.write(resampled_path, waveform, sample_rate, format="WAV", subtype="PCM_16")
        return resampled_path
    except Exception as e:
        raise e


def _transcribe_audio(model, audio_path, *, vad_filter):
    segments, _info = model.transcribe(
        audio_path,
        language="km",  # Set to "km" for Khmer, or None for auto-detection
        beam_size=10,  # Increased for RTX 5090 (better accuracy)
        vad_filter=vad_filter,
        word_timestamps=False,
        best_of=5,  # Use best of 5 candidates (higher quality)
    )
    return " ".join([segment.text.strip() for segment in segments]).strip()


def audio_prompt_response(audio):
    resampled_audio = None
    try:
        model = get_transcription_model()
        resampled_audio = resample_audio(audio)

        transcription = _transcribe_audio(model, resampled_audio, vad_filter=True)
        if transcription:
            return transcription

        print("VAD removed the full clip; retrying transcription without VAD")
        transcription = _transcribe_audio(model, resampled_audio, vad_filter=False)
        if transcription:
            return transcription

        raise ValueError(
            "No speech detected in the audio. Please speak more clearly and try again."
        )
    except FileNotFoundError as e:
        print(f"Model not found: {e}")
        raise Exception(
            "Voice transcription is currently unavailable. The speech recognition model is not installed."
        )
    except Exception as e:
        error_msg = str(e)
        print(f"Error in audio processing: {error_msg}")

        # Provide user-friendly error messages
        if "no speech detected" in error_msg.lower():
            raise Exception(error_msg)
        if "model.bin is incomplete" in error_msg or "failed to read" in error_msg:
            raise Exception(
                "Voice transcription is currently unavailable. "
                "The speech recognition model is corrupted and needs to be reinstalled. "
                "Please contact support or use text input instead."
            )
        elif "model" in error_msg.lower():
            raise Exception(
                "Voice transcription is currently unavailable. "
                "There was an issue loading the speech recognition model. "
                "Please use text input instead."
            )
        else:
            raise Exception(f"Voice transcription failed: {error_msg}")
    finally:
        if resampled_audio and os.path.exists(resampled_audio):
            os.remove(resampled_audio)

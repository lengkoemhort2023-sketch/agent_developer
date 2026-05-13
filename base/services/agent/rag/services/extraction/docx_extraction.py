import re
import os
import hashlib
import struct
import logging
import json
import io
import itertools
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin
from django.conf import settings
from langchain_core.documents import Document as LangChainDocument
import requests
import tempfile
from markdownify import markdownify as html2md
from markdownify import MarkdownConverter
import markdown as py_markdown
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

DOCX_PAGE_MARKER_PATTERN = re.compile(r"^\s*\[\[PAGE_(\d+)\]\]\s*$", re.IGNORECASE)
DOCX_SOFT_PAGE_UNIT_BUDGET = max(600, int(os.environ.get("DOCX_SOFT_PAGE_UNIT_BUDGET", "1800")))

CONTENT_TYPE_TO_EXTENSION = {
    "image/emf": ".emf",
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/x-png": ".png",
    "image/svg+xml": ".svg",
    "image/tiff": ".tif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/x-emf": ".emf",
    "image/x-wmf": ".wmf",
}


def _normalize_content_type(content_type: Any) -> str:
    return str(content_type or "").split(";", 1)[0].strip().lower()


def _get_doc_converter_timeout_seconds() -> Optional[float]:
    default_timeout = 120.0
    raw_timeout = str(os.environ.get("DOC_CONVERTER_TIMEOUT_SECONDS", "")).strip()
    if not raw_timeout:
        return default_timeout
    try:
        timeout = float(raw_timeout)
    except ValueError:
        logger.warning(
            "Invalid DOC_CONVERTER_TIMEOUT_SECONDS=%r; falling back to default %.1fs",
            raw_timeout,
            default_timeout,
        )
        return default_timeout
    if timeout <= 0:
        return default_timeout
    return timeout


def convert_docx_to_pdf(file_bytes: bytes, file_name: str) -> bytes | None:
    timeout_seconds = _get_doc_converter_timeout_seconds()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        try:
            with open(temp_file_path, "rb") as f:
                request_kwargs: dict[str, Any] = {"files": {"file": f}}
                if timeout_seconds is not None:
                    request_kwargs["timeout"] = timeout_seconds
                r = requests.post(settings.DOC_CONVERTER_URL, **request_kwargs)

            if r.status_code == 200:
                print(f"Successfully converted DOCX to PDF: {file_name}")
                return r.content
            else:
                print(f"PDF conversion failed for {file_name}: HTTP {r.status_code}")
                return None

        finally:
            os.unlink(temp_file_path)

    except requests.exceptions.Timeout:
        timeout_label = timeout_seconds if timeout_seconds is not None else "unset"
        print(
            f"DOCX to PDF conversion timed out for {file_name} "
            f"(timeout={timeout_label}s)"
        )
        return None
    except requests.exceptions.RequestException as e:
        print(f"Network error during PDF conversion for {file_name}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error during PDF conversion for {file_name}: {str(e)}")
        return None


def _resolve_extracted_image_output_dir(file_id: str) -> Optional[Path]:
    media_root = getattr(settings, "MEDIA_ROOT", None)
    raw = str(media_root or "").strip()
    if not raw or raw == ".":
        return None
    return Path(raw) / "docx_extracted_images" / str(file_id)


def _build_image_link_metadata(file_id: str, page_number: Optional[int] = None) -> dict[str, str]:
    media_url = str(getattr(settings, "MEDIA_URL", "") or "").strip() or "/media/"
    if not media_url.endswith("/"):
        media_url += "/"

    image_dir_rel = f"docx_extracted_images/{file_id}"
    metadata = {
        "image_directory_path": image_dir_rel,
        "image_directory_url": urljoin(media_url, f"{image_dir_rel}/"),
        "image_manifest_path": f"{image_dir_rel}/manifest.json",
        "image_manifest_url": urljoin(media_url, f"{image_dir_rel}/manifest.json"),
    }

    if page_number is not None:
        page_filename = f"page_{int(page_number):04d}.png"
        page_rel_path = f"{image_dir_rel}/{page_filename}"
        metadata["page_image_path"] = page_rel_path
        metadata["page_image_url"] = urljoin(media_url, page_rel_path)

    return metadata


def _persist_pdf_page_images(
    pdf_bytes: bytes,
    file_id: str,
    max_pages: Optional[int] = None,
) -> dict[str, Any]:
    output_dir = _resolve_extracted_image_output_dir(file_id)
    if output_dir is None:
        return {"saved": 0, "output_dir": None}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_bytes)
        temp_pdf_path = temp_file.name

    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # type: ignore[no-redef]

        output_dir.mkdir(parents=True, exist_ok=True)
        for stale_png in output_dir.glob("page_*.png"):
            try:
                stale_png.unlink()
            except OSError:
                pass

        pdf_document = fitz.open(temp_pdf_path)
        try:
            total_pages = pdf_document.page_count
            pages_to_render = min(total_pages, max_pages) if max_pages else total_pages
            saved_count = 0
            zoom = 150 / 72
            matrix = fitz.Matrix(zoom, zoom)

            for page_idx in range(pages_to_render):
                page = pdf_document[page_idx]
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image_path = output_dir / f"page_{page_idx + 1:04d}.png"
                pixmap.save(str(image_path))
                saved_count += 1
        finally:
            pdf_document.close()

        manifest_path = output_dir / "manifest.json"
        manifest_payload = {
            "file_id": str(file_id),
            "pages_total": total_pages,
            "pages_rendered": saved_count,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest_payload, manifest_file, ensure_ascii=False, indent=2)

        return {
            "saved": saved_count,
            "output_dir": str(output_dir),
            "manifest_path": str(manifest_path),
        }
    except Exception as exc:
        logger.warning("Failed to persist DOCX page images for %s: %s", file_id, exc)
        return {"saved": 0, "output_dir": str(output_dir), "error": str(exc)}
    finally:
        try:
            os.unlink(temp_pdf_path)
        except OSError:
            pass


def _count_pdf_pages(pdf_bytes: bytes) -> int:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_bytes)
        temp_pdf_path = temp_file.name

    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # type: ignore[no-redef]

        pdf_document = fitz.open(temp_pdf_path)
        try:
            return int(pdf_document.page_count or 0)
        finally:
            pdf_document.close()
    except Exception as exc:
        logger.warning("Failed to count PDF pages for DOCX conversion: %s", exc)
        return 0
    finally:
        try:
            os.unlink(temp_pdf_path)
        except OSError:
            pass


def _persist_docx_inline_images(file_bytes: bytes, file_id: str) -> dict[str, Any]:
    """
    Persist embedded DOCX images (word/media/*) to the same media folder used by page images.
    """
    output_dir = _resolve_extracted_image_output_dir(file_id)
    if output_dir is None:
        return {"saved": 0, "output_dir": None}

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale_image in output_dir.glob("inline_*"):
            try:
                stale_image.unlink()
            except OSError:
                pass

        saved_files: list[str] = []
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx_zip:
            media_entries = sorted(
                name
                for name in docx_zip.namelist()
                if name.startswith("word/media/") and not name.endswith("/")
            )

            for index, entry_name in enumerate(media_entries, start=1):
                raw_name = Path(entry_name).name
                suffix = Path(raw_name).suffix.lower()
                if not suffix:
                    suffix = ".bin"
                output_name = f"inline_{index:04d}{suffix}"
                output_path = output_dir / output_name

                with docx_zip.open(entry_name) as src_file:
                    with open(output_path, "wb") as target_file:
                        target_file.write(src_file.read())

                saved_files.append(output_name)

        return {
            "saved": len(saved_files),
            "output_dir": str(output_dir),
            "files": saved_files,
        }
    except Exception as exc:
        logger.warning("Failed to persist DOCX inline images for %s: %s", file_id, exc)
        return {"saved": 0, "output_dir": str(output_dir), "error": str(exc)}


def _build_inline_image_metadata(file_id: str, image_files: list[str]) -> dict[str, Any]:
    media_url = str(getattr(settings, "MEDIA_URL", "") or "").strip() or "/media/"
    if not media_url.endswith("/"):
        media_url += "/"

    image_dir_rel = f"docx_extracted_images/{file_id}"
    image_dir_url = urljoin(media_url, f"{image_dir_rel}/")

    inline_images = []
    for file_name in image_files:
        rel_path = f"{image_dir_rel}/{file_name}"
        inline_images.append(
            {
                "name": file_name,
                "path": rel_path,
                "url": urljoin(media_url, rel_path),
            }
        )

    metadata: dict[str, Any] = {
        "inline_image_count": len(inline_images),
        "inline_image_urls": [item["url"] for item in inline_images],
        "inline_images": inline_images,
    }

    if inline_images:
        metadata["inline_image_url"] = inline_images[0]["url"]
        metadata["inline_image_path"] = inline_images[0]["path"]
        metadata["image_directory_url"] = image_dir_url

    return metadata


def _build_inline_image_html(inline_image_url: Optional[str]) -> str:
    if not inline_image_url:
        return ""
    return f'<p><img src="{inline_image_url}" loading="lazy" /></p>'


def _clean_markdown(markdown: str) -> str:
    markdown = markdown.replace("\r\n", "\n")
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip() + "\n"


def _count_text_units(text: str) -> int:
    if not text:
        return 0
    # Khmer-safe approximation: count non-whitespace codepoints.
    compact = re.sub(r"\s+", "", text)
    return len(compact)


def _strip_existing_page_markers(markdown_text: str) -> str:
    if not markdown_text:
        return ""
    filtered_lines = [
        line for line in markdown_text.splitlines()
        if not DOCX_PAGE_MARKER_PATTERN.match(line)
    ]
    return "\n".join(filtered_lines)


def _collect_docx_page_break_offsets(file_bytes: bytes) -> tuple[list[int], int]:
    """
    Parse word/document.xml and estimate cumulative text-unit offsets where page
    transitions occur, based on explicit Word XML break signals.
    """
    offsets: list[int] = []
    total_units = 0
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            xml_bytes = archive.read("word/document.xml")
    except Exception as exc:
        logger.debug("Unable to read DOCX XML for page markers: %s", exc)
        return offsets, total_units

    try:
        root = ET.fromstring(xml_bytes)
        body = root.find("w:body", ns)
        if body is None:
            return offsets, total_units

        for block in list(body):
            block_text = " ".join(
                (node.text or "")
                for node in block.findall(".//w:t", ns)
                if (node.text or "").strip()
            ).strip()
            total_units += max(1, _count_text_units(block_text))

            break_count = 0
            break_count += len(block.findall(".//w:lastRenderedPageBreak", ns))
            break_count += len(block.findall(".//w:br[@w:type='page']", ns))

            # Section property within paragraph/table often indicates pagination boundary.
            if block.find(".//w:pPr/w:sectPr", ns) is not None:
                break_count += 1

            for _ in range(break_count):
                offsets.append(total_units)
    except Exception as exc:
        logger.debug("Failed parsing DOCX XML pagination markers: %s", exc)
        return [], 0

    return offsets, total_units


def _inject_page_markers_from_xml_offsets(markdown_text: str, file_bytes: bytes) -> str:
    base_markdown = _strip_existing_page_markers(markdown_text)
    if not base_markdown.strip():
        return base_markdown

    xml_offsets, xml_total_units = _collect_docx_page_break_offsets(file_bytes)
    if not xml_offsets or xml_total_units <= 0:
        return base_markdown

    lines = base_markdown.splitlines()
    if not lines:
        return base_markdown

    line_units = [_count_text_units(line) for line in lines]
    markdown_total_units = max(1, sum(line_units))

    mapped_units: list[int] = []
    for offset in xml_offsets:
        mapped = int(round((offset / xml_total_units) * markdown_total_units))
        if mapped > 0:
            mapped_units.append(mapped)
    if not mapped_units:
        return base_markdown

    mapped_units.sort()

    insertion_points: list[int] = []
    cursor = 0
    for mapped in mapped_units:
        cumulative = 0
        insert_idx = len(lines)
        for idx, units in enumerate(line_units):
            cumulative += units
            if cumulative >= mapped:
                insert_idx = idx
                break
        if insert_idx <= cursor and insertion_points:
            insert_idx = min(len(lines), cursor + 1)
        insertion_points.append(insert_idx)
        cursor = insert_idx

    # Deduplicate consecutive identical points so page numbers remain monotonic.
    normalized_points: list[int] = []
    last_idx: Optional[int] = None
    for idx in insertion_points:
        if last_idx is None or idx > last_idx:
            normalized_points.append(idx)
            last_idx = idx

    if not normalized_points:
        return base_markdown

    output_lines = list(lines)
    offset = 0
    for page_number, line_index in enumerate(normalized_points, start=2):
        marker = f"[[PAGE_{page_number}]]"
        output_lines.insert(min(len(output_lines), line_index + offset), marker)
        offset += 1

    logger.info(
        "Injected %s DOCX page markers from XML page-break signals",
        len(normalized_points),
    )
    return "\n".join(output_lines)


def _markdown_to_plain_text(markdown: str) -> str:
    text = markdown
    text = text.replace("\\\n", "\n")
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"</?[^>]+>", "", text)
    return sanitize_content(text).strip()


def _extract_image_blocks(raw: str) -> list[tuple[str, Optional[str]]]:
    if not raw:
        return []

    blocks: list[tuple[str, Optional[str]]] = []
    i = 0
    raw_len = len(raw)

    while i < raw_len:
        start = raw.find("![", i)
        if start == -1:
            break

        cursor = start + 2
        alt_chars: list[str] = []
        escaped = False
        while cursor < raw_len:
            ch = raw[cursor]
            if escaped:
                alt_chars.append(ch)
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "]":
                break
            else:
                alt_chars.append(ch)
            cursor += 1

        if cursor >= raw_len or raw[cursor] != "]":
            i = start + 2
            continue

        cursor += 1
        while cursor < raw_len and raw[cursor].isspace():
            cursor += 1
        if cursor >= raw_len or raw[cursor] != "(":
            i = start + 2
            continue

        cursor += 1
        while cursor < raw_len and raw[cursor].isspace():
            cursor += 1
        if cursor >= raw_len:
            break

        path_chars: list[str] = []
        end_index: Optional[int] = None

        if raw[cursor] == "<":
            cursor += 1
            escaped = False
            while cursor < raw_len:
                ch = raw[cursor]
                if escaped:
                    path_chars.append(ch)
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == ">":
                    cursor += 1
                    break
                else:
                    path_chars.append(ch)
                cursor += 1

            quote_char: Optional[str] = None
            escaped = False
            while cursor < raw_len:
                ch = raw[cursor]
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif quote_char:
                    if ch == quote_char:
                        quote_char = None
                elif ch in ("'", '"'):
                    quote_char = ch
                elif ch == ")":
                    end_index = cursor
                    break
                cursor += 1
        else:
            depth = 0
            escaped = False
            while cursor < raw_len:
                ch = raw[cursor]
                if escaped:
                    path_chars.append(ch)
                    escaped = False
                    cursor += 1
                    continue

                if ch == "\\":
                    escaped = True
                    cursor += 1
                    continue

                if ch == "(":
                    depth += 1
                    path_chars.append(ch)
                    cursor += 1
                    continue

                if ch == ")":
                    if depth == 0:
                        end_index = cursor
                        break
                    depth -= 1
                    path_chars.append(ch)
                    cursor += 1
                    continue

                if ch.isspace() and depth == 0:
                    title_cursor = cursor
                    quote_char: Optional[str] = None
                    escaped_title = False
                    while title_cursor < raw_len:
                        t_char = raw[title_cursor]
                        if escaped_title:
                            escaped_title = False
                        elif t_char == "\\":
                            escaped_title = True
                        elif quote_char:
                            if t_char == quote_char:
                                quote_char = None
                        elif t_char in ("'", '"'):
                            quote_char = t_char
                        elif t_char == ")":
                            end_index = title_cursor
                            break
                        title_cursor += 1
                    break

                path_chars.append(ch)
                cursor += 1

        if end_index is None:
            i = start + 2
            continue

        image_path = "".join(path_chars).strip()
        alt_text = sanitize_content("".join(alt_chars))
        if image_path:
            blocks.append((image_path, alt_text or None))
        i = end_index + 1

    return blocks


_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
_METAFILE_EXTS = {".emf", ".wmf"}

_MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
_DRAWINGML_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_RELS_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_VML_NS = "urn:schemas-microsoft-com:vml"


def _build_vml_to_drawingml_map(docx_bytes: bytes) -> dict[bytes, tuple[str, bytes]]:
    """
    Scan mc:AlternateContent blocks in the DOCX for EMF/WMF VML fallbacks that
    have a raster DrawingML counterpart (PNG/JPEG) in the same element.
    Returns {md5(emf_bytes): (preferred_ext, preferred_bytes)}.
    """
    result: dict[bytes, tuple[str, bytes]] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
            try:
                rels_xml = zf.read("word/_rels/document.xml.rels")
                doc_xml = zf.read("word/document.xml")
            except KeyError:
                return result

            rel_to_path: dict[str, str] = {}
            for rel in ET.fromstring(rels_xml):
                rel_id = rel.get("Id", "")
                target = rel.get("Target", "")
                if rel_id and target:
                    zip_path = f"word/{target}" if not target.startswith("/") else target.lstrip("/")
                    rel_to_path[rel_id] = zip_path

            for alt in ET.fromstring(doc_xml).iter(f"{{{_MC_NS}}}AlternateContent"):
                drawingml_rel = None
                for blip in alt.iter(f"{{{_DRAWINGML_NS}}}blip"):
                    embed = blip.get(f"{{{_RELS_NS}}}embed")
                    if embed:
                        drawingml_rel = embed
                        break

                vml_rel = None
                for imgdata in alt.iter(f"{{{_VML_NS}}}imagedata"):
                    rid = imgdata.get(f"{{{_RELS_NS}}}id") or imgdata.get(f"{{{_RELS_NS}}}href")
                    if rid:
                        vml_rel = rid
                        break

                if not drawingml_rel or not vml_rel:
                    continue

                drawingml_path = rel_to_path.get(drawingml_rel, "")
                vml_path = rel_to_path.get(vml_rel, "")

                if (
                    Path(vml_path).suffix.lower() not in _METAFILE_EXTS
                    or Path(drawingml_path).suffix.lower() not in _RASTER_EXTS
                ):
                    continue

                try:
                    emf_bytes = zf.read(vml_path)
                    raster_bytes = zf.read(drawingml_path)
                except KeyError:
                    continue

                ext = Path(drawingml_path).suffix.lower().lstrip(".")
                result[hashlib.md5(emf_bytes).digest()] = (ext, raster_bytes)

    except Exception as exc:
        logger.debug("_build_vml_to_drawingml_map failed: %s", exc)

    return result


def _parse_emf_size_mm(emf_bytes: bytes) -> tuple[float, float]:
    """Return (width_mm, height_mm) from the EMF header frame rect. Falls back to 150×100."""
    try:
        if len(emf_bytes) >= 40 and struct.unpack_from("<I", emf_bytes, 0)[0] == 1:
            left, top, right, bottom = struct.unpack_from("<iiii", emf_bytes, 24)
            w, h = (right - left) / 100.0, (bottom - top) / 100.0
            if w > 0 and h > 0:
                return w, h
    except Exception:
        pass
    return 150.0, 100.0


def _build_minimal_docx_with_emf(emf_bytes: bytes) -> bytes:
    """Wrap a single EMF image in a minimal DOCX, sized to the image dimensions."""
    img_w, img_h = _parse_emf_size_mm(emf_bytes)
    margin = 10.0
    T = 56.692  # mm → twips

    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
        ' xmlns:v="urn:schemas-microsoft-com:vml"'
        ' xmlns:o="urn:schemas-microsoft-com:office:office"'
        ' xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:pict>"
        f'<v:shape style="width:{img_w:.2f}mm;height:{img_h:.2f}mm">'
        '<v:imagedata r:id="rId1" o:title=""/>'
        "</v:shape>"
        "</w:pict></w:r></w:p>"
        f"<w:sectPr>"
        f'<w:pgSz w:w="{int((img_w + 2*margin)*T)}" w:h="{int((img_h + 2*margin)*T)}"/>'
        f'<w:pgMar w:top="{int(margin*T)}" w:right="{int(margin*T)}"'
        f' w:bottom="{int(margin*T)}" w:left="{int(margin*T)}"/>'
        "</w:sectPr>"
        "</w:body></w:document>"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Default Extension="emf" ContentType="image/x-emf"/>'
            '<Override PartName="/word/document.xml"'
            ' ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1"'
            ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
            ' Target="word/document.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "word/_rels/document.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1"'
            ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"'
            ' Target="media/image1.emf"/>'
            "</Relationships>",
        )
        zf.writestr("word/document.xml", doc_xml)
        zf.writestr("word/media/image1.emf", emf_bytes)
    return buf.getvalue()


def _convert_emf_via_doc_converter(emf_bytes: bytes) -> bytes | None:
    """
    Convert EMF bytes → PNG bytes by wrapping in a minimal DOCX,
    sending to DOC_CONVERTER_URL → PDF, then rendering page 0 with PyMuPDF.
    """
    converter_url = getattr(settings, "DOC_CONVERTER_URL", None)
    if not converter_url:
        logger.debug("DOC_CONVERTER_URL not set; skipping EMF conversion")
        return None

    docx_bytes = _build_minimal_docx_with_emf(emf_bytes)
    try:
        resp = requests.post(
            converter_url,
            files={"file": ("emf_wrap.docx", docx_bytes,
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            timeout=60,
        )
        if resp.status_code != 200:
            logger.warning("Doc converter returned %s during EMF conversion", resp.status_code)
            return None
        pdf_bytes = resp.content
    except Exception as exc:
        logger.warning("Doc converter request failed during EMF conversion: %s", exc)
        return None

    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # type: ignore[no-redef]
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            pix = doc[0].get_pixmap(dpi=150)
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception as exc:
        logger.warning("PyMuPDF render failed during EMF conversion: %s", exc)
        return None


def _build_mammoth_image_converter(file_id: str, docx_bytes: bytes | None = None):
    counter = itertools.count(1)
    output_dir = _resolve_extracted_image_output_dir(file_id)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    media_url = str(getattr(settings, "MEDIA_URL", "") or "").strip() or "/media/"
    if not media_url.endswith("/"):
        media_url += "/"

    vml_map = _build_vml_to_drawingml_map(docx_bytes) if docx_bytes else {}

    def convert_image(image):
        normalized_content_type = _normalize_content_type(getattr(image, "content_type", ""))
        extension = CONTENT_TYPE_TO_EXTENSION.get(normalized_content_type, ".bin")
        image_index = next(counter)

        if output_dir is None:
            with image.open() as raw:
                raw.read()
            return {"src": f"image-{image_index:03d}{extension}"}

        if extension.lower() in _METAFILE_EXTS:
            with image.open() as raw:
                emf_bytes = raw.read()

            preferred = vml_map.get(hashlib.md5(emf_bytes).digest())
            if preferred:
                preferred_ext, preferred_bytes = preferred
            else:
                # No sibling in DOCX — convert via doc converter microservice
                png_bytes = _convert_emf_via_doc_converter(emf_bytes)
                if not png_bytes:
                    logger.debug("EMF could not be converted for file_id=%s; skipping", file_id)
                    return {"src": ""}
                preferred_ext, preferred_bytes = "png", png_bytes

            filename = f"image-{image_index:03d}.{preferred_ext}"
            (output_dir / filename).write_bytes(preferred_bytes)
            rel_path = f"docx_extracted_images/{file_id}/{filename}"
            return {"src": urljoin(media_url, rel_path)}

        filename = f"image-{image_index:03d}{extension}"
        image_path = output_dir / filename
        with image.open() as raw:
            image_path.write_bytes(raw.read())
        rel_path = f"docx_extracted_images/{file_id}/{filename}"
        return {"src": urljoin(media_url, rel_path)}

    try:
        import mammoth
        return mammoth.images.img_element(convert_image)
    except Exception:
        return None


def _extract_docx_with_mammoth_markdown(file_bytes: bytes, file_id: str) -> str:
    try:
        import mammoth
    except Exception as exc:
        logger.debug("mammoth unavailable for ragparser markdown extraction: %s", exc)
        return ""

    image_converter = _build_mammoth_image_converter(file_id=file_id, docx_bytes=file_bytes)
    options: dict[str, Any] = {}
    if image_converter is not None:
        options["convert_image"] = image_converter

    try:
        # Use Mammoth HTML output and keep complex tables as HTML. Pipe Markdown
        # tables cannot represent rowspan/colspan, so converting Word tables into
        # pipes breaks the structure before the frontend gets it.
        result = mammoth.convert_to_html(io.BytesIO(file_bytes), **options)
        markdown = _html_to_markdown_preserving_tables(result.value or "")
        markdown = _normalize_mammoth_markdown(markdown)
        if markdown.strip():
            return markdown
    except Exception as exc:
        logger.warning("mammoth markdown extraction failed for %s: %s", file_id, exc)
        return ""
    return ""


class _TablePreservingMarkdownConverter(MarkdownConverter):
    def convert_table(self, el, text, parent_tags):
        return "\n\n" + str(el).strip() + "\n\n"


def _html_to_markdown_preserving_tables(html_text: str) -> str:
    return _TablePreservingMarkdownConverter(
        heading_style="ATX",
        table_infer_header=False,
    ).convert(html_text or "")


def _build_content_with_images(raw_content: str, html_fragment: str) -> str:
    content = sanitize_content(raw_content)
    if not html_fragment:
        return content
    image_blocks = _extract_image_blocks(raw_content)
    if image_blocks:
        markdown_image = f"![{image_blocks[0][1] or 'Image'}]({image_blocks[0][0]})"
        if markdown_image in content or image_blocks[0][0] in content:
            return content
    image_src = None
    img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html_fragment, flags=re.IGNORECASE)
    if img_match:
        image_src = img_match.group(1)
    if not image_src:
        return content
    markdown_image = f"![Image]({image_src})"
    if markdown_image in content or image_src in content:
        return content
    return f"{content}\n\n{markdown_image}".strip() if content else markdown_image


def _normalize_mammoth_markdown(markdown: str) -> str:
    text = _clean_markdown(markdown)
    text = convert_tab_separated_to_markdown_table(text)
    text = fix_markdown_tables(text)
    return text


def _build_chunk_html(markdown_body: str, header_text: str, header_level: int) -> str:
    body = (markdown_body or "").strip()
    header = (header_text or "").strip()
    level = max(1, min(int(header_level or 1), 6))

    source_markdown = body
    if header:
        source_markdown = f"{'#' * level} {header}\n\n{body}" if body else f"{'#' * level} {header}"

    if not source_markdown.strip():
        return ""

    try:
        html = py_markdown.markdown(
            source_markdown,
            extensions=["tables", "sane_lists", "nl2br"],
            output_format="html5",
        )
        # Add CSS class so frontend can apply consistent table styling.
        html = re.sub(
            r"<table(?![^>]*\bclass=)([^>]*)>",
            r'<table class="generated-table"\1>',
            html,
            flags=re.IGNORECASE,
        )
        return html
    except Exception:
        logger.warning("Failed to render DOCX chunk HTML from markdown")
        return ""


def _get_cell_rich_text(cell) -> str:
    """Extract rich text from a DOCX table cell, preserving bullet/list structure."""
    from docx.oxml.ns import qn as _qn
    parts: list[str] = []
    for para in cell.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        is_list = False
        try:
            style_name = (para.style.name or "").lower() if para.style else ""
            is_list = "list" in style_name or "bullet" in style_name
            if not is_list and para._p.pPr is not None:
                is_list = para._p.pPr.find(_qn("w:numPr")) is not None
        except Exception:
            pass
        parts.append(f"• {text}" if is_list else text)
    cell_text = "<br>".join(parts)
    cell_text = re.sub(r"[ \t]{2,}", " ", cell_text)
    return cell_text.replace("|", "\\|")


def _is_vmerge_continuation(cell) -> bool:
    """Return True if this cell is a vertical-merge continuation (not the top cell)."""
    from docx.oxml.ns import qn as _qn
    try:
        tcp = cell._tc.tcPr
        if tcp is None:
            return False
        vm = tcp.find(_qn("w:vMerge"))
        if vm is None:
            return False
        return vm.get(_qn("w:val")) != "restart"
    except Exception:
        return False


def _has_markdown_table(content: str) -> bool:
    if not content:
        return False
    if re.search(r"<table\b", content, flags=re.IGNORECASE):
        return True
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    for idx in range(len(lines) - 1):
        if "|" not in lines[idx]:
            continue
        next_line = lines[idx + 1]
        if re.match(r"^\|?\s*:?-{3,}:?(?:\s*\|\s*:?-{3,}:?)*\s*\|?$", next_line):
            return True
    return False


def _iter_docx_block_items(doc: Any):
    """
    Yield paragraph/table blocks from the DOCX body in original document order.
    This keeps output structure close to MarkItDown formatting.
    """
    try:
        from docx.oxml.text.paragraph import CT_P
        from docx.oxml.table import CT_Tbl
        from docx.text.paragraph import Paragraph
        from docx.table import Table
    except Exception:
        return

    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def _to_alpha_counter(value: int, uppercase: bool = False) -> str:
    value = max(1, int(value))
    chars: list[str] = []
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        base = ord("A") if uppercase else ord("a")
        chars.append(chr(base + remainder))
    return "".join(reversed(chars))


def _to_roman_counter(value: int, uppercase: bool = False) -> str:
    value = max(1, int(value))
    symbols = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    result: list[str] = []
    current = value
    for amount, numeral in symbols:
        while current >= amount:
            result.append(numeral)
            current -= amount
    roman = "".join(result)
    return roman if uppercase else roman.lower()


def _format_number_token(num_fmt: str, value: int) -> str:
    fmt = (num_fmt or "").strip().lower()
    if fmt == "lowerletter":
        return _to_alpha_counter(value, uppercase=False)
    if fmt == "upperletter":
        return _to_alpha_counter(value, uppercase=True)
    if fmt == "lowerroman":
        return _to_roman_counter(value, uppercase=False)
    if fmt == "upperroman":
        return _to_roman_counter(value, uppercase=True)
    return str(max(1, int(value)))


def _read_num_pr(num_pr: Any) -> tuple[Optional[int], int]:
    if num_pr is None:
        return None, 0

    num_id: Optional[int] = None
    level = 0

    try:
        if getattr(num_pr, "numId", None) is not None and getattr(num_pr.numId, "val", None) is not None:
            num_id = int(num_pr.numId.val)
    except Exception:
        num_id = None

    try:
        if getattr(num_pr, "ilvl", None) is not None and getattr(num_pr.ilvl, "val", None) is not None:
            level = int(num_pr.ilvl.val)
    except Exception:
        level = 0

    return num_id, max(0, level)


def _extract_paragraph_numbering(paragraph: Any) -> tuple[Optional[int], int]:
    direct_num_id, direct_level = _read_num_pr(getattr(getattr(paragraph, "_p", None), "pPr", None) and getattr(paragraph._p.pPr, "numPr", None))
    if direct_num_id is not None:
        return direct_num_id, direct_level

    style = getattr(paragraph, "style", None)
    style_element = getattr(style, "element", None)
    style_ppr = getattr(style_element, "pPr", None)
    if style_ppr is None:
        return None, 0
    return _read_num_pr(getattr(style_ppr, "numPr", None))


def _build_numbering_format_map(doc: Any) -> dict[tuple[int, int], tuple[str, str]]:
    try:
        from docx.oxml.ns import qn
    except Exception:
        return {}

    format_map: dict[tuple[int, int], tuple[str, str]] = {}

    try:
        numbering_part = getattr(doc.part, "numbering_part", None)
        numbering_root = getattr(numbering_part, "_element", None)
        if numbering_root is None:
            return format_map

        num_to_abstract: dict[int, int] = {}
        for num_node in numbering_root.findall(qn("w:num")):
            num_id = num_node.get(qn("w:numId"))
            abstract_node = num_node.find(qn("w:abstractNumId"))
            abstract_id = abstract_node.get(qn("w:val")) if abstract_node is not None else None
            if num_id is None or abstract_id is None:
                continue
            num_to_abstract[int(num_id)] = int(abstract_id)

        abstract_level_map: dict[tuple[int, int], tuple[str, str]] = {}
        for abstract_node in numbering_root.findall(qn("w:abstractNum")):
            abstract_id = abstract_node.get(qn("w:abstractNumId"))
            if abstract_id is None:
                continue
            abstract_id_int = int(abstract_id)
            for level_node in abstract_node.findall(qn("w:lvl")):
                level_id = level_node.get(qn("w:ilvl"))
                if level_id is None:
                    continue
                num_fmt_node = level_node.find(qn("w:numFmt"))
                level_text_node = level_node.find(qn("w:lvlText"))
                num_fmt = num_fmt_node.get(qn("w:val")) if num_fmt_node is not None else ""
                level_text = level_text_node.get(qn("w:val")) if level_text_node is not None else ""
                abstract_level_map[(abstract_id_int, int(level_id))] = (num_fmt or "", level_text or "")

        for num_id, abstract_id in num_to_abstract.items():
            for (mapped_abstract_id, level), value in abstract_level_map.items():
                if mapped_abstract_id == abstract_id:
                    format_map[(num_id, level)] = value
    except Exception as exc:
        logger.debug("Unable to parse DOCX numbering definitions: %s", exc)

    return format_map


def _build_list_prefix(
    num_id: Optional[int],
    level: int,
    style_name: str,
    numbering_formats: dict[tuple[int, int], tuple[str, str]],
    numbering_counters: dict[tuple[int, int], int],
    style_counters: dict[tuple[str, int], int],
) -> tuple[str, int] | None:
    normalized_style = (style_name or "").strip().lower()
    style_level_match = re.search(r"list\s+(?:bullet|number|paragraph)\s*(\d+)$", normalized_style)
    style_level = max(int(style_level_match.group(1)) - 1, 0) if style_level_match else 0

    if num_id is None:
        if "list bullet" in normalized_style or "list paragraph" in normalized_style:
            return "•", style_level
        if "list number" in normalized_style:
            style_key = (normalized_style, style_level)
            style_counters[style_key] = style_counters.get(style_key, 0) + 1
            return f"{style_counters[style_key]}.", style_level
        return None

    for counter_key in list(numbering_counters.keys()):
        if counter_key[0] == num_id and counter_key[1] > level:
            numbering_counters.pop(counter_key, None)

    current_key = (num_id, level)
    numbering_counters[current_key] = numbering_counters.get(current_key, 0) + 1
    current_value = numbering_counters[current_key]

    num_fmt, lvl_text = numbering_formats.get((num_id, level), ("", ""))
    if (num_fmt or "").strip().lower() == "bullet":
        clean_marker = (lvl_text or "").strip()
        if clean_marker and "%" not in clean_marker:
            return clean_marker, level
        return "•", level

    if lvl_text and "%" in lvl_text:
        def replace_token(match: re.Match[str]) -> str:
            ref_level = max(int(match.group(1)) - 1, 0)
            ref_value = numbering_counters.get((num_id, ref_level), 1)
            ref_fmt, _ = numbering_formats.get((num_id, ref_level), (num_fmt, lvl_text))
            return _format_number_token(ref_fmt, ref_value)

        rendered = re.sub(r"%(\d+)", replace_token, lvl_text).strip()
        if rendered:
            return rendered, level

    return f"{_format_number_token(num_fmt, current_value)}.", level


def _extract_docx_with_python_docx(file_bytes: bytes) -> str:
    """
    Unicode-safe DOCX extraction using python-docx.
    Produces markdown-like text with headings, paragraphs, and pipe tables.
    Preserves paragraph/table ordering for MarkItDown-compatible formatting.
    """
    try:
        from docx import Document as DocxDocument
        from docx.text.paragraph import Paragraph
        from docx.table import Table
    except Exception as exc:
        logger.warning("python-docx unavailable for DOCX extraction: %s", exc)
        return ""

    doc = DocxDocument(io.BytesIO(file_bytes))
    lines: list[str] = []
    numbering_formats = _build_numbering_format_map(doc)
    numbering_counters: dict[tuple[int, int], int] = {}
    style_counters: dict[tuple[str, int], int] = {}
    previous_was_list = False

    for block in _iter_docx_block_items(doc):
        if isinstance(block, Paragraph):
            text = (block.text or "").strip()
            if not text:
                continue

            style_name = str(getattr(getattr(block, "style", None), "name", "") or "").lower().strip()
            heading_match = re.match(r"heading\s*(\d+)", style_name)

            if style_name == "title":
                if previous_was_list and (not lines or lines[-1] != ""):
                    lines.append("")
                lines.append(f"# {text}")
                lines.append("")
                previous_was_list = False
                continue

            if heading_match:
                if previous_was_list and (not lines or lines[-1] != ""):
                    lines.append("")
                level = max(1, min(6, int(heading_match.group(1))))
                lines.append(f"{'#' * level} {text}")
                lines.append("")
                previous_was_list = False
                continue

            num_id, list_level = _extract_paragraph_numbering(block)
            list_prefix = _build_list_prefix(
                num_id=num_id,
                level=list_level,
                style_name=style_name,
                numbering_formats=numbering_formats,
                numbering_counters=numbering_counters,
                style_counters=style_counters,
            )
            if list_prefix is not None:
                marker, marker_level = list_prefix
                indent = "  " * max(0, marker_level)
                lines.append(f"{indent}{marker} {text}".rstrip())
                previous_was_list = True
                continue

            if previous_was_list and (not lines or lines[-1] != ""):
                lines.append("")
            previous_was_list = False
            lines.append(text)
            lines.append("")
            continue

        if isinstance(block, Table):
            if previous_was_list and (not lines or lines[-1] != ""):
                lines.append("")
            previous_was_list = False
            matrix: list[list[str]] = []
            for row in block.rows:
                row_data: list[str] = []
                prev_tc = None
                for cell in row.cells:
                    tc = cell._tc
                    if tc is prev_tc:
                        # Horizontally merged cell — add empty placeholder
                        row_data.append("")
                    elif _is_vmerge_continuation(cell):
                        # Vertically merged continuation row — add empty placeholder
                        row_data.append("")
                    else:
                        row_data.append(_get_cell_rich_text(cell))
                    prev_tc = tc
                matrix.append(row_data)

            if not matrix:
                continue

            # Drop rows where every cell is empty (e.g. vertical-merge
            # continuation rows that carry no text).
            matrix = [r for r in matrix if any(c.strip() for c in r)]
            if not matrix:
                continue

            max_cols = max(len(row) for row in matrix)
            if max_cols == 0:
                continue

            normalized: list[list[str]] = []
            for row in matrix:
                if len(row) < max_cols:
                    row = row + [""] * (max_cols - len(row))
                elif len(row) > max_cols:
                    row = row[:max_cols]
                normalized.append(row)

            lines.append("| " + " | ".join(normalized[0]) + " |")
            lines.append("| " + " | ".join(["---"] * max_cols) + " |")
            for row in normalized[1:]:
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    if previous_was_list and (not lines or lines[-1] != ""):
        lines.append("")

    markdown_text = "\n".join(lines).strip()
    # Keep block spacing stable for downstream chunking and markitdown-style rendering.
    markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text)
    return markdown_text


def _replace_data_image_placeholders(content: str, inline_image_urls: list[str]) -> str:
    if not content or not inline_image_urls:
        return content

    pattern = re.compile(r"!\[[^\]]*\]\((data:image[^)]+)\)", re.IGNORECASE)
    urls_iter = iter(inline_image_urls)

    def replacement(match: re.Match[str]) -> str:
        try:
            replacement_url = next(urls_iter)
        except StopIteration:
            return match.group(0)
        alt_text_match = re.match(r"!\[([^\]]*)\]", match.group(0))
        alt_text = alt_text_match.group(1) if alt_text_match else ""
        return f"![{alt_text}]({replacement_url})"

    return pattern.sub(replacement, content)


def chunk_by_headers(markdown_text: str):
    """
    Split markdown text by headers into chunks.
    Preserves table formatting and newlines.
    """
    chunks = []
    current_header = None
    current_level = None
    current_content_lines = []
    preamble_lines = []
    current_page = 1
    current_page_origin = "initial"
    units_since_break = 0

    current_chunk_page_start = 1
    current_chunk_page_end = 1
    current_chunk_origins: set[str] = {"initial"}

    preamble_page_start = 1
    preamble_page_end = 1
    preamble_origins: set[str] = {"initial"}

    def resolve_page_meta(origins: set[str]) -> tuple[str, str]:
        normalized = set(origins or {"initial"})
        if "explicit" in normalized and "soft" in normalized:
            return "mixed-marker-soft", "medium"
        if "explicit" in normalized:
            return "explicit-marker", "high"
        if "soft" in normalized:
            return "soft-estimate", "low"
        return "initial-estimate", "medium"

    def apply_soft_page_break_if_needed(line: str) -> None:
        nonlocal current_page, current_page_origin, units_since_break
        line_units = _count_text_units(line)
        if line_units <= 0:
            return

        if (
            units_since_break > 0
            and units_since_break + line_units > DOCX_SOFT_PAGE_UNIT_BUDGET
        ):
            current_page += 1
            current_page_origin = "soft"
            units_since_break = 0

        units_since_break += line_units

    for line in markdown_text.splitlines():
        marker_match = DOCX_PAGE_MARKER_PATTERN.match(line)
        if marker_match:
            marker_page = max(1, int(marker_match.group(1)))
            current_page = marker_page
            current_page_origin = "explicit"
            units_since_break = 0
            continue

        apply_soft_page_break_if_needed(line)

        m = re.match(r'^(#{1,4})\s+(.*)', line)
        if m:
            if current_header is not None:
                # Join content preserving all newlines (important for tables)
                content = "\n".join(current_content_lines)
                page_method, page_confidence = resolve_page_meta(current_chunk_origins)
                chunks.append(
                    {
                        "header": current_header,
                        "level": current_level,
                        "content": content,
                        "has_table": _has_markdown_table(content),
                        "page_start": current_chunk_page_start,
                        "page_end": max(current_chunk_page_start, current_chunk_page_end),
                        "page_method": page_method,
                        "page_confidence": page_confidence,
                    }
                )
            elif preamble_lines:
                content = "\n".join(preamble_lines)
                page_method, page_confidence = resolve_page_meta(preamble_origins)
                chunks.append(
                    {
                        "header": "Content",
                        "level": 1,
                        "content": content,
                        "has_table": _has_markdown_table(content),
                        "page_start": preamble_page_start,
                        "page_end": max(preamble_page_start, preamble_page_end),
                        "page_method": page_method,
                        "page_confidence": page_confidence,
                    }
                )
            current_level = len(m.group(1))
            current_header = m.group(2).strip()
            current_content_lines = []
            current_chunk_page_start = current_page
            current_chunk_page_end = current_page
            current_chunk_origins = {current_page_origin}
        else:
            if current_header is None:
                preamble_lines.append(line)
                preamble_page_end = current_page
                preamble_origins.add(current_page_origin)
            else:
                current_content_lines.append(line)
                current_chunk_page_end = current_page
                current_chunk_origins.add(current_page_origin)

    # Process final chunk
    if current_header is not None:
        content = "\n".join(current_content_lines)
        page_method, page_confidence = resolve_page_meta(current_chunk_origins)
        chunks.append(
            {
                "header": current_header,
                "level": current_level,
                "content": content,
                "has_table": _has_markdown_table(content),
                "page_start": current_chunk_page_start,
                "page_end": max(current_chunk_page_start, current_chunk_page_end),
                "page_method": page_method,
                "page_confidence": page_confidence,
            }
        )
    elif preamble_lines:
        content = "\n".join(preamble_lines)
        page_method, page_confidence = resolve_page_meta(preamble_origins)
        chunks.append(
            {
                "header": "Content",
                "level": 1,
                "content": content,
                "has_table": _has_markdown_table(content),
                "page_start": preamble_page_start,
                "page_end": max(preamble_page_start, preamble_page_end),
                "page_method": page_method,
                "page_confidence": page_confidence,
            }
        )

    return chunks

def convert_tab_separated_to_markdown_table(text: str) -> str:
    """
    Convert tab-separated table format to proper markdown pipe table format.
    Handles cases like:
      Header1\tHeader2\tHeader3
      ---\t---\t---
      Data1\tData2\tData3
    Converts to:
      | Header1 | Header2 | Header3 |
      | --- | --- | --- |
      | Data1 | Data2 | Data3 |
    """
    lines = text.split("\n")
    result_lines: list[str] = []

    def format_row(cells: list[str]) -> str:
        # Escape literal pipes inside cell content to preserve table structure
        return "| " + " | ".join(cell.strip().replace("|", "\\|") for cell in cells) + " |"

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()

        if "\t" not in line or stripped_line.startswith("|"):
            result_lines.append(line)
            i += 1
            continue

        block: list[str] = []
        j = i
        while j < len(lines):
            candidate = lines[j]
            if "\t" in candidate and not candidate.strip().startswith("|"):
                block.append(candidate)
                j += 1
                continue
            break

        rows = [[cell.strip() for cell in row.split("\t")] for row in block]
        max_cols = max((len(row) for row in rows), default=0)

        if len(rows) >= 2 and max_cols >= 2:
            normalized_rows: list[list[str]] = []
            for row in rows:
                if len(row) < max_cols:
                    row = row + [""] * (max_cols - len(row))
                elif len(row) > max_cols:
                    row = row[:max_cols]
                normalized_rows.append(row)

            def is_separator_row(row_cells: list[str]) -> bool:
                return all(re.match(r"^:?-{3,}:?$", cell or "") for cell in row_cells)

            result_lines.append(format_row(normalized_rows[0]))
            if len(normalized_rows) > 1 and is_separator_row(normalized_rows[1]):
                result_lines.append(format_row(["---"] * max_cols))
                start_idx = 2
            else:
                result_lines.append(format_row(["---"] * max_cols))
                start_idx = 1

            for row in normalized_rows[start_idx:]:
                result_lines.append(format_row(row))
        else:
            result_lines.extend(block)

        i = j

    return "\n".join(result_lines)


def sanitize_content(text: str) -> str:
    """
    Sanitize content while preserving table structure.
    """
    if not text:
        return ""
    
    # Remove zero-width characters but preserve structure
    text = text.replace('\u200b', '').replace('\ufeff', '')
    # Replace non-breaking spaces with regular spaces
    text = text.replace('\xa0', ' ')
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Convert tab-separated tables to markdown pipe tables
    text = convert_tab_separated_to_markdown_table(text)
    
    # Fix table formatting (preserves newlines)
    text = fix_markdown_tables(text)
    
    return text


def fix_markdown_tables(text: str) -> str:
    """
    Fix markdown tables by ensuring proper formatting.
    Only fixes alignment issues, preserves original structure.
    """
    if not text or '|' not in text:
        return text
    
    lines = text.split("\n")
    result_lines: list[str] = []
    table_buffer: list[str] = []
    in_html_table = False

    def flush_table_buffer() -> None:
        nonlocal table_buffer
        if table_buffer:
            fixed_table = fix_table_alignment(table_buffer)
            result_lines.extend(fixed_table)
            table_buffer = []

    for line in lines:
        stripped = line.strip()
        if re.search(r"<table\b", stripped, flags=re.IGNORECASE):
            flush_table_buffer()
            in_html_table = True
            result_lines.append(line)
            if re.search(r"</table>", stripped, flags=re.IGNORECASE):
                in_html_table = False
            continue

        if in_html_table:
            result_lines.append(line)
            if re.search(r"</table>", stripped, flags=re.IGNORECASE):
                in_html_table = False
            continue

        if "|" in stripped and re.search(r"\|", stripped):
            table_buffer.append(line)
            continue

        flush_table_buffer()
        result_lines.append(line)

    flush_table_buffer()
    return "\n".join(result_lines)


def fix_table_alignment(table_lines: list) -> list:
    """
    Fix table alignment while preserving structure.
    Ensures all rows have consistent column count and proper | delimiters.
    """
    if not table_lines:
        return table_lines
    
    def parse_row(line: str) -> list[str]:
        stripped = line.strip()
        if not stripped:
            return []

        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]

        # Split on unescaped pipes only (preserve \| inside cell content)
        return [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped)]

    rows = [parse_row(line) for line in table_lines if line.strip()]
    rows = [row for row in rows if row]
    if not rows:
        return table_lines

    num_cols = max(len(row) for row in rows)
    if num_cols < 2:
        return table_lines

    def is_separator_row(row: list[str]) -> bool:
        return len(row) > 0 and all(
            re.match(r"^:?-{3,}:?$", cell.strip()) for cell in row if cell.strip() or cell == ""
        )

    normalized_rows: list[list[str]] = []
    for row in rows:
        if len(row) < num_cols:
            row = row + [""] * (num_cols - len(row))
        elif len(row) > num_cols:
            row = row[:num_cols]
        normalized_rows.append(row)

    output_rows: list[list[str]] = []
    output_rows.append(normalized_rows[0])

    if len(normalized_rows) > 1 and is_separator_row(normalized_rows[1]):
        output_rows.append(["---"] * num_cols)
        output_rows.extend(normalized_rows[2:])
    else:
        output_rows.append(["---"] * num_cols)
        output_rows.extend(normalized_rows[1:])

    return ["| " + " | ".join(cell.strip() for cell in row) + " |" for row in output_rows]


def _format_chunk_id(file_id: str, chunk_index: int) -> str:
    return f"{file_id}-{chunk_index + 1:04d}"


def _estimate_chunk_page_number(chunk_index: int, total_chunks: int, total_pages: int) -> int:
    if total_pages <= 0:
        return 1
    if total_chunks <= 1:
        return 1
    ratio = chunk_index / max(total_chunks - 1, 1)
    estimated = int(round(ratio * (total_pages - 1))) + 1
    return max(1, min(total_pages, estimated))


def _find_best_matching_page_for_chunk(
    markdown_image_path: Optional[str],
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    total_pages: int,
) -> int:
    if total_pages <= 0:
        return 1

    image_match = re.search(r"image-(\d{3})\.(?:png|jpg|jpeg|gif|webp|bmp|tif|tiff)$", markdown_image_path or "", re.IGNORECASE)
    if image_match:
        page_from_image = int(image_match.group(1))
        return max(1, min(total_pages, page_from_image))

    marker_match = re.search(r"Page\s+(\d+)", chunk_text or "", re.IGNORECASE)
    if marker_match:
        page_from_marker = int(marker_match.group(1))
        return max(1, min(total_pages, page_from_marker))

    return _estimate_chunk_page_number(chunk_index, total_chunks, total_pages)


def extract_ragparser_to_langchain_docs(file_bytes: bytes, file_id: str, file_name: str):
    """
    Extract DOCX content with ragparser-style markdown conversion and convert to LangChain documents.
    Uses a strict ragparser path (no parser fallback) to preserve deterministic behavior.
    Prefers python-docx for better Unicode support, falls back to Mammoth if needed.
    """
    try:
        # Prefer Mammoth because it preserves complex DOCX tables as HTML.
        # Fall back to python-docx only when Mammoth cannot extract content.
        markdown_text = _extract_docx_with_mammoth_markdown(file_bytes, str(file_id))

        if not markdown_text.strip():
            logger.debug("Mammoth extraction empty for %s, trying python-docx", file_name)
            markdown_text = _extract_docx_with_python_docx(file_bytes)
        
        if not markdown_text.strip():
            logger.warning("No text extracted from DOCX parser for %s", file_name)
            return []

        # Convert DOCX -> PDF and render page preview images into /media/.
        # Runs independently of text extraction; failures are non-fatal.
        pdf_bytes: bytes | None = None
        base_image_meta: dict[str, Any] = {}
        try:
            pdf_bytes = convert_docx_to_pdf(file_bytes, file_name)
            if pdf_bytes:
                _persist_pdf_page_images(pdf_bytes, str(file_id))
                base_image_meta = _build_image_link_metadata(str(file_id))
        except Exception as img_exc:
            logger.warning("Page image generation failed for %s: %s", file_name, img_exc)
            pdf_bytes = None
            base_image_meta = {}

        markdown_with_page_markers = _inject_page_markers_from_xml_offsets(markdown_text, file_bytes)
        chunks = chunk_by_headers(markdown_with_page_markers)

        documents = []
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            raw_content = chunk.get('content', '') or ''
            header_text = chunk.get('header', '')
            header_level = chunk.get('level', 1)
            has_table = bool(chunk.get("has_table", False))

            image_in_chunk = _extract_image_blocks(raw_content)
            image_url = image_in_chunk[0][0] if image_in_chunk else None

            page_start = int(chunk.get("page_start") or 1)
            page_end = int(chunk.get("page_end") or page_start)
            page_number = page_start
            # Only attach an image to chunk content when that chunk already contains
            # an inline DOCX image. Do not inject generic page preview images.
            html_fragment = _build_inline_image_html(image_url)

            content = _build_content_with_images(raw_content, html_fragment)
            html_content = _build_chunk_html(content, header_text, header_level)
            # Avoid duplicating the image: append html_fragment only when the
            # image URL is not already present in the converted HTML.
            if html_fragment and image_url and image_url not in html_content:
                html_content = f"{html_content}\n{html_fragment}".strip() if html_content else html_fragment
            detected_has_table = _has_markdown_table(raw_content) or _has_markdown_table(content)
            has_table = has_table or detected_has_table

            detected_lang = 'km' if any('\u1780' <= char <= '\u17FF' for char in content) else 'en'

            header_prefix = '#' * header_level
            if content:
                # Keep single newline after header to stay close to prior MarkItDown chunk shape.
                page_content_val = f"{header_prefix} {header_text}\n{content}"
            else:
                page_content_val = f"{header_prefix} {header_text}"

            page_img_meta = _build_image_link_metadata(str(file_id), page_number) if pdf_bytes else {}

            documents.append(LangChainDocument(
                page_content=page_content_val,
                metadata={
                    'source': file_id,
                    'chunk_id': _format_chunk_id(str(file_id), i),
                    'file_name': file_name,
                    'page_number': page_number,
                    'page_start': page_start,
                    'page_end': page_end,
                    'page_confidence': chunk.get("page_confidence", "medium"),
                    'page_method': chunk.get("page_method", "initial-estimate"),
                    'language': detected_lang,
                    'extraction_method': 'ragparser',
                    'header': header_text,
                    'header_level': header_level,
                    'chunk_index': i,
                    'has_table': has_table,
                    'content': content,
                    'html': html_content,
                    'page_image_url': page_img_meta.get("page_image_url", ""),
                    'page_image_path': page_img_meta.get("page_image_path", ""),
                    'image_manifest_url': base_image_meta.get("image_manifest_url", ""),
                    'image_manifest_path': base_image_meta.get("image_manifest_path", ""),
                }
            ))

        print(f"Successfully extracted DOCX {file_name}: {len(documents)} chunks using ragparser")
        return documents

    except Exception as e:
        print(f"Error in extract_ragparser_to_langchain_docs for {file_name}: {str(e)}")
        logging.exception("Detailed error in DOCX extraction")
        raise


def extract_word_to_langchain_docs(file_bytes: bytes, file_id: str, file_name: str):
    """Extract Word document to LangChain documents using the strict RAG parser path."""
    return extract_ragparser_to_langchain_docs(file_bytes, file_id, file_name)


def save_extraction(docx_path: str, output_md_path: str | None = None):
    try:
        with open(docx_path, 'rb') as f:
            file_bytes = f.read()

        documents = extract_ragparser_to_langchain_docs(file_bytes, os.path.basename(docx_path), os.path.basename(docx_path))

        combined_content = ""
        for i, doc in enumerate(documents):
            if i > 0:
                combined_content += "\n\n"
            combined_content += doc.page_content

        if output_md_path is None:
            output_md_path = docx_path.replace('.docx', '.md')

        os.makedirs(os.path.dirname(output_md_path), exist_ok=True) if os.path.dirname(output_md_path) else None
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(combined_content)

        print("Extraction test completed")
        print(f"Input DOCX : {docx_path}")
        print(f"Output MD  : {output_md_path}")
        print(f"Characters: {len(combined_content)}")
        print(f"Words     : {len(combined_content.split())}")
        print(f"Chunks    : {len(documents)}")

    except Exception as e:
        print(f"Error in save_extraction: {str(e)}")
        raise

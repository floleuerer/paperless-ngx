import dataclasses
import json
import logging
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from django.conf import settings

from documents.parsers import DocumentParser
from documents.parsers import make_thumbnail_from_pdf
from documents.parsers import run_convert
from documents.utils import run_subprocess
from paperless.config import LlmOcrConfig
from paperless.models import ArchiveFileChoices

logger = logging.getLogger("paperless.parsing.llm")

VALID_TEXT_LENGTH = 50


@dataclasses.dataclass
class HocrWord:
    bbox: tuple[int, int, int, int]  # left, top, right, bottom
    text: str


@dataclasses.dataclass
class HocrLine:
    index: int
    bbox: tuple[int, int, int, int]
    text: str
    words: list[HocrWord]


@dataclasses.dataclass
class PageOcrData:
    text: str
    hocr: str | None = None  # Pre-aligned hOCR, None = use approximate


def _post_process_text(text: str | None) -> str | None:
    """Basic whitespace cleanup, mirrors paperless_tesseract post_process_text."""
    if not text:
        return None
    collapsed_spaces = re.sub(r"([^\S\r\n]+)", " ", text)
    no_leading_whitespace = re.sub(r"([\n\r]+)([^\S\n\r]+)", "\\1", collapsed_spaces)
    no_trailing_whitespace = re.sub(r"([^\S\n\r]+)$", "", no_leading_whitespace)
    return no_trailing_whitespace.strip().replace("\0", " ") or None


class LlmDocumentParser(DocumentParser):
    """
    Parser that uses vision-capable LLMs (OpenAI, Gemini, Ollama) to
    extract text from PDFs and images.
    """

    logging_name = "paperless.parsing.llm"

    # MIME types natively supported by most vision LLMs
    NATIVE_IMAGE_TYPES = frozenset(
        {
            "image/png",
            "image/jpeg",
            "image/webp",
            # "image/heic",
            # "image/heif",
        },
    )

    def get_settings(self) -> LlmOcrConfig:
        return LlmOcrConfig()

    def supported_mime_types(self):
        if self.settings.is_enabled():
            return {
                "application/pdf": ".pdf",
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/webp": ".webp",
                "image/heic": ".heic",
                "image/heif": ".heif",
                "image/tiff": ".tiff",
                "image/bmp": ".bmp",
                "image/gif": ".gif",
            }
        return {}

    def get_multi_modal_llm(self):
        backend = self.settings.llm_ocr_backend

        if backend == "openai":
            from llama_index.llms.openai import OpenAI

            return OpenAI(
                model=self.settings.llm_ocr_model or "gpt-5-mini",
                api_key=self.settings.llm_ocr_api_key,
                api_base=self.settings.llm_ocr_endpoint or None,
            )
        elif backend == "gemini":
            from llama_index.llms.google_genai import GoogleGenAI

            return GoogleGenAI(
                model=self.settings.llm_ocr_model or "models/gemini-3-flash-preview",
                api_key=self.settings.llm_ocr_api_key,
            )
        elif backend == "ollama":
            from llama_index.llms.ollama import Ollama

            return Ollama(
                model=self.settings.llm_ocr_model or "llava",
                base_url=self.settings.llm_ocr_endpoint or "http://localhost:11434",
            )
        else:
            raise ValueError(f"Unsupported LLM OCR backend: {backend}")

    # ------------------------------------------------------------------
    # Text extraction helpers
    # ------------------------------------------------------------------

    def _ocr_image(self, image_path: Path, mime_type: str) -> str:
        """Send a single image to the multi-modal LLM for OCR."""
        from llama_index.core.llms import ChatMessage
        from llama_index.core.llms import ImageBlock
        from llama_index.core.llms import TextBlock

        llm = self.get_multi_modal_llm()

        prompt = (
            "Extract all text from this document image as plain text. "
            "You must adhere to the following strict requirements:\n"
            "1. Fidelity: The content must be EXACTLY as it appears in the document. Do not summarize, "
            "rephrase, or correct perceived typos.\n"
            "2. Language: Keep the original language of the document. Do not translate.\n"
            "3. Structure: Preserve the original layout and flow of the text. "
            "Use line breaks to separate paragraphs and sections as they appear in the original.\n"
            "4. Tables: Represent tables using aligned plain text columns.\n"
            "5. Lists: Preserve numbered and bulleted lists as they appear.\n\n"
            "Return ONLY the extracted plain text. Do not use any markup or formatting syntax "
            "(no Markdown, no HTML). Do not include code block delimiters or any introductory/explanatory text."
        )
        messages = [
            ChatMessage(
                role="user",
                blocks=[
                    ImageBlock(path=str(image_path), image_mimetype=mime_type),
                    TextBlock(text=prompt),
                ],
            ),
        ]
        response = llm.chat(messages, temperature=0.0)
        return str(response.message.content).strip()

    def _extract_text_pdftotext(self, pdf_path: Path) -> str | None:
        """Try to extract existing text from a PDF using pdftotext."""
        if not pdf_path.is_file():
            return None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+",
                dir=self.tempdir,
            ) as tmp:
                run_subprocess(
                    [
                        "pdftotext",
                        "-q",
                        "-layout",
                        "-enc",
                        "UTF-8",
                        str(pdf_path),
                        tmp.name,
                    ],
                    logger=self.log,
                )
                text = self.read_file_handle_unicode_errors(Path(tmp.name))
            return _post_process_text(text)
        except Exception:
            self.log.debug(
                "pdftotext extraction failed for %s",
                pdf_path,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _is_image(self, mime_type: str) -> bool:
        return mime_type.startswith("image/")

    def _prepare_image(self, image_path: Path, mime_type: str) -> tuple[Path, str]:
        """
        If the image format is not natively supported by vision LLMs,
        convert it to PNG. Returns (path, mime_type) to use.
        """
        if mime_type in self.NATIVE_IMAGE_TYPES:
            return image_path, mime_type

        self.log.info(
            "Converting %s to PNG for LLM OCR",
            mime_type,
        )
        from PIL import Image

        converted_path = self.tempdir / f"{image_path.stem}_converted.png"
        with Image.open(image_path) as img:
            img.save(str(converted_path), "PNG")
        return converted_path, "image/png"

    def _get_image_page_count(self, image_path: Path) -> int:
        """Return the number of frames/pages in an image (>1 for multi-page
        TIFF, animated GIF, etc.)."""
        from PIL import Image

        try:
            with Image.open(image_path) as img:
                return getattr(img, "n_frames", 1)
        except Exception:
            return 1

    def _extract_image_pages(self, image_path: Path) -> list[Path]:
        """Extract individual pages from a multi-page image (e.g. TIFF).
        Returns a list of PNG paths, one per page."""
        from PIL import Image

        pages: list[Path] = []
        with Image.open(image_path) as img:
            n_frames = getattr(img, "n_frames", 1)
            self.log.info(
                "Multi-page image has %d page(s), extracting frames",
                n_frames,
            )
            for i in range(n_frames):
                img.seek(i)
                page_path = self.tempdir / f"{image_path.stem}_frame_{i}.png"
                img.save(str(page_path), "PNG")
                pages.append(page_path)
        return pages

    def _get_image_dpi(self, image_path: Path) -> int:
        """Return image DPI or a sensible default."""
        try:
            from PIL import Image

            with Image.open(image_path) as im:
                x, _ = im.info["dpi"]
                return max(round(x), 72)
        except Exception:
            return 300

    # ------------------------------------------------------------------
    # Tesseract + LLM alignment for layout-aware hOCR
    # ------------------------------------------------------------------

    def _run_tesseract_hocr(self, image_path: Path) -> str | None:
        """Run Tesseract on the image to get hOCR with accurate bounding boxes.
        Returns the hOCR string or None if Tesseract is unavailable or fails."""
        self.log.debug(
            "Running Tesseract hOCR on %s for layout alignment",
            image_path.name,
        )
        try:
            output_prefix = self.tempdir / f"tess_{image_path.stem}"
            lang = settings.OCR_LANGUAGE
            run_subprocess(
                [
                    "tesseract",
                    str(image_path),
                    str(output_prefix),
                    "-l",
                    lang,
                    "hocr",
                ],
                logger=self.log,
            )
            hocr_file = Path(f"{output_prefix}.hocr")
            if hocr_file.exists():
                hocr_content = hocr_file.read_text(encoding="utf-8")
                self.log.debug(
                    "Tesseract hOCR generated successfully (%d bytes)",
                    len(hocr_content),
                )
                return hocr_content
            self.log.debug("Tesseract hOCR output file not found: %s", hocr_file)
            return None
        except FileNotFoundError:
            self.log.info(
                "Tesseract not found on system, skipping layout alignment "
                "(archive PDF will use approximate text positions)",
            )
            return None
        except Exception:
            self.log.warning(
                "Tesseract hOCR generation failed for %s",
                image_path,
                exc_info=True,
            )
            return None

    def _parse_hocr_lines(
        self,
        hocr: str,
    ) -> tuple[int, int, list[HocrLine]]:
        """Parse hOCR XML and extract page dimensions and line/word data."""
        self.log.debug("Parsing Tesseract hOCR XML (%d bytes)", len(hocr))
        root = ET.fromstring(hocr)
        ns = {"xhtml": "http://www.w3.org/1999/xhtml"}

        # Find page element — try with and without namespace
        page_el = root.find(".//*[@class='ocr_page']")
        if page_el is None:
            page_el = root.find(".//xhtml:*[@class='ocr_page']", ns)
        if page_el is None:
            self.log.debug("No ocr_page element found in hOCR")
            return 0, 0, []

        page_title = page_el.get("title", "")
        page_w, page_h = 0, 0
        bbox_match = re.search(r"bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", page_title)
        if bbox_match:
            page_w = int(bbox_match.group(3))
            page_h = int(bbox_match.group(4))

        lines: list[HocrLine] = []
        line_index = 0

        # Search for line elements in the tree
        for el in root.iter():
            el_class = el.get("class", "")
            if "ocr_line" not in el_class and "ocr_header" not in el_class:
                continue

            title = el.get("title", "")
            bbox_match = re.search(
                r"bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
                title,
            )
            if not bbox_match:
                continue

            line_bbox = (
                int(bbox_match.group(1)),
                int(bbox_match.group(2)),
                int(bbox_match.group(3)),
                int(bbox_match.group(4)),
            )

            words: list[HocrWord] = []
            for word_el in el.iter():
                word_class = word_el.get("class", "")
                if "ocrx_word" not in word_class:
                    continue
                word_title = word_el.get("title", "")
                wbbox_match = re.search(
                    r"bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
                    word_title,
                )
                if not wbbox_match:
                    continue
                word_text = (word_el.text or "") + "".join(
                    (child.text or "") + (child.tail or "") for child in word_el
                )
                word_text = word_text.strip()
                if word_text:
                    words.append(
                        HocrWord(
                            bbox=(
                                int(wbbox_match.group(1)),
                                int(wbbox_match.group(2)),
                                int(wbbox_match.group(3)),
                                int(wbbox_match.group(4)),
                            ),
                            text=word_text,
                        ),
                    )

            line_text = " ".join(w.text for w in words)
            if line_text:
                lines.append(
                    HocrLine(
                        index=line_index,
                        bbox=line_bbox,
                        text=line_text,
                        words=words,
                    ),
                )
                line_index += 1

        self.log.debug(
            "Parsed hOCR: page=%dx%d, %d lines, %d total words",
            page_w,
            page_h,
            len(lines),
            sum(len(line.words) for line in lines),
        )
        return page_w, page_h, lines

    def _align_with_llm(
        self,
        image_path: Path,
        mime_type: str,
        tesseract_lines: list[HocrLine],
        llm_text: str,
    ) -> tuple[dict[int, str], list[str]]:
        """Use the LLM to align high-quality OCR text to Tesseract bounding boxes.
        Returns (line_index → corrected_text, extra_lines)."""
        from llama_index.core.llms import ChatMessage
        from llama_index.core.llms import ImageBlock
        from llama_index.core.llms import TextBlock

        self.log.debug(
            "Starting LLM alignment: %d Tesseract lines, %d chars LLM text",
            len(tesseract_lines),
            len(llm_text),
        )

        tess_listing = "\n".join(
            f'[{line.index}] "{line.text}"' for line in tesseract_lines
        )

        prompt = (
            "You are aligning two OCR results for the same document page "
            "shown in the image.\n\n"
            "TESSERACT OCR (has accurate text positions but may have "
            "text recognition errors):\n"
            f"{tess_listing}\n\n"
            "HIGH-QUALITY OCR (accurate text, but no position information):\n"
            f"{llm_text}\n\n"
            "For each numbered TESSERACT line, provide the corrected text "
            "using the HIGH-QUALITY OCR.\n"
            "Use the document image to verify which text corresponds to "
            "which line.\n"
            "If HIGH-QUALITY OCR contains text not present in any TESSERACT "
            'line, include it in "extra".\n\n'
            "Return ONLY valid JSON (no markdown, no code blocks):\n"
            '{"lines": {"0": "corrected", "1": "corrected", ...}, '
            '"extra": ["any unmatched text"]}'
        )

        try:
            llm = self.get_multi_modal_llm()
            messages = [
                ChatMessage(
                    role="user",
                    blocks=[
                        ImageBlock(
                            path=str(image_path),
                            image_mimetype=mime_type,
                        ),
                        TextBlock(text=prompt),
                    ],
                ),
            ]
            response = llm.chat(messages, temperature=0.0)
            raw = str(response.message.content).strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)
            corrections: dict[int, str] = {}
            for k, v in data.get("lines", {}).items():
                corrections[int(k)] = str(v)
            extra = [str(e) for e in data.get("extra", [])]
            self.log.debug(
                "LLM alignment successful: %d/%d lines corrected, %d extra lines",
                len(corrections),
                len(tesseract_lines),
                len(extra),
            )
            return corrections, extra
        except json.JSONDecodeError:
            self.log.warning(
                "LLM alignment returned invalid JSON (len=%d), "
                "falling back to approximate hOCR. Response: %.200s",
                len(raw),
                raw,
            )
            return {}, []
        except Exception:
            self.log.warning(
                "LLM alignment failed, falling back to approximate hOCR",
                exc_info=True,
            )
            return {}, []

    def _rebuild_hocr(
        self,
        page_w: int,
        page_h: int,
        lines: list[HocrLine],
        corrections: dict[int, str],
        extra: list[str],
    ) -> str:
        """Rebuild hOCR XML using Tesseract bboxes with LLM-corrected text."""
        import html as html_mod

        self.log.debug(
            "Rebuilding hOCR: page=%dx%d, %d lines, %d corrections, %d extra",
            page_w,
            page_h,
            len(lines),
            len(corrections),
            len(extra),
        )

        line_elements = []
        for line in lines:
            corrected = corrections.get(line.index, line.text)
            escaped = html_mod.escape(corrected)
            words = escaped.split()
            if not words:
                continue

            lx0, ly0, lx1, ly1 = line.bbox
            line_w = max(lx1 - lx0, 1)

            # Distribute word bboxes proportionally by character count,
            # including spaces so that gaps appear between word boxes.
            total_chars = max(
                sum(len(w) for w in words) + max(len(words) - 1, 0),
                1,
            )
            word_spans = []
            char_offset = 0
            for word in words:
                wx0 = lx0 + (char_offset * line_w) // total_chars
                wx1 = lx0 + ((char_offset + len(word)) * line_w) // total_chars
                word_spans.append(
                    f'<span class="ocrx_word" '
                    f'title="bbox {wx0} {ly0} {wx1} {ly1}">{word}</span>',
                )
                char_offset += len(word) + 1  # +1 for inter-word space

            line_elements.append(
                f'<span class="ocr_line" '
                f'title="bbox {lx0} {ly0} {lx1} {ly1}">'
                + " ".join(word_spans)
                + "</span>",
            )

        # Append extra text lines at the bottom of the page
        if extra:
            extra_y = page_h - len(extra) * 20  # rough estimate
            for extra_line in extra:
                if not extra_line.strip():
                    continue
                escaped = html_mod.escape(extra_line)
                words = escaped.split()
                if not words:
                    continue
                ey0 = max(extra_y, 0)
                ey1 = min(extra_y + 18, page_h)
                total_chars = max(
                    sum(len(w) for w in words) + max(len(words) - 1, 0),
                    1,
                )
                word_spans = []
                char_offset = 0
                for word in words:
                    wx0 = (char_offset * page_w) // total_chars
                    wx1 = ((char_offset + len(word)) * page_w) // total_chars
                    word_spans.append(
                        f'<span class="ocrx_word" '
                        f'title="bbox {wx0} {ey0} {wx1} {ey1}">{word}</span>',
                    )
                    char_offset += len(word) + 1  # +1 for inter-word space
                line_elements.append(
                    f'<span class="ocr_line" '
                    f'title="bbox 0 {ey0} {page_w} {ey1}">'
                    + " ".join(word_spans)
                    + "</span>",
                )
                extra_y += 20

        body = "\n".join(line_elements)

        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<!DOCTYPE html PUBLIC "
            '"-//W3C//DTD XHTML 1.0 Transitional//EN"\n'
            ' "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'
            '<html xmlns="http://www.w3.org/1999/xhtml">\n'
            "<head>\n"
            ' <meta charset="utf-8"/>\n'
            ' <meta name="ocr-system" content="paperless-llm-aligned"/>\n'
            "</head>\n"
            "<body>\n"
            f'<div class="ocr_page" title="bbox 0 0 {page_w} {page_h}; ppageno 0">\n'
            f'<div class="ocr_carea" title="bbox 0 0 {page_w} {page_h}">\n'
            f'<p class="ocr_par" title="bbox 0 0 {page_w} {page_h}">\n'
            f"{body}\n"
            "</p>\n</div>\n</div>\n"
            "</body>\n</html>"
        )

    def _generate_aligned_hocr(
        self,
        image_path: Path,
        mime_type: str,
        llm_text: str,
    ) -> str | None:
        """Orchestrate Tesseract + LLM alignment for one page.
        Returns aligned hOCR string, or None on failure (triggers fallback)."""
        self.log.info(
            "Attempting layout-aware hOCR alignment for %s",
            image_path.name,
        )
        try:
            hocr_raw = self._run_tesseract_hocr(image_path)
            if hocr_raw is None:
                self.log.info(
                    "Tesseract unavailable — will use approximate text positions",
                )
                return None

            page_w, page_h, lines = self._parse_hocr_lines(hocr_raw)
            if not lines:
                self.log.info(
                    "No text lines found by Tesseract — will use approximate "
                    "text positions",
                )
                return None

            corrections, extra = self._align_with_llm(
                image_path,
                mime_type,
                lines,
                llm_text,
            )
            if not corrections:
                self.log.info(
                    "LLM alignment returned no corrections — will use "
                    "approximate text positions",
                )
                return None

            result = self._rebuild_hocr(
                page_w,
                page_h,
                lines,
                corrections,
                extra,
            )
            self.log.info(
                "Layout-aware hOCR alignment completed successfully (%d bytes hOCR)",
                len(result),
            )
            return result
        except Exception:
            self.log.warning(
                "Aligned hOCR generation failed — will use approximate text positions",
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Archive PDF creation
    # ------------------------------------------------------------------

    def _create_archive_pdf(
        self,
        document_path: Path,
        mime_type: str,
        page_data_list: list[PageOcrData],
    ) -> Path:
        """Create a searchable PDF/A archive using ocrmypdf with our LLM
        text injected via a custom OCR engine plugin."""
        import ocrmypdf

        from paperless_llm.ocrmypdf_engine import page_data_store

        archive_path = Path(self.tempdir) / "archive.pdf"

        # Load the pre-computed page data into the shared store so the engine
        # can return them page-by-page.
        page_data_store.set_pages(page_data_list)

        ocrmypdf_args = {
            "input_file": str(document_path),
            "output_file": str(archive_path),
            "use_threads": True,
            "jobs": 1,
            "force_ocr": True,
            "output_type": settings.OCR_OUTPUT_TYPE,
            "progress_bar": False,
            "plugins": ["paperless_llm.ocrmypdf_engine"],
        }

        if self._is_image(mime_type):
            ocrmypdf_args["image_dpi"] = self._get_image_dpi(document_path)

        self.log.debug("Creating archive PDF via ocrmypdf: %s", ocrmypdf_args)
        ocrmypdf.ocr(**ocrmypdf_args)
        return archive_path

    # ------------------------------------------------------------------
    # Main parse entry point
    # ------------------------------------------------------------------

    def parse(self, document_path: Path, mime_type, file_name=None):
        if not self.settings.is_enabled():
            self.log.warning(
                "LLM OCR is not enabled, content will be empty.",
            )
            self.text = ""
            return

        skip_archive_file = settings.OCR_SKIP_ARCHIVE_FILE

        # ----- For PDFs, check if text already exists -----
        text_original = None
        original_has_text = False

        if mime_type == "application/pdf":
            text_original = self._extract_text_pdftotext(document_path)
            original_has_text = (
                text_original is not None and len(text_original) > VALID_TEXT_LENGTH
            )

        # If PDF already has text and we should skip the archive, we are done.
        skip_archive_for_text = skip_archive_file in {
            ArchiveFileChoices.WITH_TEXT,
            ArchiveFileChoices.ALWAYS,
        }
        if skip_archive_for_text and original_has_text:
            self.log.debug(
                "PDF already has extractable text and archive creation is "
                "skipped — using existing text without LLM OCR.",
            )
            self.text = text_original
            return

        # If PDF has text we still use it (saves LLM API calls) but create
        # an archive from the original PDF so paperless has one on file.
        if original_has_text:
            self.log.info(
                "PDF already has extractable text — using it directly.",
            )
            self.text = text_original
            if skip_archive_file != ArchiveFileChoices.ALWAYS:
                archive_path = Path(self.tempdir) / "archive.pdf"
                shutil.copy2(document_path, archive_path)
                self.archive_path = archive_path
            return

        # ----- No usable text — proceed with LLM OCR -----
        self.log.info(
            "Using LLM OCR backend '%s' with model '%s'",
            self.settings.llm_ocr_backend,
            self.settings.llm_ocr_model,
        )

        # --- Step 1: Extract text via LLM OCR ---
        page_data_list: list[PageOcrData] | None = None
        try:
            if self._is_image(mime_type):
                page_data_list = self._parse_image_pages(
                    document_path,
                    mime_type,
                )
                self.text = "\n\n".join(p.text for p in page_data_list)
            elif mime_type == "application/pdf":
                page_data_list = self._parse_pdf_pages(document_path)
                self.text = "\n\n".join(p.text for p in page_data_list)
            else:
                self.log.warning(
                    "Unsupported mime type for LLM OCR: %s",
                    mime_type,
                )
                self.text = ""
                return
        except Exception as e:
            self.log.exception("LLM OCR text extraction failed: %s", e)
            if original_has_text:
                self.text = text_original
            else:
                self.text = ""
            return

        self.log.debug(
            "LLM OCR extracted %d chars of text from %d page(s)",
            len(self.text),
            len(page_data_list),
        )

        # --- Step 2: Create searchable archive PDF ---
        # This is separate so that a failure here does not wipe self.text.
        if skip_archive_file != ArchiveFileChoices.ALWAYS:
            try:
                self.archive_path = self._create_archive_pdf(
                    document_path,
                    mime_type,
                    page_data_list,
                )
                self.log.debug(
                    "Created archive PDF at %s",
                    self.archive_path,
                )
            except Exception as e:
                self.log.exception(
                    "Archive PDF creation failed (text is preserved): %s",
                    e,
                )

        # Final safeguard: self.text must never be None
        if self.text is None:
            self.log.warning(
                "self.text is unexpectedly None after parse — setting to empty",
            )
            self.text = ""

    def _parse_image_pages(
        self,
        document_path: Path,
        mime_type: str,
    ) -> list[PageOcrData]:
        """Process an image file (possibly multi-page TIFF/GIF).
        Returns list of per-page PageOcrData."""
        n_pages = self._get_image_page_count(document_path)

        if n_pages > 1:
            # Multi-page image (e.g. TIFF)
            self.log.info(
                "Multi-page %s with %d pages, processing each page",
                mime_type,
                n_pages,
            )
            frame_paths = self._extract_image_pages(document_path)
            results: list[PageOcrData] = []
            for i, frame_path in enumerate(frame_paths):
                self.log.debug(
                    "Processing image page %d/%d",
                    i + 1,
                    len(frame_paths),
                )
                page_text = self._ocr_image(frame_path, mime_type="image/png")
                self.log.debug(
                    "Image page %d LLM OCR returned %d chars",
                    i + 1,
                    len(page_text),
                )
                aligned_hocr = self._generate_aligned_hocr(
                    frame_path,
                    "image/png",
                    page_text,
                )
                results.append(PageOcrData(text=page_text, hocr=aligned_hocr))
                self.log.debug(
                    "Image page %d hOCR: %s",
                    i + 1,
                    "aligned" if aligned_hocr else "approximate (fallback)",
                )
                self.progress(i + 1, len(frame_paths))
            return results

        # Single-page image
        prepared_path, prepared_mime = self._prepare_image(
            document_path,
            mime_type,
        )
        page_text = self._ocr_image(prepared_path, mime_type=prepared_mime)
        self.log.debug("Image LLM OCR returned %d chars", len(page_text))
        aligned_hocr = self._generate_aligned_hocr(
            prepared_path,
            prepared_mime,
            page_text,
        )
        self.log.info(
            "Image hOCR: %s",
            "aligned" if aligned_hocr else "approximate (fallback)",
        )
        return [PageOcrData(text=page_text, hocr=aligned_hocr)]

    def _parse_pdf_pages(self, document_path: Path) -> list[PageOcrData]:
        """Convert PDF pages to images and OCR each page.
        Returns the list of per-page PageOcrData."""
        from pdf2image import convert_from_path

        pages = convert_from_path(str(document_path), dpi=200)
        self.log.info("PDF has %d page(s), processing with LLM OCR", len(pages))

        results: list[PageOcrData] = []
        for i, page_image in enumerate(pages):
            image_path = self.tempdir / f"page_{i}.png"
            page_image.save(str(image_path), "PNG")
            self.log.debug("Processing page %d/%d", i + 1, len(pages))
            page_text = self._ocr_image(image_path, mime_type="image/png")
            self.log.debug(
                "Page %d LLM OCR returned %d chars",
                i + 1,
                len(page_text),
            )
            aligned_hocr = self._generate_aligned_hocr(
                image_path,
                "image/png",
                page_text,
            )
            results.append(PageOcrData(text=page_text, hocr=aligned_hocr))
            self.log.debug(
                "Page %d hOCR: %s",
                i + 1,
                "aligned" if aligned_hocr else "approximate (fallback)",
            )
            self.progress(i + 1, len(pages))

        aligned_count = sum(1 for r in results if r.hocr is not None)
        self.log.info(
            "PDF processing complete: %d pages, %d with aligned hOCR, "
            "%d with approximate hOCR",
            len(results),
            aligned_count,
            len(results) - aligned_count,
        )
        return results

    # ------------------------------------------------------------------
    # Thumbnail / metadata
    # ------------------------------------------------------------------

    def get_thumbnail(self, document_path, mime_type, file_name=None):
        if mime_type == "application/pdf":
            return make_thumbnail_from_pdf(
                self.archive_path or document_path,
                self.tempdir,
                self.logging_group,
            )
        # For images, convert to thumbnail
        out_path = self.tempdir / "convert.webp"
        run_convert(
            density=300,
            scale="500x5000>",
            alpha="remove",
            strip=True,
            trim=False,
            auto_orient=True,
            input_file=str(document_path),
            output_file=str(out_path),
            logging_group=self.logging_group,
        )
        return out_path

    def get_page_count(self, document_path, mime_type):
        if mime_type == "application/pdf":
            try:
                import pikepdf

                with pikepdf.open(document_path) as pdf:
                    return len(pdf.pages)
            except Exception as e:
                self.log.warning("Could not determine page count: %s", e)
                return None
        return 1

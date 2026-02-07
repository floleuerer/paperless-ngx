import logging
import re
import shutil
import tempfile
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
    NATIVE_IMAGE_TYPES = frozenset({
        "image/png",
        "image/jpeg",
        "image/webp",
        #"image/heic",
        #"image/heif",
    })

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
                model=self.settings.llm_ocr_model or "gpt-4o",
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
        from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock

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
            )
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
    # Archive PDF creation
    # ------------------------------------------------------------------

    def _create_archive_pdf(
        self,
        document_path: Path,
        mime_type: str,
        page_texts: list[str],
    ) -> Path:
        """Create a searchable PDF/A archive using ocrmypdf with our LLM
        text injected via a custom OCR engine plugin."""
        import ocrmypdf

        from paperless_llm.ocrmypdf_engine import page_text_store

        archive_path = Path(self.tempdir) / "archive.pdf"

        # Load the pre-computed texts into the shared store so the engine
        # can return them page-by-page.
        page_text_store.set_texts(page_texts)

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
                text_original is not None
                and len(text_original) > VALID_TEXT_LENGTH
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

        try:
            if self._is_image(mime_type):
                prepared_path, prepared_mime = self._prepare_image(
                    document_path,
                    mime_type,
                )
                page_text = self._ocr_image(prepared_path, mime_type=prepared_mime)
                page_texts = [page_text]
                self.text = page_text
            elif mime_type == "application/pdf":
                page_texts = self._parse_pdf_pages(document_path)
                self.text = "\n\n".join(page_texts)
            else:
                self.log.warning(
                    "Unsupported mime type for LLM OCR: %s",
                    mime_type,
                )
                self.text = ""
                return

            # Create searchable archive unless configured to always skip
            if skip_archive_file != ArchiveFileChoices.ALWAYS:
                self.archive_path = self._create_archive_pdf(
                    document_path,
                    mime_type,
                    page_texts,
                )
                self.log.debug("Created archive PDF at %s", self.archive_path)
        except Exception as e:
            self.log.error("LLM OCR parsing failed: %s", e, exc_info=True)
            # Fall back to any text we got from the original PDF
            if original_has_text:
                self.text = text_original
            else:
                self.text = ""

    def _parse_pdf_pages(self, document_path: Path) -> list[str]:
        """Convert PDF pages to images and OCR each page.
        Returns the list of per-page texts."""
        from pdf2image import convert_from_path

        pages = convert_from_path(str(document_path), dpi=300)
        self.log.info("PDF has %d page(s), processing with LLM OCR", len(pages))

        texts = []
        for i, page_image in enumerate(pages):
            image_path = self.tempdir / f"page_{i}.png"
            page_image.save(str(image_path), "PNG")
            self.log.debug("Processing page %d/%d", i + 1, len(pages))
            page_text = self._ocr_image(image_path, mime_type="image/png")
            texts.append(page_text)
            self.progress(i + 1, len(pages))

        return texts

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

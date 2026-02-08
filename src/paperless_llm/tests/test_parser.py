import uuid
from pathlib import Path
from unittest import mock

from django.test import TestCase
from django.test import override_settings

from documents.tests.utils import DirectoriesMixin
from documents.tests.utils import FileSystemAssertsMixin
from paperless_llm.parsers import VALID_TEXT_LENGTH
from paperless_llm.parsers import HocrLine
from paperless_llm.parsers import HocrWord
from paperless_llm.parsers import LlmDocumentParser
from paperless_llm.parsers import PageOcrData
from paperless_llm.parsers import _post_process_text
from paperless_llm.signals import get_parser

SAMPLE_DIR = (
    Path(__file__).resolve().parent.parent.parent / "documents" / "tests" / "samples"
)

OPENAI_SETTINGS = {
    "LLM_OCR_ENABLED": True,
    "LLM_OCR_BACKEND": "openai",
    "LLM_OCR_MODEL": "gpt-4o",
    "LLM_OCR_API_KEY": "sk-test",
    "LLM_OCR_ENDPOINT": None,
}


class TestPostProcessText(TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(_post_process_text(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_post_process_text(""))

    def test_whitespace_only_returns_none(self):
        self.assertIsNone(_post_process_text("   \n  \t  "))

    def test_collapses_multiple_spaces(self):
        self.assertEqual(_post_process_text("hello   world"), "hello world")

    def test_strips_leading_whitespace_on_lines(self):
        self.assertEqual(_post_process_text("line1\n  line2"), "line1\nline2")

    def test_strips_trailing_whitespace(self):
        self.assertEqual(_post_process_text("hello   "), "hello")

    def test_replaces_null_characters(self):
        self.assertEqual(_post_process_text("hello\0world"), "hello world")

    def test_preserves_line_breaks(self):
        result = _post_process_text("line1\nline2\nline3")
        self.assertEqual(result, "line1\nline2\nline3")

    def test_complex_cleanup(self):
        text = "  hello   world  \n   foo  bar  \n\n  baz  "
        result = _post_process_text(text)
        self.assertEqual(result, "hello world\nfoo bar\n\nbaz")


class TestLlmParserMimeTypes(DirectoriesMixin, FileSystemAssertsMixin, TestCase):
    @override_settings(LLM_OCR_ENABLED=False, LLM_OCR_BACKEND=None)
    def test_supported_mime_types_disabled(self):
        parser = LlmDocumentParser(uuid.uuid4())
        self.assertEqual(parser.supported_mime_types(), {})

    @override_settings(**OPENAI_SETTINGS)
    def test_supported_mime_types_enabled(self):
        parser = LlmDocumentParser(uuid.uuid4())
        types = parser.supported_mime_types()
        self.assertEqual(len(types), 9)
        self.assertIn("application/pdf", types)
        self.assertIn("image/png", types)
        self.assertIn("image/jpeg", types)
        self.assertIn("image/webp", types)
        self.assertIn("image/heic", types)
        self.assertIn("image/heif", types)
        self.assertIn("image/tiff", types)
        self.assertIn("image/bmp", types)
        self.assertIn("image/gif", types)


class TestLlmParserHelpers(DirectoriesMixin, FileSystemAssertsMixin, TestCase):
    @override_settings(**OPENAI_SETTINGS)
    def test_is_image_true(self):
        parser = LlmDocumentParser(uuid.uuid4())
        for mime in ("image/png", "image/jpeg", "image/tiff", "image/webp"):
            self.assertTrue(parser._is_image(mime), f"Expected True for {mime}")

    @override_settings(**OPENAI_SETTINGS)
    def test_is_image_false(self):
        parser = LlmDocumentParser(uuid.uuid4())
        self.assertFalse(parser._is_image("application/pdf"))
        self.assertFalse(parser._is_image("text/plain"))

    @override_settings(**OPENAI_SETTINGS)
    def test_prepare_image_native_no_conversion(self):
        parser = LlmDocumentParser(uuid.uuid4())
        original = SAMPLE_DIR / "simple.png"
        result_path, result_mime = parser._prepare_image(original, "image/png")
        self.assertEqual(result_path, original)
        self.assertEqual(result_mime, "image/png")

    @override_settings(**OPENAI_SETTINGS)
    def test_prepare_image_converts_tiff_to_png(self):
        parser = LlmDocumentParser(uuid.uuid4())
        result_path, result_mime = parser._prepare_image(
            SAMPLE_DIR / "simple.tiff",
            "image/tiff",
        )
        self.assertTrue(str(result_path).endswith("_converted.png"))
        self.assertEqual(result_mime, "image/png")
        self.assertTrue(result_path.exists())

    @override_settings(**OPENAI_SETTINGS)
    def test_get_image_dpi_nonexistent_returns_default(self):
        parser = LlmDocumentParser(uuid.uuid4())
        dpi = parser._get_image_dpi(Path("/nonexistent/file.png"))
        self.assertEqual(dpi, 300)

    @override_settings(**OPENAI_SETTINGS)
    @mock.patch("paperless_llm.parsers.run_subprocess")
    def test_extract_text_pdftotext_success(self, mock_run):
        parser = LlmDocumentParser(uuid.uuid4())
        with mock.patch.object(
            parser,
            "read_file_handle_unicode_errors",
            return_value="  Hello   World  ",
        ):
            result = parser._extract_text_pdftotext(SAMPLE_DIR / "simple.pdf")
        mock_run.assert_called_once()
        self.assertEqual(result, "Hello World")

    @override_settings(**OPENAI_SETTINGS)
    def test_extract_text_pdftotext_nonexistent_file(self):
        parser = LlmDocumentParser(uuid.uuid4())
        result = parser._extract_text_pdftotext(Path("/nonexistent.pdf"))
        self.assertIsNone(result)

    @override_settings(**OPENAI_SETTINGS)
    @mock.patch(
        "paperless_llm.parsers.run_subprocess",
        side_effect=Exception("pdftotext failed"),
    )
    def test_extract_text_pdftotext_error(self, mock_run):
        parser = LlmDocumentParser(uuid.uuid4())
        result = parser._extract_text_pdftotext(SAMPLE_DIR / "simple.pdf")
        self.assertIsNone(result)

    @override_settings(**OPENAI_SETTINGS)
    @mock.patch("paperless_llm.parsers.LlmDocumentParser.get_multi_modal_llm")
    def test_ocr_image(self, mock_get_llm):
        mock_llm = mock.Mock()
        mock_response = mock.Mock()
        mock_response.message.content = "  Extracted text  "
        mock_llm.chat.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        parser = LlmDocumentParser(uuid.uuid4())
        result = parser._ocr_image(SAMPLE_DIR / "simple.png", "image/png")

        self.assertEqual(result, "Extracted text")
        mock_llm.chat.assert_called_once()
        call_kwargs = mock_llm.chat.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.0)


class TestLlmParserParse(DirectoriesMixin, FileSystemAssertsMixin, TestCase):
    @override_settings(LLM_OCR_ENABLED=False, LLM_OCR_BACKEND=None)
    def test_parse_disabled(self):
        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.pdf", "application/pdf")
        self.assertEqual(parser.text, "")

    @override_settings(**OPENAI_SETTINGS, OCR_SKIP_ARCHIVE_FILE="never")
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._create_archive_pdf")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._generate_aligned_hocr",
        return_value=None,
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    @mock.patch("pdf2image.convert_from_path")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._extract_text_pdftotext",
    )
    def test_parse_pdf(
        self,
        mock_extract,
        mock_convert,
        mock_ocr,
        mock_aligned,
        mock_archive,
    ):
        mock_extract.return_value = None
        mock_page1, mock_page2 = mock.Mock(), mock.Mock()
        mock_convert.return_value = [mock_page1, mock_page2]
        mock_ocr.side_effect = ["Page 1 text", "Page 2 text"]
        mock_archive.return_value = Path("/tmp/archive.pdf")

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.pdf", "application/pdf")

        self.assertEqual(parser.text, "Page 1 text\n\nPage 2 text")
        self.assertEqual(mock_ocr.call_count, 2)
        mock_page1.save.assert_called_once()
        mock_page2.save.assert_called_once()
        mock_archive.assert_called_once()

    @override_settings(**OPENAI_SETTINGS, OCR_SKIP_ARCHIVE_FILE="never")
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._create_archive_pdf")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._generate_aligned_hocr",
        return_value=None,
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    def test_parse_image_png(self, mock_ocr, mock_aligned, mock_archive):
        mock_ocr.return_value = "Extracted text"
        mock_archive.return_value = Path("/tmp/archive.pdf")

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.png", "image/png")

        self.assertEqual(parser.text, "Extracted text")
        mock_ocr.assert_called_once()
        mock_archive.assert_called_once()

    @override_settings(**OPENAI_SETTINGS, OCR_SKIP_ARCHIVE_FILE="never")
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._create_archive_pdf")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._generate_aligned_hocr",
        return_value=None,
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    def test_parse_image_jpeg(self, mock_ocr, mock_aligned, mock_archive):
        mock_ocr.return_value = "JPEG text"
        mock_archive.return_value = Path("/tmp/archive.pdf")

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.jpg", "image/jpeg")

        self.assertEqual(parser.text, "JPEG text")
        call_args = mock_ocr.call_args
        self.assertEqual(call_args[1]["mime_type"], "image/jpeg")

    @override_settings(**OPENAI_SETTINGS, OCR_SKIP_ARCHIVE_FILE="never")
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._create_archive_pdf")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._generate_aligned_hocr",
        return_value=None,
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    def test_parse_tiff_converts_to_png(self, mock_ocr, mock_aligned, mock_archive):
        mock_ocr.return_value = "Text from tiff"
        mock_archive.return_value = Path("/tmp/archive.pdf")

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.tiff", "image/tiff")

        self.assertEqual(parser.text, "Text from tiff")
        call_args = mock_ocr.call_args
        self.assertTrue(str(call_args[0][0]).endswith("_converted.png"))
        self.assertEqual(call_args[1]["mime_type"], "image/png")

    @override_settings(**OPENAI_SETTINGS, OCR_SKIP_ARCHIVE_FILE="never")
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._create_archive_pdf")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._generate_aligned_hocr",
        return_value=None,
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    def test_parse_native_image_no_conversion(
        self,
        mock_ocr,
        mock_aligned,
        mock_archive,
    ):
        mock_ocr.return_value = "Text from png"
        mock_archive.return_value = Path("/tmp/archive.pdf")

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.png", "image/png")

        self.assertEqual(parser.text, "Text from png")
        call_args = mock_ocr.call_args
        self.assertEqual(str(call_args[0][0]), str(SAMPLE_DIR / "simple.png"))
        self.assertEqual(call_args[1]["mime_type"], "image/png")

    @override_settings(**OPENAI_SETTINGS)
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    def test_parse_error_handled_empty_fallback(self, mock_ocr):
        mock_ocr.side_effect = RuntimeError("API error")

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.png", "image/png")

        self.assertEqual(parser.text, "")

    @override_settings(**OPENAI_SETTINGS, OCR_SKIP_ARCHIVE_FILE="never")
    def test_parse_unsupported_mime_type(self):
        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.txt", "text/plain")

        self.assertEqual(parser.text, "")

    # --- PDF with existing text ---

    @override_settings(**OPENAI_SETTINGS, OCR_SKIP_ARCHIVE_FILE="with_text")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._extract_text_pdftotext",
    )
    def test_parse_pdf_existing_text_skip_with_text(self, mock_extract):
        existing = "A" * (VALID_TEXT_LENGTH + 1)
        mock_extract.return_value = existing

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.pdf", "application/pdf")

        self.assertEqual(parser.text, existing)
        self.assertIsNone(parser.archive_path)

    @override_settings(**OPENAI_SETTINGS, OCR_SKIP_ARCHIVE_FILE="always")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._extract_text_pdftotext",
    )
    def test_parse_pdf_existing_text_skip_always(self, mock_extract):
        existing = "A" * (VALID_TEXT_LENGTH + 1)
        mock_extract.return_value = existing

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.pdf", "application/pdf")

        self.assertEqual(parser.text, existing)
        self.assertIsNone(parser.archive_path)

    @override_settings(**OPENAI_SETTINGS, OCR_SKIP_ARCHIVE_FILE="never")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._extract_text_pdftotext",
    )
    def test_parse_pdf_existing_text_skip_never_copies_archive(self, mock_extract):
        existing = "A" * (VALID_TEXT_LENGTH + 1)
        mock_extract.return_value = existing

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.pdf", "application/pdf")

        self.assertEqual(parser.text, existing)
        self.assertIsNotNone(parser.archive_path)
        self.assertTrue(parser.archive_path.exists())

    @override_settings(**OPENAI_SETTINGS, OCR_SKIP_ARCHIVE_FILE="never")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._extract_text_pdftotext",
    )
    def test_parse_pdf_short_text_uses_llm(self, mock_extract):
        mock_extract.return_value = "Short"  # < VALID_TEXT_LENGTH

        with (
            mock.patch(
                "paperless_llm.parsers.LlmDocumentParser._create_archive_pdf",
            ) as mock_archive,
            mock.patch(
                "paperless_llm.parsers.LlmDocumentParser._generate_aligned_hocr",
                return_value=None,
            ),
            mock.patch("pdf2image.convert_from_path") as mock_convert,
            mock.patch(
                "paperless_llm.parsers.LlmDocumentParser._ocr_image",
            ) as mock_ocr,
        ):
            mock_page = mock.Mock()
            mock_convert.return_value = [mock_page]
            mock_ocr.return_value = "Full OCR text"
            mock_archive.return_value = Path("/tmp/archive.pdf")

            parser = get_parser(uuid.uuid4())
            parser.parse(SAMPLE_DIR / "simple.pdf", "application/pdf")

            self.assertEqual(parser.text, "Full OCR text")
            mock_ocr.assert_called_once()

    @override_settings(**OPENAI_SETTINGS, OCR_SKIP_ARCHIVE_FILE="always")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._generate_aligned_hocr",
        return_value=None,
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    def test_parse_image_skip_archive_always(self, mock_ocr, mock_aligned):
        mock_ocr.return_value = "Image text"

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.png", "image/png")

        self.assertEqual(parser.text, "Image text")
        self.assertIsNone(parser.archive_path)

    # --- _parse_pdf_pages ---

    @override_settings(**OPENAI_SETTINGS)
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._generate_aligned_hocr",
        return_value=None,
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    @mock.patch("pdf2image.convert_from_path")
    def test_parse_pdf_pages_calls_progress(
        self,
        mock_convert,
        mock_ocr,
        mock_aligned,
    ):
        mock_pages = [mock.Mock(), mock.Mock(), mock.Mock()]
        mock_convert.return_value = mock_pages
        mock_ocr.side_effect = ["Text 1", "Text 2", "Text 3"]

        parser = LlmDocumentParser(uuid.uuid4())
        with mock.patch.object(parser, "progress") as mock_progress:
            results = parser._parse_pdf_pages(SAMPLE_DIR / "simple.pdf")

        self.assertEqual(len(results), 3)
        self.assertEqual([r.text for r in results], ["Text 1", "Text 2", "Text 3"])
        for r in results:
            self.assertIsNone(r.hocr)
        self.assertEqual(mock_progress.call_count, 3)
        mock_progress.assert_any_call(1, 3)
        mock_progress.assert_any_call(2, 3)
        mock_progress.assert_any_call(3, 3)

    # --- _create_archive_pdf ---

    @override_settings(**OPENAI_SETTINGS, OCR_OUTPUT_TYPE="pdfa")
    @mock.patch("ocrmypdf.ocr")
    @mock.patch("paperless_llm.ocrmypdf_engine.page_data_store")
    def test_create_archive_pdf_for_pdf(self, mock_store, mock_ocr):
        parser = LlmDocumentParser(uuid.uuid4())
        page_data_list = [
            PageOcrData(text="Page 1"),
            PageOcrData(text="Page 2"),
        ]

        parser._create_archive_pdf(
            SAMPLE_DIR / "simple.pdf",
            "application/pdf",
            page_data_list,
        )

        mock_store.set_pages.assert_called_once_with(page_data_list)
        mock_ocr.assert_called_once()
        call_kwargs = mock_ocr.call_args[1]
        self.assertEqual(call_kwargs["output_type"], "pdfa")
        self.assertTrue(call_kwargs["force_ocr"])
        self.assertIn("paperless_llm.ocrmypdf_engine", call_kwargs["plugins"])
        self.assertNotIn("image_dpi", call_kwargs)

    @override_settings(**OPENAI_SETTINGS, OCR_OUTPUT_TYPE="pdfa")
    @mock.patch("ocrmypdf.ocr")
    @mock.patch("paperless_llm.ocrmypdf_engine.page_data_store")
    def test_create_archive_pdf_for_image_includes_dpi(self, mock_store, mock_ocr):
        parser = LlmDocumentParser(uuid.uuid4())

        parser._create_archive_pdf(
            SAMPLE_DIR / "simple.png",
            "image/png",
            [PageOcrData(text="text")],
        )

        call_kwargs = mock_ocr.call_args[1]
        self.assertIn("image_dpi", call_kwargs)


class TestLlmParserBackends(DirectoriesMixin, FileSystemAssertsMixin, TestCase):
    @override_settings(**OPENAI_SETTINGS)
    def test_get_multi_modal_llm_openai(self):
        with mock.patch("llama_index.llms.openai.OpenAI") as mock_cls:
            parser = LlmDocumentParser(uuid.uuid4())
            parser.get_multi_modal_llm()
            mock_cls.assert_called_once_with(
                model="gpt-4o",
                api_key="sk-test",
                api_base=None,
            )

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="openai",
        LLM_OCR_MODEL=None,
        LLM_OCR_API_KEY="sk-test",
        LLM_OCR_ENDPOINT="https://custom.api/v1",
    )
    def test_get_multi_modal_llm_openai_defaults_and_custom_endpoint(self):
        with mock.patch("llama_index.llms.openai.OpenAI") as mock_cls:
            parser = LlmDocumentParser(uuid.uuid4())
            parser.get_multi_modal_llm()
            mock_cls.assert_called_once_with(
                model="gpt-5-mini",
                api_key="sk-test",
                api_base="https://custom.api/v1",
            )

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="gemini",
        LLM_OCR_MODEL="models/gemini-2.0-flash",
        LLM_OCR_API_KEY="test-key",
        LLM_OCR_ENDPOINT=None,
    )
    def test_get_multi_modal_llm_gemini(self):
        with mock.patch(
            "llama_index.llms.google_genai.GoogleGenAI",
        ) as mock_cls:
            parser = LlmDocumentParser(uuid.uuid4())
            parser.get_multi_modal_llm()
            mock_cls.assert_called_once_with(
                model="models/gemini-2.0-flash",
                api_key="test-key",
            )

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="gemini",
        LLM_OCR_MODEL=None,
        LLM_OCR_API_KEY="test-key",
        LLM_OCR_ENDPOINT=None,
    )
    def test_get_multi_modal_llm_gemini_default_model(self):
        with mock.patch(
            "llama_index.llms.google_genai.GoogleGenAI",
        ) as mock_cls:
            parser = LlmDocumentParser(uuid.uuid4())
            parser.get_multi_modal_llm()
            mock_cls.assert_called_once_with(
                model="models/gemini-3-flash-preview",
                api_key="test-key",
            )

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="ollama",
        LLM_OCR_MODEL="llava",
        LLM_OCR_API_KEY=None,
        LLM_OCR_ENDPOINT="http://localhost:11434",
    )
    def test_get_multi_modal_llm_ollama(self):
        with mock.patch("llama_index.llms.ollama.Ollama") as mock_cls:
            parser = LlmDocumentParser(uuid.uuid4())
            parser.get_multi_modal_llm()
            mock_cls.assert_called_once_with(
                model="llava",
                base_url="http://localhost:11434",
            )

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="ollama",
        LLM_OCR_MODEL=None,
        LLM_OCR_API_KEY=None,
        LLM_OCR_ENDPOINT=None,
    )
    def test_get_multi_modal_llm_ollama_defaults(self):
        with mock.patch("llama_index.llms.ollama.Ollama") as mock_cls:
            parser = LlmDocumentParser(uuid.uuid4())
            parser.get_multi_modal_llm()
            mock_cls.assert_called_once_with(
                model="llava",
                base_url="http://localhost:11434",
            )

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="invalid",
        LLM_OCR_MODEL=None,
        LLM_OCR_API_KEY=None,
        LLM_OCR_ENDPOINT=None,
    )
    def test_get_multi_modal_llm_unsupported(self):
        parser = LlmDocumentParser(uuid.uuid4())
        with self.assertRaises(ValueError):
            parser.get_multi_modal_llm()


class TestLlmParserMetadata(DirectoriesMixin, FileSystemAssertsMixin, TestCase):
    @override_settings(**OPENAI_SETTINGS)
    def test_get_page_count_pdf(self):
        parser = LlmDocumentParser(uuid.uuid4())
        count = parser.get_page_count(SAMPLE_DIR / "simple.pdf", "application/pdf")
        self.assertIsNotNone(count)
        self.assertGreaterEqual(count, 1)

    @override_settings(**OPENAI_SETTINGS)
    def test_get_page_count_image(self):
        parser = LlmDocumentParser(uuid.uuid4())
        count = parser.get_page_count(SAMPLE_DIR / "simple.png", "image/png")
        self.assertEqual(count, 1)

    @override_settings(**OPENAI_SETTINGS)
    def test_get_page_count_pdf_error(self):
        parser = LlmDocumentParser(uuid.uuid4())
        count = parser.get_page_count(Path("/nonexistent.pdf"), "application/pdf")
        self.assertIsNone(count)

    @override_settings(**OPENAI_SETTINGS)
    @mock.patch("paperless_llm.parsers.make_thumbnail_from_pdf")
    def test_get_thumbnail_pdf(self, mock_thumb):
        expected = Path("/tmp/thumb.webp")
        mock_thumb.return_value = expected

        parser = LlmDocumentParser(uuid.uuid4())
        result = parser.get_thumbnail(
            SAMPLE_DIR / "simple.pdf",
            "application/pdf",
        )

        self.assertEqual(result, expected)
        mock_thumb.assert_called_once()

    @override_settings(**OPENAI_SETTINGS)
    @mock.patch("paperless_llm.parsers.make_thumbnail_from_pdf")
    def test_get_thumbnail_pdf_uses_archive_if_available(self, mock_thumb):
        expected = Path("/tmp/thumb.webp")
        mock_thumb.return_value = expected

        parser = LlmDocumentParser(uuid.uuid4())
        parser.archive_path = Path("/tmp/archive.pdf")
        result = parser.get_thumbnail(
            SAMPLE_DIR / "simple.pdf",
            "application/pdf",
        )

        self.assertEqual(result, expected)
        args = mock_thumb.call_args[0]
        self.assertEqual(args[0], Path("/tmp/archive.pdf"))

    @override_settings(**OPENAI_SETTINGS)
    @mock.patch("paperless_llm.parsers.run_convert")
    def test_get_thumbnail_image(self, mock_convert):
        parser = LlmDocumentParser(uuid.uuid4())
        result = parser.get_thumbnail(SAMPLE_DIR / "simple.png", "image/png")

        self.assertTrue(str(result).endswith("convert.webp"))
        mock_convert.assert_called_once()
        call_kwargs = mock_convert.call_args[1]
        self.assertEqual(call_kwargs["density"], 300)
        self.assertEqual(call_kwargs["scale"], "500x5000>")


class TestLlmSignals(DirectoriesMixin, FileSystemAssertsMixin, TestCase):
    @override_settings(**OPENAI_SETTINGS)
    def test_signal_declaration(self):
        from paperless_llm.signals import llm_consumer_declaration

        result = llm_consumer_declaration(None)
        self.assertEqual(result["weight"], 10)
        self.assertIn("parser", result)
        self.assertIn("mime_types", result)
        self.assertTrue(callable(result["parser"]))

    @override_settings(**OPENAI_SETTINGS)
    def test_signal_parser_factory(self):
        from paperless_llm.signals import llm_consumer_declaration

        result = llm_consumer_declaration(None)
        parser = result["parser"](uuid.uuid4())
        self.assertIsInstance(parser, LlmDocumentParser)

    @override_settings(**OPENAI_SETTINGS)
    def test_signal_mime_types_match_parser(self):
        from paperless_llm.signals import llm_consumer_declaration

        result = llm_consumer_declaration(None)
        parser = LlmDocumentParser(uuid.uuid4())
        self.assertEqual(result["mime_types"], parser.supported_mime_types())


class TestTesseractHocr(DirectoriesMixin, FileSystemAssertsMixin, TestCase):
    @override_settings(**OPENAI_SETTINGS, OCR_LANGUAGE="eng")
    @mock.patch("paperless_llm.parsers.run_subprocess")
    def test_run_tesseract_hocr_success(self, mock_run):
        parser = LlmDocumentParser(uuid.uuid4())
        hocr_content = '<html><body><div class="ocr_page">test</div></body></html>'

        def write_hocr(cmd, **kwargs):
            # Tesseract writes <prefix>.hocr
            output_prefix = cmd[2]
            Path(f"{output_prefix}.hocr").write_text(
                hocr_content,
                encoding="utf-8",
            )

        mock_run.side_effect = write_hocr
        result = parser._run_tesseract_hocr(SAMPLE_DIR / "simple.png")

        self.assertEqual(result, hocr_content)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "tesseract")
        self.assertIn("-l", cmd)
        self.assertIn("eng", cmd)
        self.assertIn("hocr", cmd)

    @override_settings(**OPENAI_SETTINGS, OCR_LANGUAGE="eng")
    @mock.patch(
        "paperless_llm.parsers.run_subprocess",
        side_effect=FileNotFoundError("tesseract not found"),
    )
    def test_run_tesseract_hocr_not_installed(self, mock_run):
        parser = LlmDocumentParser(uuid.uuid4())
        result = parser._run_tesseract_hocr(SAMPLE_DIR / "simple.png")
        self.assertIsNone(result)

    @override_settings(**OPENAI_SETTINGS, OCR_LANGUAGE="eng")
    @mock.patch(
        "paperless_llm.parsers.run_subprocess",
        side_effect=Exception("tesseract crashed"),
    )
    def test_run_tesseract_hocr_error(self, mock_run):
        parser = LlmDocumentParser(uuid.uuid4())
        result = parser._run_tesseract_hocr(SAMPLE_DIR / "simple.png")
        self.assertIsNone(result)


class TestParseHocrLines(DirectoriesMixin, FileSystemAssertsMixin, TestCase):
    SAMPLE_HOCR = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
        "<body>\n"
        '<div class="ocr_page" title="bbox 0 0 2550 3300; ppageno 0">\n'
        '<div class="ocr_carea" title="bbox 100 100 2450 3200">\n'
        '<p class="ocr_par" title="bbox 100 100 2450 3200">\n'
        '<span class="ocr_line" title="bbox 100 100 2000 150">\n'
        '<span class="ocrx_word" title="bbox 100 100 300 150">Hello</span>\n'
        '<span class="ocrx_word" title="bbox 350 100 600 150">World</span>\n'
        "</span>\n"
        '<span class="ocr_line" title="bbox 100 200 2000 250">\n'
        '<span class="ocrx_word" title="bbox 100 200 400 250">Second</span>\n'
        '<span class="ocrx_word" title="bbox 450 200 600 250">line</span>\n'
        "</span>\n"
        "</p>\n</div>\n</div>\n"
        "</body>\n</html>"
    )

    @override_settings(**OPENAI_SETTINGS)
    def test_parse_hocr_lines(self):
        parser = LlmDocumentParser(uuid.uuid4())
        page_w, page_h, lines = parser._parse_hocr_lines(self.SAMPLE_HOCR)

        self.assertEqual(page_w, 2550)
        self.assertEqual(page_h, 3300)
        self.assertEqual(len(lines), 2)

        self.assertEqual(lines[0].index, 0)
        self.assertEqual(lines[0].bbox, (100, 100, 2000, 150))
        self.assertEqual(lines[0].text, "Hello World")
        self.assertEqual(len(lines[0].words), 2)
        self.assertEqual(lines[0].words[0].text, "Hello")
        self.assertEqual(lines[0].words[0].bbox, (100, 100, 300, 150))

        self.assertEqual(lines[1].index, 1)
        self.assertEqual(lines[1].bbox, (100, 200, 2000, 250))
        self.assertEqual(lines[1].text, "Second line")

    @override_settings(**OPENAI_SETTINGS)
    def test_parse_hocr_lines_empty(self):
        empty_hocr = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml">\n'
            "<body>\n"
            '<div class="ocr_page" title="bbox 0 0 200 100">\n'
            "</div>\n"
            "</body>\n</html>"
        )
        parser = LlmDocumentParser(uuid.uuid4())
        page_w, page_h, lines = parser._parse_hocr_lines(empty_hocr)
        self.assertEqual(page_w, 200)
        self.assertEqual(page_h, 100)
        self.assertEqual(lines, [])


class TestAlignWithLlm(DirectoriesMixin, FileSystemAssertsMixin, TestCase):
    @override_settings(**OPENAI_SETTINGS)
    @mock.patch("paperless_llm.parsers.LlmDocumentParser.get_multi_modal_llm")
    def test_align_with_llm_success(self, mock_get_llm):
        mock_llm = mock.Mock()
        mock_response = mock.Mock()
        mock_response.message.content = (
            '{"lines": {"0": "Hello World", "1": "Second line"}, "extra": []}'
        )
        mock_llm.chat.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        parser = LlmDocumentParser(uuid.uuid4())
        lines = [
            HocrLine(
                index=0,
                bbox=(100, 100, 2000, 150),
                text="He1lo Wor1d",
                words=[
                    HocrWord(bbox=(100, 100, 300, 150), text="He1lo"),
                    HocrWord(bbox=(350, 100, 600, 150), text="Wor1d"),
                ],
            ),
            HocrLine(
                index=1,
                bbox=(100, 200, 2000, 250),
                text="Sec0nd l1ne",
                words=[
                    HocrWord(bbox=(100, 200, 400, 250), text="Sec0nd"),
                    HocrWord(bbox=(450, 200, 600, 250), text="l1ne"),
                ],
            ),
        ]
        corrections, extra = parser._align_with_llm(
            SAMPLE_DIR / "simple.png",
            "image/png",
            lines,
            "Hello World\nSecond line",
        )

        self.assertEqual(corrections, {0: "Hello World", 1: "Second line"})
        self.assertEqual(extra, [])
        mock_llm.chat.assert_called_once()

    @override_settings(**OPENAI_SETTINGS)
    @mock.patch("paperless_llm.parsers.LlmDocumentParser.get_multi_modal_llm")
    def test_align_with_llm_with_extra(self, mock_get_llm):
        mock_llm = mock.Mock()
        mock_response = mock.Mock()
        mock_response.message.content = (
            '{"lines": {"0": "Hello"}, "extra": ["Footer text"]}'
        )
        mock_llm.chat.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        parser = LlmDocumentParser(uuid.uuid4())
        lines = [
            HocrLine(
                index=0,
                bbox=(100, 100, 500, 150),
                text="He1lo",
                words=[HocrWord(bbox=(100, 100, 500, 150), text="He1lo")],
            ),
        ]
        corrections, extra = parser._align_with_llm(
            SAMPLE_DIR / "simple.png",
            "image/png",
            lines,
            "Hello\nFooter text",
        )
        self.assertEqual(corrections, {0: "Hello"})
        self.assertEqual(extra, ["Footer text"])

    @override_settings(**OPENAI_SETTINGS)
    @mock.patch("paperless_llm.parsers.LlmDocumentParser.get_multi_modal_llm")
    def test_align_with_llm_bad_json_fallback(self, mock_get_llm):
        mock_llm = mock.Mock()
        mock_response = mock.Mock()
        mock_response.message.content = "This is not valid JSON at all!"
        mock_llm.chat.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        parser = LlmDocumentParser(uuid.uuid4())
        lines = [
            HocrLine(
                index=0,
                bbox=(100, 100, 500, 150),
                text="Hello",
                words=[HocrWord(bbox=(100, 100, 500, 150), text="Hello")],
            ),
        ]
        corrections, extra = parser._align_with_llm(
            SAMPLE_DIR / "simple.png",
            "image/png",
            lines,
            "Hello",
        )
        self.assertEqual(corrections, {})
        self.assertEqual(extra, [])

    @override_settings(**OPENAI_SETTINGS)
    @mock.patch("paperless_llm.parsers.LlmDocumentParser.get_multi_modal_llm")
    def test_align_with_llm_strips_markdown_fences(self, mock_get_llm):
        mock_llm = mock.Mock()
        mock_response = mock.Mock()
        mock_response.message.content = (
            '```json\n{"lines": {"0": "Hello"}, "extra": []}\n```'
        )
        mock_llm.chat.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        parser = LlmDocumentParser(uuid.uuid4())
        lines = [
            HocrLine(
                index=0,
                bbox=(100, 100, 500, 150),
                text="He1lo",
                words=[HocrWord(bbox=(100, 100, 500, 150), text="He1lo")],
            ),
        ]
        corrections, _ = parser._align_with_llm(
            SAMPLE_DIR / "simple.png",
            "image/png",
            lines,
            "Hello",
        )
        self.assertEqual(corrections, {0: "Hello"})


class TestRebuildHocr(DirectoriesMixin, FileSystemAssertsMixin, TestCase):
    @override_settings(**OPENAI_SETTINGS)
    def test_rebuild_hocr_preserves_bboxes_replaces_text(self):
        parser = LlmDocumentParser(uuid.uuid4())
        lines = [
            HocrLine(
                index=0,
                bbox=(100, 100, 2000, 150),
                text="He1lo Wor1d",
                words=[
                    HocrWord(bbox=(100, 100, 300, 150), text="He1lo"),
                    HocrWord(bbox=(350, 100, 600, 150), text="Wor1d"),
                ],
            ),
            HocrLine(
                index=1,
                bbox=(100, 200, 2000, 250),
                text="Sec0nd l1ne",
                words=[
                    HocrWord(bbox=(100, 200, 400, 250), text="Sec0nd"),
                    HocrWord(bbox=(450, 200, 600, 250), text="l1ne"),
                ],
            ),
        ]
        corrections = {0: "Hello World", 1: "Second line"}
        result = parser._rebuild_hocr(2550, 3300, lines, corrections, [])

        # Verify it's valid XML
        import xml.etree.ElementTree as ET

        ET.fromstring(result)

        # Verify corrected text is present
        self.assertIn("Hello", result)
        self.assertIn("World", result)
        self.assertIn("Second", result)
        self.assertIn("line", result)

        # Verify original bboxes are preserved for lines
        self.assertIn("bbox 100 100 2000 150", result)
        self.assertIn("bbox 100 200 2000 250", result)

        # Verify OCR system tag
        self.assertIn("paperless-llm-aligned", result)

        # Verify old text not present
        self.assertNotIn("He1lo", result)
        self.assertNotIn("Wor1d", result)

    @override_settings(**OPENAI_SETTINGS)
    def test_rebuild_hocr_with_extra_lines(self):
        parser = LlmDocumentParser(uuid.uuid4())
        lines = [
            HocrLine(
                index=0,
                bbox=(100, 100, 500, 150),
                text="Hello",
                words=[HocrWord(bbox=(100, 100, 500, 150), text="Hello")],
            ),
        ]
        corrections = {0: "Hello"}
        extra = ["Footer text"]
        result = parser._rebuild_hocr(2550, 3300, lines, corrections, extra)

        self.assertIn("Footer", result)
        self.assertIn("text", result)
        self.assertIn("Hello", result)

    @override_settings(**OPENAI_SETTINGS)
    def test_rebuild_hocr_uncorrected_line_keeps_original(self):
        parser = LlmDocumentParser(uuid.uuid4())
        lines = [
            HocrLine(
                index=0,
                bbox=(100, 100, 500, 150),
                text="Original text",
                words=[
                    HocrWord(bbox=(100, 100, 300, 150), text="Original"),
                    HocrWord(bbox=(350, 100, 500, 150), text="text"),
                ],
            ),
        ]
        # No correction for line 0
        result = parser._rebuild_hocr(2550, 3300, lines, {}, [])
        self.assertIn("Original", result)
        self.assertIn("text", result)


class TestGenerateAlignedHocr(DirectoriesMixin, FileSystemAssertsMixin, TestCase):
    @override_settings(**OPENAI_SETTINGS, OCR_LANGUAGE="eng")
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._align_with_llm")
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._run_tesseract_hocr")
    def test_generate_aligned_hocr_full_pipeline(self, mock_tess, mock_align):
        sample_hocr = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml">\n'
            "<body>\n"
            '<div class="ocr_page" title="bbox 0 0 2550 3300">\n'
            '<span class="ocr_line" title="bbox 100 100 2000 150">\n'
            '<span class="ocrx_word" title="bbox 100 100 300 150">He1lo</span>\n'
            '<span class="ocrx_word" title="bbox 350 100 600 150">Wor1d</span>\n'
            "</span>\n"
            "</div>\n</body>\n</html>"
        )
        mock_tess.return_value = sample_hocr
        mock_align.return_value = ({0: "Hello World"}, [])

        parser = LlmDocumentParser(uuid.uuid4())
        result = parser._generate_aligned_hocr(
            SAMPLE_DIR / "simple.png",
            "image/png",
            "Hello World",
        )

        self.assertIsNotNone(result)
        self.assertIn("Hello", result)
        self.assertIn("World", result)
        self.assertIn("paperless-llm-aligned", result)

    @override_settings(**OPENAI_SETTINGS, OCR_LANGUAGE="eng")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._run_tesseract_hocr",
        return_value=None,
    )
    def test_generate_aligned_hocr_tesseract_fails(self, mock_tess):
        parser = LlmDocumentParser(uuid.uuid4())
        result = parser._generate_aligned_hocr(
            SAMPLE_DIR / "simple.png",
            "image/png",
            "Hello World",
        )
        self.assertIsNone(result)

    @override_settings(**OPENAI_SETTINGS, OCR_LANGUAGE="eng")
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._align_with_llm")
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._run_tesseract_hocr")
    def test_generate_aligned_hocr_alignment_fails(self, mock_tess, mock_align):
        sample_hocr = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml">\n'
            "<body>\n"
            '<div class="ocr_page" title="bbox 0 0 2550 3300">\n'
            '<span class="ocr_line" title="bbox 100 100 2000 150">\n'
            '<span class="ocrx_word" title="bbox 100 100 300 150">Hello</span>\n'
            "</span>\n"
            "</div>\n</body>\n</html>"
        )
        mock_tess.return_value = sample_hocr
        mock_align.return_value = ({}, [])  # Empty corrections = failure

        parser = LlmDocumentParser(uuid.uuid4())
        result = parser._generate_aligned_hocr(
            SAMPLE_DIR / "simple.png",
            "image/png",
            "Hello",
        )
        self.assertIsNone(result)


class TestParsePdfPagesReturnsPageOcrData(
    DirectoriesMixin,
    FileSystemAssertsMixin,
    TestCase,
):
    @override_settings(**OPENAI_SETTINGS)
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._generate_aligned_hocr",
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    @mock.patch("pdf2image.convert_from_path")
    def test_parse_pdf_pages_returns_page_ocr_data(
        self,
        mock_convert,
        mock_ocr,
        mock_aligned,
    ):
        mock_pages = [mock.Mock(), mock.Mock()]
        mock_convert.return_value = mock_pages
        mock_ocr.side_effect = ["Text 1", "Text 2"]
        mock_aligned.side_effect = ["<hocr>aligned1</hocr>", None]

        parser = LlmDocumentParser(uuid.uuid4())
        with mock.patch.object(parser, "progress"):
            results = parser._parse_pdf_pages(SAMPLE_DIR / "simple.pdf")

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], PageOcrData)
        self.assertEqual(results[0].text, "Text 1")
        self.assertEqual(results[0].hocr, "<hocr>aligned1</hocr>")
        self.assertEqual(results[1].text, "Text 2")
        self.assertIsNone(results[1].hocr)

    @override_settings(**OPENAI_SETTINGS, OCR_SKIP_ARCHIVE_FILE="never")
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._create_archive_pdf")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._generate_aligned_hocr",
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    @mock.patch("pdf2image.convert_from_path")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._extract_text_pdftotext",
    )
    def test_archive_uses_aligned_hocr(
        self,
        mock_extract,
        mock_convert,
        mock_ocr,
        mock_aligned,
        mock_archive,
    ):
        mock_extract.return_value = None
        mock_page = mock.Mock()
        mock_convert.return_value = [mock_page]
        mock_ocr.return_value = "Page text"
        mock_aligned.return_value = "<hocr>aligned</hocr>"
        mock_archive.return_value = Path("/tmp/archive.pdf")

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.pdf", "application/pdf")

        self.assertEqual(parser.text, "Page text")
        mock_archive.assert_called_once()
        page_data_list = mock_archive.call_args[0][2]
        self.assertEqual(len(page_data_list), 1)
        self.assertIsInstance(page_data_list[0], PageOcrData)
        self.assertEqual(page_data_list[0].text, "Page text")
        self.assertEqual(page_data_list[0].hocr, "<hocr>aligned</hocr>")


class TestMultiPageImage(DirectoriesMixin, FileSystemAssertsMixin, TestCase):
    @override_settings(**OPENAI_SETTINGS, OCR_SKIP_ARCHIVE_FILE="never")
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._create_archive_pdf")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._generate_aligned_hocr",
        return_value=None,
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._extract_image_pages",
    )
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._get_image_page_count",
        return_value=3,
    )
    def test_parse_multi_page_tiff(
        self,
        mock_page_count,
        mock_extract_pages,
        mock_ocr,
        mock_aligned,
        mock_archive,
    ):
        mock_extract_pages.return_value = [
            Path("/tmp/frame_0.png"),
            Path("/tmp/frame_1.png"),
            Path("/tmp/frame_2.png"),
        ]
        mock_ocr.side_effect = ["Page 1 text", "Page 2 text", "Page 3 text"]
        mock_archive.return_value = Path("/tmp/archive.pdf")

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.tiff", "image/tiff")

        self.assertEqual(
            parser.text,
            "Page 1 text\n\nPage 2 text\n\nPage 3 text",
        )
        self.assertEqual(mock_ocr.call_count, 3)
        mock_archive.assert_called_once()
        page_data_list = mock_archive.call_args[0][2]
        self.assertEqual(len(page_data_list), 3)

    @override_settings(**OPENAI_SETTINGS, OCR_SKIP_ARCHIVE_FILE="never")
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._create_archive_pdf")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._generate_aligned_hocr",
        return_value=None,
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._get_image_page_count",
        return_value=1,
    )
    def test_parse_single_page_image_unchanged(
        self,
        mock_page_count,
        mock_ocr,
        mock_aligned,
        mock_archive,
    ):
        mock_ocr.return_value = "Single page text"
        mock_archive.return_value = Path("/tmp/archive.pdf")

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.png", "image/png")

        self.assertEqual(parser.text, "Single page text")
        mock_ocr.assert_called_once()


class TestTextPreservationOnArchiveFailure(
    DirectoriesMixin,
    FileSystemAssertsMixin,
    TestCase,
):
    @override_settings(**OPENAI_SETTINGS, OCR_SKIP_ARCHIVE_FILE="never")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._create_archive_pdf",
        side_effect=RuntimeError("ocrmypdf failed"),
    )
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._generate_aligned_hocr",
        return_value=None,
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    @mock.patch("pdf2image.convert_from_path")
    @mock.patch(
        "paperless_llm.parsers.LlmDocumentParser._extract_text_pdftotext",
    )
    def test_text_preserved_when_archive_creation_fails(
        self,
        mock_extract,
        mock_convert,
        mock_ocr,
        mock_aligned,
        mock_archive,
    ):
        """Text should be preserved even if archive PDF creation fails."""
        mock_extract.return_value = None
        mock_page = mock.Mock()
        mock_convert.return_value = [mock_page]
        mock_ocr.return_value = "Extracted text that should survive"

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.pdf", "application/pdf")

        # Text must be preserved despite archive failure
        self.assertEqual(parser.text, "Extracted text that should survive")
        # Archive path should NOT be set
        self.assertIsNone(parser.archive_path)

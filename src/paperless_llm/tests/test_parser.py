import uuid
from pathlib import Path
from unittest import mock

from django.test import TestCase
from django.test import override_settings

from documents.tests.utils import DirectoriesMixin
from documents.tests.utils import FileSystemAssertsMixin
from paperless_llm.parsers import LlmDocumentParser
from paperless_llm.signals import get_parser


SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "documents" / "tests" / "samples"


class TestLlmParser(DirectoriesMixin, FileSystemAssertsMixin, TestCase):
    @override_settings(
        LLM_OCR_ENABLED=False,
        LLM_OCR_BACKEND=None,
    )
    def test_supported_mime_types_disabled(self) -> None:
        parser = LlmDocumentParser(uuid.uuid4())
        self.assertEqual(parser.supported_mime_types(), {})

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="openai",
        LLM_OCR_MODEL="gpt-4o",
        LLM_OCR_API_KEY="sk-test",
        LLM_OCR_ENDPOINT=None,
    )
    def test_supported_mime_types_enabled(self) -> None:
        parser = LlmDocumentParser(uuid.uuid4())
        expected_types = {
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
        self.assertEqual(parser.supported_mime_types(), expected_types)

    @override_settings(
        LLM_OCR_ENABLED=False,
        LLM_OCR_BACKEND=None,
    )
    def test_parse_disabled(self) -> None:
        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.pdf", "application/pdf")
        self.assertEqual(parser.text, "")

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="openai",
        LLM_OCR_MODEL="gpt-4o",
        LLM_OCR_API_KEY="sk-test",
        LLM_OCR_ENDPOINT=None,
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    @mock.patch("pdf2image.convert_from_path")
    def test_parse_pdf(self, mock_convert, mock_ocr) -> None:
        # Simulate a 2-page PDF
        mock_page1 = mock.Mock()
        mock_page2 = mock.Mock()
        mock_convert.return_value = [mock_page1, mock_page2]
        mock_ocr.side_effect = ["Page 1 text", "Page 2 text"]

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.pdf", "application/pdf")

        self.assertEqual(parser.text, "Page 1 text\n\nPage 2 text")
        self.assertEqual(mock_ocr.call_count, 2)
        self.assertEqual(mock_page1.save.call_count, 1)
        self.assertEqual(mock_page2.save.call_count, 1)

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="gemini",
        LLM_OCR_MODEL="models/gemini-2.0-flash",
        LLM_OCR_API_KEY="test-key",
        LLM_OCR_ENDPOINT=None,
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    def test_parse_image(self, mock_ocr) -> None:
        mock_ocr.return_value = "Extracted text from image"

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.png", "image/png")

        self.assertEqual(parser.text, "Extracted text from image")
        mock_ocr.assert_called_once()

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="openai",
        LLM_OCR_MODEL="gpt-4o",
        LLM_OCR_API_KEY="sk-test",
        LLM_OCR_ENDPOINT=None,
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    def test_parse_tiff_converts_to_png(self, mock_ocr) -> None:
        mock_ocr.return_value = "Text from tiff"

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.tiff", "image/tiff")

        self.assertEqual(parser.text, "Text from tiff")
        # _ocr_image should be called with a converted PNG path and mime type
        call_args = mock_ocr.call_args
        self.assertTrue(str(call_args[0][0]).endswith("_converted.png"))
        self.assertEqual(call_args[1]["mime_type"], "image/png")

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="openai",
        LLM_OCR_MODEL="gpt-4o",
        LLM_OCR_API_KEY="sk-test",
        LLM_OCR_ENDPOINT=None,
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    def test_parse_native_image_no_conversion(self, mock_ocr) -> None:
        mock_ocr.return_value = "Text from png"

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.png", "image/png")

        self.assertEqual(parser.text, "Text from png")
        # _ocr_image should be called with original path and mime type
        call_args = mock_ocr.call_args
        self.assertEqual(str(call_args[0][0]), str(SAMPLE_DIR / "simple.png"))
        self.assertEqual(call_args[1]["mime_type"], "image/png")

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="openai",
        LLM_OCR_MODEL="gpt-4o",
        LLM_OCR_API_KEY="sk-test",
        LLM_OCR_ENDPOINT=None,
    )
    @mock.patch("paperless_llm.parsers.LlmDocumentParser._ocr_image")
    def test_parse_error_handled(self, mock_ocr) -> None:
        mock_ocr.side_effect = RuntimeError("API error")

        parser = get_parser(uuid.uuid4())
        parser.parse(SAMPLE_DIR / "simple.png", "image/png")

        self.assertEqual(parser.text, "")

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="openai",
        LLM_OCR_MODEL="gpt-4o",
        LLM_OCR_API_KEY="sk-test",
        LLM_OCR_ENDPOINT=None,
    )
    def test_get_multi_modal_llm_openai(self) -> None:
        with mock.patch(
            "llama_index.multi_modal_llms.openai.OpenAIMultiModal",
        ) as mock_cls:
            parser = LlmDocumentParser(uuid.uuid4())
            parser.get_multi_modal_llm()
            mock_cls.assert_called_once()

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="gemini",
        LLM_OCR_MODEL="models/gemini-2.0-flash",
        LLM_OCR_API_KEY="test-key",
        LLM_OCR_ENDPOINT=None,
    )
    def test_get_multi_modal_llm_gemini(self) -> None:
        with mock.patch(
            "llama_index.multi_modal_llms.gemini.GeminiMultiModal",
        ) as mock_cls:
            parser = LlmDocumentParser(uuid.uuid4())
            parser.get_multi_modal_llm()
            mock_cls.assert_called_once()

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="ollama",
        LLM_OCR_MODEL="llava",
        LLM_OCR_API_KEY=None,
        LLM_OCR_ENDPOINT="http://localhost:11434",
    )
    def test_get_multi_modal_llm_ollama(self) -> None:
        with mock.patch(
            "llama_index.multi_modal_llms.ollama.OllamaMultiModal",
        ) as mock_cls:
            parser = LlmDocumentParser(uuid.uuid4())
            parser.get_multi_modal_llm()
            mock_cls.assert_called_once()

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="invalid",
        LLM_OCR_MODEL=None,
        LLM_OCR_API_KEY=None,
        LLM_OCR_ENDPOINT=None,
    )
    def test_get_multi_modal_llm_unsupported(self) -> None:
        parser = LlmDocumentParser(uuid.uuid4())
        with self.assertRaises(ValueError):
            parser.get_multi_modal_llm()

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="openai",
        LLM_OCR_MODEL="gpt-4o",
        LLM_OCR_API_KEY="sk-test",
        LLM_OCR_ENDPOINT=None,
    )
    def test_get_page_count_pdf(self) -> None:
        parser = LlmDocumentParser(uuid.uuid4())
        count = parser.get_page_count(SAMPLE_DIR / "simple.pdf", "application/pdf")
        self.assertIsNotNone(count)
        self.assertGreaterEqual(count, 1)

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="openai",
        LLM_OCR_MODEL="gpt-4o",
        LLM_OCR_API_KEY="sk-test",
        LLM_OCR_ENDPOINT=None,
    )
    def test_get_page_count_image(self) -> None:
        parser = LlmDocumentParser(uuid.uuid4())
        count = parser.get_page_count(SAMPLE_DIR / "simple.png", "image/png")
        self.assertEqual(count, 1)

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="openai",
        LLM_OCR_MODEL="gpt-4o",
        LLM_OCR_API_KEY="sk-test",
        LLM_OCR_ENDPOINT=None,
    )
    def test_signal_weight(self) -> None:
        from paperless_llm.signals import llm_consumer_declaration

        result = llm_consumer_declaration(None)
        self.assertEqual(result["weight"], 10)

from unittest import TestCase

from django.test import override_settings

from paperless_llm import check_llm_ocr_configured


class TestChecks(TestCase):
    @override_settings(LLM_OCR_ENABLED=False)
    def test_disabled(self) -> None:
        msgs = check_llm_ocr_configured(None)
        self.assertEqual(len(msgs), 0)

    @override_settings(LLM_OCR_ENABLED=True, LLM_OCR_BACKEND=None)
    def test_enabled_no_backend(self) -> None:
        msgs = check_llm_ocr_configured(None)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].id, "paperless_llm.W001")

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="openai",
        LLM_OCR_API_KEY=None,
    )
    def test_openai_no_api_key(self) -> None:
        msgs = check_llm_ocr_configured(None)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].id, "paperless_llm.W002")

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="gemini",
        LLM_OCR_API_KEY=None,
    )
    def test_gemini_no_api_key(self) -> None:
        msgs = check_llm_ocr_configured(None)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].id, "paperless_llm.W002")

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="ollama",
        LLM_OCR_API_KEY=None,
    )
    def test_ollama_no_api_key_ok(self) -> None:
        msgs = check_llm_ocr_configured(None)
        self.assertEqual(len(msgs), 0)

    @override_settings(
        LLM_OCR_ENABLED=True,
        LLM_OCR_BACKEND="openai",
        LLM_OCR_API_KEY="sk-test",
    )
    def test_valid_config(self) -> None:
        msgs = check_llm_ocr_configured(None)
        self.assertEqual(len(msgs), 0)

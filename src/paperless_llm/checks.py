from django.conf import settings
from django.core.checks import Warning
from django.core.checks import register


@register()
def check_llm_ocr_configured(app_configs, **kwargs):
    if not settings.LLM_OCR_ENABLED:
        return []

    errors = []

    if not settings.LLM_OCR_BACKEND:
        errors.append(
            Warning(
                "LLM OCR is enabled but no backend is configured. "
                "Set PAPERLESS_LLM_OCR_BACKEND to 'openai', 'gemini', or 'ollama'.",
                id="paperless_llm.W001",
            ),
        )

    if settings.LLM_OCR_BACKEND in ("openai", "gemini") and not settings.LLM_OCR_API_KEY:
        errors.append(
            Warning(
                f"LLM OCR backend '{settings.LLM_OCR_BACKEND}' requires an API key. "
                "Set PAPERLESS_LLM_OCR_API_KEY.",
                id="paperless_llm.W002",
            ),
        )

    return errors

# this is here so that django finds the checks.
from paperless_llm.checks import check_llm_ocr_configured

__all__ = ["check_llm_ocr_configured"]

import html
import threading
from pathlib import Path

from ocrmypdf import hookimpl
from ocrmypdf.pluginspec import OcrEngine
from ocrmypdf.pluginspec import OrientationConfidence


class _LlmPageStore:
    """Thread-safe store for passing pre-computed page data to the OCR engine."""

    def __init__(self):
        self._pages: list = []  # list[PageOcrData]
        self._index = 0
        self._lock = threading.Lock()

    def set_pages(self, pages: list):
        with self._lock:
            self._pages = list(pages)
            self._index = 0

    def next_page(self):
        """Return the next PageOcrData, or a fallback with empty text."""
        from paperless_llm.parsers import PageOcrData

        with self._lock:
            if self._index < len(self._pages):
                page = self._pages[self._index]
                self._index += 1
                return page
            return PageOcrData(text="")


page_data_store = _LlmPageStore()


def _generate_approximate_hocr(text: str, w: int, h: int) -> str:
    """Build hOCR with approximate word positions (fallback when no
    pre-computed aligned hOCR is available)."""
    escaped = html.escape(text)
    lines = escaped.split("\n")
    line_height = max(1, h // max(len(lines), 1))

    line_elements = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        y0 = i * line_height
        y1 = min((i + 1) * line_height, h)
        words = line.split()
        if not words:
            continue
        # Distribute by character count including spaces for proper gaps
        total_chars = max(
            sum(len(wd) for wd in words) + max(len(words) - 1, 0),
            1,
        )
        word_spans = []
        char_offset = 0
        for word in words:
            x0 = (char_offset * w) // total_chars
            x1 = ((char_offset + len(word)) * w) // total_chars
            word_spans.append(
                f'<span class="ocrx_word" '
                f'title="bbox {x0} {y0} {x1} {y1}">{word}</span>',
            )
            char_offset += len(word) + 1  # +1 for inter-word space
        line_elements.append(
            f'<span class="ocr_line" title="bbox 0 {y0} {w} {y1}">'
            + " ".join(word_spans)
            + "</span>",
        )

    body = "\n".join(line_elements)

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE html PUBLIC "
        '"-//W3C//DTD XHTML 1.0 Transitional//EN"\n'
        ' "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'
        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
        "<head>\n"
        ' <meta charset="utf-8"/>\n'
        ' <meta name="ocr-system" content="paperless-llm"/>\n'
        "</head>\n"
        "<body>\n"
        f'<div class="ocr_page" title="bbox 0 0 {w} {h}; ppageno 0">\n'
        f'<div class="ocr_carea" title="bbox 0 0 {w} {h}">\n'
        f'<p class="ocr_par" title="bbox 0 0 {w} {h}">\n'
        f"{body}\n"
        "</p>\n</div>\n</div>\n"
        "</body>\n</html>"
    )


class LlmOcrEngine(OcrEngine):
    """OCR engine that returns pre-computed LLM-extracted text instead of
    running traditional OCR.  Used as an ocrmypdf plugin so that ocrmypdf
    handles PDF/A creation and text-layer embedding."""

    @staticmethod
    def version() -> str:
        return "1.0"

    @staticmethod
    def creator_tag(options) -> str:
        return "paperless-llm 1.0"

    def __str__(self):
        return "paperless-llm 1.0"

    @staticmethod
    def languages(options) -> set:
        return set()

    @staticmethod
    def get_orientation(input_file, options):
        return OrientationConfidence(angle=0, confidence=0)

    @staticmethod
    def generate_hocr(
        input_file: Path,
        output_hocr: Path,
        output_text: Path,
        options,
    ) -> None:
        page_data = page_data_store.next_page()
        output_text.write_text(page_data.text, encoding="utf-8")

        if page_data.hocr is not None:
            # Use pre-computed aligned hOCR directly
            output_hocr.write_text(page_data.hocr, encoding="utf-8")
        else:
            # Fall back to approximate hOCR generation
            from PIL import Image

            with Image.open(input_file) as im:
                w, h = im.size

            hocr = _generate_approximate_hocr(page_data.text, w, h)
            output_hocr.write_text(hocr, encoding="utf-8")

    @staticmethod
    def generate_pdf(
        input_file: Path,
        output_pdf: Path,
        output_text: Path,
        options,
    ) -> None:
        # Minimal stub — only called when the sandwich renderer is selected;
        # the default hocr renderer uses generate_hocr instead.
        page_data = page_data_store.next_page()
        output_text.write_text(page_data.text, encoding="utf-8")

        import pikepdf

        pdf = pikepdf.Pdf.new()
        pdf.pages.append(
            pdf.make_indirect(
                pikepdf.Dictionary(
                    Type=pikepdf.Name.Page,
                    MediaBox=[0, 0, 612, 792],
                ),
            ),
        )
        pdf.save(str(output_pdf))


@hookimpl
def get_ocr_engine():
    return LlmOcrEngine()

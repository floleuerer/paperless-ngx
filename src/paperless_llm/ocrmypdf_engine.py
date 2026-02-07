import html
import threading
from pathlib import Path

from ocrmypdf import hookimpl
from ocrmypdf.pluginspec import OcrEngine
from ocrmypdf.pluginspec import OrientationConfidence


class _LlmTextStore:
    """Thread-safe store for passing pre-computed page texts to the OCR engine."""

    def __init__(self):
        self._texts: list[str] = []
        self._index = 0
        self._lock = threading.Lock()

    def set_texts(self, texts: list[str]):
        with self._lock:
            self._texts = list(texts)
            self._index = 0

    def next_text(self) -> str:
        with self._lock:
            if self._index < len(self._texts):
                text = self._texts[self._index]
                self._index += 1
                return text
            return ""


page_text_store = _LlmTextStore()


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
        from PIL import Image

        text = page_text_store.next_text()
        output_text.write_text(text, encoding="utf-8")

        with Image.open(input_file) as im:
            w, h = im.size

        # Build hOCR with approximate word positions so the text layer
        # is searchable and roughly selectable in PDF viewers.
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
            word_width = max(1, w // len(words))
            word_spans = []
            for j, word in enumerate(words):
                x0 = j * word_width
                x1 = min((j + 1) * word_width, w)
                word_spans.append(
                    f'<span class="ocrx_word" '
                    f'title="bbox {x0} {y0} {x1} {y1}">{word}</span>'
                )
            line_elements.append(
                f'<span class="ocr_line" title="bbox 0 {y0} {w} {y1}">'
                + " ".join(word_spans)
                + "</span>"
            )

        body = "\n".join(line_elements)

        hocr = (
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
        text = page_text_store.next_text()
        output_text.write_text(text, encoding="utf-8")

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

import tempfile
from pathlib import Path
from unittest import TestCase

from PIL import Image

from paperless_llm.ocrmypdf_engine import LlmOcrEngine
from paperless_llm.ocrmypdf_engine import _LlmTextStore
from paperless_llm.ocrmypdf_engine import get_ocr_engine
from paperless_llm.ocrmypdf_engine import page_text_store


class TestLlmTextStore(TestCase):
    def setUp(self):
        self.store = _LlmTextStore()

    def test_set_and_get_texts(self):
        self.store.set_texts(["page1", "page2", "page3"])
        self.assertEqual(self.store.next_text(), "page1")
        self.assertEqual(self.store.next_text(), "page2")
        self.assertEqual(self.store.next_text(), "page3")

    def test_next_text_beyond_length(self):
        self.store.set_texts(["only one"])
        self.assertEqual(self.store.next_text(), "only one")
        self.assertEqual(self.store.next_text(), "")
        self.assertEqual(self.store.next_text(), "")

    def test_set_texts_resets_index(self):
        self.store.set_texts(["first", "second"])
        self.store.next_text()  # consume "first"
        self.store.set_texts(["new1", "new2"])
        self.assertEqual(self.store.next_text(), "new1")

    def test_empty_texts(self):
        self.store.set_texts([])
        self.assertEqual(self.store.next_text(), "")

    def test_set_texts_copies_list(self):
        texts = ["a", "b"]
        self.store.set_texts(texts)
        texts.append("c")  # modify original
        self.store.next_text()  # "a"
        self.store.next_text()  # "b"
        self.assertEqual(self.store.next_text(), "")  # no "c"


class TestLlmOcrEngine(TestCase):
    def test_version(self):
        self.assertEqual(LlmOcrEngine.version(), "1.0")

    def test_creator_tag(self):
        self.assertEqual(LlmOcrEngine.creator_tag(None), "paperless-llm 1.0")

    def test_str(self):
        engine = LlmOcrEngine()
        self.assertEqual(str(engine), "paperless-llm 1.0")

    def test_languages(self):
        self.assertEqual(LlmOcrEngine.languages(None), set())

    def test_get_orientation(self):
        result = LlmOcrEngine.get_orientation(None, None)
        self.assertEqual(result.angle, 0)
        self.assertEqual(result.confidence, 0)

    def test_generate_hocr(self):
        page_text_store.set_texts(["Hello World\nSecond line"])

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            Image.new("RGB", (200, 100)).save(str(img_path))

            hocr_path = Path(tmpdir) / "output.hocr"
            text_path = Path(tmpdir) / "output.txt"

            LlmOcrEngine.generate_hocr(img_path, hocr_path, text_path, None)

            self.assertEqual(
                text_path.read_text(encoding="utf-8"),
                "Hello World\nSecond line",
            )

            hocr = hocr_path.read_text(encoding="utf-8")
            self.assertIn('<?xml version="1.0"', hocr)
            self.assertIn("ocr_page", hocr)
            self.assertIn("ocrx_word", hocr)
            self.assertIn("Hello", hocr)
            self.assertIn("World", hocr)
            self.assertIn("Second", hocr)
            self.assertIn("bbox", hocr)
            self.assertIn("200", hocr)
            self.assertIn("100", hocr)

    def test_generate_hocr_empty_text(self):
        page_text_store.set_texts([""])

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            Image.new("RGB", (200, 100)).save(str(img_path))

            hocr_path = Path(tmpdir) / "output.hocr"
            text_path = Path(tmpdir) / "output.txt"

            LlmOcrEngine.generate_hocr(img_path, hocr_path, text_path, None)

            self.assertEqual(text_path.read_text(encoding="utf-8"), "")
            hocr = hocr_path.read_text(encoding="utf-8")
            self.assertIn("ocr_page", hocr)
            self.assertNotIn("ocrx_word", hocr)

    def test_generate_hocr_escapes_html(self):
        page_text_store.set_texts(['<b>bold</b> & "quotes"'])

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            Image.new("RGB", (200, 100)).save(str(img_path))

            hocr_path = Path(tmpdir) / "output.hocr"
            text_path = Path(tmpdir) / "output.txt"

            LlmOcrEngine.generate_hocr(img_path, hocr_path, text_path, None)

            hocr = hocr_path.read_text(encoding="utf-8")
            self.assertIn("&lt;b&gt;", hocr)
            self.assertIn("&amp;", hocr)
            self.assertIn("&quot;quotes&quot;", hocr)

    def test_generate_pdf(self):
        import pikepdf

        page_text_store.set_texts(["PDF page text"])

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            pdf_path = Path(tmpdir) / "output.pdf"
            text_path = Path(tmpdir) / "output.txt"

            LlmOcrEngine.generate_pdf(input_path, pdf_path, text_path, None)

            self.assertEqual(
                text_path.read_text(encoding="utf-8"),
                "PDF page text",
            )
            self.assertTrue(pdf_path.exists())

            with pikepdf.open(pdf_path) as pdf:
                self.assertEqual(len(pdf.pages), 1)

    def test_generate_hocr_multiline(self):
        page_text_store.set_texts(["Line one\nLine two\n\nLine four"])

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            Image.new("RGB", (300, 200)).save(str(img_path))

            hocr_path = Path(tmpdir) / "output.hocr"
            text_path = Path(tmpdir) / "output.txt"

            LlmOcrEngine.generate_hocr(img_path, hocr_path, text_path, None)

            hocr = hocr_path.read_text(encoding="utf-8")
            self.assertEqual(hocr.count("ocr_line"), 3)


class TestGetOcrEngine(TestCase):
    def test_returns_instance(self):
        engine = get_ocr_engine()
        self.assertIsInstance(engine, LlmOcrEngine)

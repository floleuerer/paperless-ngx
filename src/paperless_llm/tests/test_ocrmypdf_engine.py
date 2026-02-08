import tempfile
from pathlib import Path
from unittest import TestCase

from PIL import Image

from paperless_llm.ocrmypdf_engine import LlmOcrEngine
from paperless_llm.ocrmypdf_engine import _generate_approximate_hocr
from paperless_llm.ocrmypdf_engine import _LlmPageStore
from paperless_llm.ocrmypdf_engine import get_ocr_engine
from paperless_llm.ocrmypdf_engine import page_data_store
from paperless_llm.parsers import PageOcrData


class TestLlmPageStore(TestCase):
    def setUp(self):
        self.store = _LlmPageStore()

    def test_set_and_get_pages(self):
        pages = [
            PageOcrData(text="page1"),
            PageOcrData(text="page2"),
            PageOcrData(text="page3"),
        ]
        self.store.set_pages(pages)
        self.assertEqual(self.store.next_page().text, "page1")
        self.assertEqual(self.store.next_page().text, "page2")
        self.assertEqual(self.store.next_page().text, "page3")

    def test_next_page_beyond_length(self):
        self.store.set_pages([PageOcrData(text="only one")])
        self.assertEqual(self.store.next_page().text, "only one")
        self.assertEqual(self.store.next_page().text, "")

    def test_set_pages_resets_index(self):
        self.store.set_pages([PageOcrData(text="first"), PageOcrData(text="second")])
        self.store.next_page()  # consume "first"
        self.store.set_pages([PageOcrData(text="new1"), PageOcrData(text="new2")])
        self.assertEqual(self.store.next_page().text, "new1")

    def test_empty_pages(self):
        self.store.set_pages([])
        self.assertEqual(self.store.next_page().text, "")

    def test_set_pages_copies_list(self):
        pages = [PageOcrData(text="a"), PageOcrData(text="b")]
        self.store.set_pages(pages)
        pages.append(PageOcrData(text="c"))  # modify original
        self.store.next_page()  # "a"
        self.store.next_page()  # "b"
        self.assertEqual(self.store.next_page().text, "")  # no "c"

    def test_page_with_hocr(self):
        pages = [PageOcrData(text="text", hocr="<hocr>content</hocr>")]
        self.store.set_pages(pages)
        page = self.store.next_page()
        self.assertEqual(page.text, "text")
        self.assertEqual(page.hocr, "<hocr>content</hocr>")

    def test_page_without_hocr(self):
        pages = [PageOcrData(text="text")]
        self.store.set_pages(pages)
        page = self.store.next_page()
        self.assertEqual(page.text, "text")
        self.assertIsNone(page.hocr)


class TestGenerateApproximateHocr(TestCase):
    def test_basic_output(self):
        hocr = _generate_approximate_hocr("Hello World\nSecond line", 200, 100)
        self.assertIn('<?xml version="1.0"', hocr)
        self.assertIn("ocr_page", hocr)
        self.assertIn("ocrx_word", hocr)
        self.assertIn("Hello", hocr)
        self.assertIn("World", hocr)
        self.assertIn("Second", hocr)

    def test_empty_text(self):
        hocr = _generate_approximate_hocr("", 200, 100)
        self.assertIn("ocr_page", hocr)
        self.assertNotIn("ocrx_word", hocr)

    def test_escapes_html(self):
        hocr = _generate_approximate_hocr('<b>bold</b> & "quotes"', 200, 100)
        self.assertIn("&lt;b&gt;", hocr)
        self.assertIn("&amp;", hocr)


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

    def test_generate_hocr_with_precomputed_hocr(self):
        precomputed_hocr = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<html><body><div class="ocr_page" '
            'title="bbox 0 0 200 100">aligned</div></body></html>'
        )
        page_data_store.set_pages(
            [PageOcrData(text="Hello World", hocr=precomputed_hocr)],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            Image.new("RGB", (200, 100)).save(str(img_path))

            hocr_path = Path(tmpdir) / "output.hocr"
            text_path = Path(tmpdir) / "output.txt"

            LlmOcrEngine.generate_hocr(img_path, hocr_path, text_path, None)

            self.assertEqual(
                text_path.read_text(encoding="utf-8"),
                "Hello World",
            )
            hocr = hocr_path.read_text(encoding="utf-8")
            self.assertEqual(hocr, precomputed_hocr)

    def test_generate_hocr_fallback_no_precomputed(self):
        page_data_store.set_pages(
            [PageOcrData(text="Hello World\nSecond line")],
        )

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
        page_data_store.set_pages([PageOcrData(text="")])

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
        page_data_store.set_pages(
            [PageOcrData(text='<b>bold</b> & "quotes"')],
        )

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

        page_data_store.set_pages([PageOcrData(text="PDF page text")])

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
        page_data_store.set_pages(
            [PageOcrData(text="Line one\nLine two\n\nLine four")],
        )

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

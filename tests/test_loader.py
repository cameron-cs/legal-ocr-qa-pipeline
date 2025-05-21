import unittest
from unittest.mock import patch, mock_open

from src.loader.ocr_loader import JSONOCRReader
from src.objects.ocr_semantic_page import OCRPageBlock


class TestJSONOCRReader(unittest.TestCase):

    def setUp(self):
        self.sample_json = {
            "pages": [
                {
                    "page_number": 1,
                    "lines": [
                        {
                            "content": "Line one.",
                            "words": [{"content": "Line", "confidence": 0.99}, {"content": "one.", "confidence": 0.98}]
                        },
                        {
                            "content": "Line two with low confidence.",
                            "words": [{"content": "low", "confidence": 0.4}]
                        }
                    ]
                },
                {
                    "page_number": 2,
                    "lines": [
                        {
                            "content": "Page two line.",
                            "words": [{"content": "Page", "confidence": 0.9}]
                        }
                    ]
                }
            ]
        }

    @patch("src.loader.ocr_loader.Path.exists", return_value=True)
    @patch("src.loader.ocr_loader.open", new_callable=mock_open, read_data="")
    @patch("src.loader.ocr_loader.json.load")
    def test_load_and_parse_json(self, mock_json_load, mock_file, mock_exists):
        mock_json_load.return_value = self.sample_json

        reader = JSONOCRReader("dummy.json")
        self.assertEqual(reader.data, self.sample_json)
        mock_exists.assert_called_once()
        mock_file.assert_called_once()

    @patch("src.loader.ocr_loader.Path.exists", return_value=False)
    def test_file_not_found(self, mock_exists):
        with self.assertRaises(FileNotFoundError):
            JSONOCRReader("missing.json")

    @patch("src.loader.ocr_loader.Path.exists", return_value=True)
    @patch("src.loader.ocr_loader.open", new_callable=mock_open, read_data="")
    @patch("src.loader.ocr_loader.json.load")
    def test_get_all_pages_with_confidence_filtering(self, mock_json_load, mock_file, mock_exists):
        mock_json_load.return_value = self.sample_json

        reader = JSONOCRReader("dummy.json")
        pages = reader.get_all_pages(min_confidence=0.85)

        self.assertEqual(len(pages), 2)  # Page 1 (filtered) and Page 2
        self.assertIsInstance(pages[0], OCRPageBlock)
        self.assertEqual(pages[0].page, 1)
        self.assertIn("Line one.", pages[0].text)
        self.assertNotIn("low confidence", pages[0].text)  # should be filtered out

    @patch("src.loader.ocr_loader.Path.exists", return_value=True)
    @patch("src.loader.ocr_loader.open", new_callable=mock_open, read_data="")
    @patch("src.loader.ocr_loader.json.load")
    def test_get_all_pages_with_no_filtering(self, mock_json_load, mock_file, mock_exists):
        mock_json_load.return_value = self.sample_json

        reader = JSONOCRReader("dummy.json")
        pages = reader.get_all_pages(min_confidence=0.0)

        self.assertEqual(len(pages), 2)
        self.assertTrue(any("low confidence" in line for line in pages[0].lines))


if __name__ == '__main__':
    unittest.main()

import unittest

from app import build_parser


class AppCliTest(unittest.TestCase):
    def test_parser_requires_annotator_and_supports_no_browser(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--annotator", "annotator_1", "--port", "9999", "--no-browser"]
        )
        self.assertEqual(args.annotator, "annotator_1")
        self.assertEqual(args.port, 9999)
        self.assertTrue(args.no_browser)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WebAssetsTest(unittest.TestCase):
    def test_page_contains_annotation_workspace_without_external_urls(self):
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="guideline-text"', html)
        self.assertIn('id="score-buttons"', html)
        self.assertIn('id="save-status"', html)
        self.assertIn('id="overall-progress"', html)
        self.assertIn('id="query-translation"', html)
        self.assertIn('id="document-translation"', html)
        self.assertIn("/vendor/katex.min.js", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("http://", html)

    def test_javascript_saves_scores_and_supports_keyboard_navigation(self):
        javascript = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn('fetch("/api/annotations"', javascript)
        self.assertIn('addEventListener("keydown"', javascript)
        self.assertIn("renderMathInElement", javascript)
        self.assertIn("textContent", javascript)

    def test_styles_include_responsive_two_panel_layout(self):
        css = (ROOT / "web" / "styles.css").read_text(encoding="utf-8")
        self.assertIn("grid-template-columns", css)
        self.assertIn("@media", css)


if __name__ == "__main__":
    unittest.main()

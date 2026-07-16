"""Unit tests for shared_libraries.web_search_util."""

from unittest import mock

from machine_learning_engineering.shared_libraries import web_search_util


class TestSearchWeb:
    """Tests for search_web."""

    def test_returns_results_from_ddgs(self):
        fake_results = [
            {
                "title": "Result A",
                "href": "https://a.example",
                "body": "Body A",
            },
            {
                "title": "Result B",
                "href": "https://b.example",
                "body": "Body B",
            },
        ]

        ddgs_instance = mock.MagicMock()
        ddgs_instance.text.return_value = iter(fake_results)

        with mock.patch(
            "machine_learning_engineering.shared_libraries.web_search_util.DDGS"
        ) as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__ = mock.MagicMock(
                return_value=ddgs_instance
            )
            mock_ddgs_class.return_value.__exit__ = mock.MagicMock(
                return_value=False
            )
            results = web_search_util.search_web("test query", num_results=2)

        assert results == fake_results

    def test_returns_empty_list_on_failure(self):
        with mock.patch(
            "machine_learning_engineering.shared_libraries.web_search_util.DDGS",
            side_effect=RuntimeError("network failure"),
        ):
            results = web_search_util.search_web("test query")

        assert results == []


class TestFormatResultsForPrompt:
    """Tests for format_results_for_prompt."""

    def test_empty_results(self):
        formatted = web_search_util.format_results_for_prompt([])
        assert formatted == "No web search results available."

    def test_non_empty_results(self):
        results = [
            {"title": "Result A", "body": "Body A\nwith newline"},
            {"title": "Result B", "body": "Body B"},
        ]
        formatted = web_search_util.format_results_for_prompt(results)
        assert "1. Result A - Body A with newline" in formatted
        assert "2. Result B - Body B" in formatted

    def test_missing_body(self):
        results = [{"title": "Result A"}]
        formatted = web_search_util.format_results_for_prompt(results)
        assert formatted == "1. Result A - "

    def test_missing_title(self):
        results = [{"body": "Body A"}]
        formatted = web_search_util.format_results_for_prompt(results)
        assert formatted == "1. No title - Body A"

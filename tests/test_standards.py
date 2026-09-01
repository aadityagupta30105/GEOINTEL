"""
Engineering standards enforcement.

These tests encode the platform presentation contract as executable checks so
that a regression fails the suite rather than reaching an operator console.
"""

from __future__ import annotations

import ast
import unicodedata
from pathlib import Path
from typing import Final

import pytest

# Directories excluded from source scanning.
_EXCLUDED_DIRS: Final[frozenset[str]] = frozenset({
    "venv", ".venv", ".git", "__pycache__", "output", "node_modules", ".pytest_cache",
})

# Unicode general categories covering pictographs and modifier symbols.
_EMOJI_CATEGORIES: Final[frozenset[str]] = frozenset({"So", "Sk"})

# Code points that are symbols by category but are legitimate typographic
# signs in a technical interface, not pictographs. Box-drawing characters are
# included so that directory trees in documentation are not misreported.
_ALLOWED_SYMBOLS: Final[frozenset[str]] = frozenset(
    {
        "°",  # degree sign
        "´",  # acute accent
        "ˆ",  # modifier circumflex
        "←", "↑", "→", "↓",  # arrows
        "↔", "↕",                      # bidirectional arrows
        "▲", "▼", "◆", "■",  # geometric shapes
    }
    # U+2500 to U+257F: box drawing, used for directory trees and rules.
    | {chr(code) for code in range(0x2500, 0x2580)}
)

# Constructed by code point so that this module stays pure ASCII and does not
# trip its own detector.
_VARIATION_SELECTOR_16: Final[str] = chr(0xFE0F)
_ZERO_WIDTH_JOINER: Final[str] = chr(0x200D)


def _source_files(root: Path) -> list[Path]:
    """Collect first-party Python sources under ``root``.

    Parameters
    ----------
    root : pathlib.Path
        Repository root.

    Returns
    -------
    list of pathlib.Path
        Python files outside the excluded directories.
    """
    return sorted(
        path
        for path in root.rglob("*.py")
        if not _EXCLUDED_DIRS & set(path.relative_to(root).parts)
    )


def _emoji_positions(text: str) -> list[tuple[int, str]]:
    """Locate pictographic code points in a string.

    ASCII is skipped outright: no emoji is representable in ASCII, and several
    ASCII punctuation marks (notably the grave accent used by reStructuredText
    literals) share the ``Sk`` modifier-symbol category with true pictographs.

    Parameters
    ----------
    text : str
        Text to scan.

    Returns
    -------
    list of tuple of (int, str)
        Line number (1-indexed) and offending character for each match.
    """
    findings: list[tuple[int, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for character in line:
            if character.isascii() or character in _ALLOWED_SYMBOLS:
                continue
            if character in (_VARIATION_SELECTOR_16, _ZERO_WIDTH_JOINER):
                findings.append((line_number, character))
                continue
            if unicodedata.category(character) in _EMOJI_CATEGORIES:
                findings.append((line_number, character))
    return findings


class TestZeroEmojiPolicy:
    """Emojis are banned from all first-party source, output and markup."""

    def test_no_emoji_in_python_sources(self, project_root: Path) -> None:
        violations: list[str] = []
        for path in _source_files(project_root):
            content = path.read_text(encoding="utf-8")
            for line_number, character in _emoji_positions(content):
                violations.append(
                    f"{path.relative_to(project_root)}:{line_number} "
                    f"U+{ord(character):04X} {unicodedata.name(character, 'UNNAMED')}"
                )
        assert not violations, "Emoji policy violations:\n" + "\n".join(violations)

    def test_no_emoji_in_string_literals(self, project_root: Path) -> None:
        """Literals reach the console and the DOM, so they are checked directly."""
        violations: list[str] = []
        for path in _source_files(project_root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if _emoji_positions(node.value):
                        violations.append(
                            f"{path.relative_to(project_root)}:{node.lineno}"
                        )
        assert not violations, "Emoji in string literals:\n" + "\n".join(violations)


class TestStatusMarkers:
    """Console status reporting uses bracketed ASCII markers."""

    def test_markers_are_ascii(self) -> None:
        from utils.logging_config import ARROW, ERROR, OK, WARN

        for marker in (OK, WARN, ERROR, ARROW):
            assert marker.isascii()

    def test_marker_format(self) -> None:
        from utils.logging_config import ERROR, OK, WARN

        assert (OK, WARN, ERROR) == ("[OK]", "[WARN]", "[ERROR]")

    def test_formatter_strips_non_ascii(self) -> None:
        """A non-ASCII payload must not raise on a legacy console code page."""
        import io
        import logging

        from utils.logging_config import configure_logging

        stream = io.StringIO()
        logger = configure_logging(stream=stream, force=True)
        logger.info("payload %s", "café — naïve")
        rendered = stream.getvalue()

        assert rendered.isascii()
        assert "payload" in rendered

        configure_logging(force=True)  # restore the default handler

    def test_rule_and_section_are_ascii(self) -> None:
        from utils.logging_config import rule, section

        assert rule().isascii()
        banner = section("stage 1: collection")
        assert banner.isascii()
        assert "STAGE 1: COLLECTION" in banner


class TestTypeAnnotations:
    """Public callables are fully annotated."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "utils/logging_config.py",
            "data/gdelt_collector.py",
            "analysis/graph_builder.py",
            "analysis/narrator.py",
            "models/event_classifier.py",
            "dashboard/theme.py",
            "dashboard/geodata.py",
            "dashboard/blocs.py",
            "dashboard/figures.py",
            "dashboard/app.py",
            "main.py",
        ],
    )
    def test_functions_declare_return_types(
        self, project_root: Path, module_path: str
    ) -> None:
        tree = ast.parse(
            (project_root / module_path).read_text(encoding="utf-8"),
            filename=module_path,
        )
        missing = [
            f"{module_path}:{node.lineno} {node.name}"
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.returns is None
        ]
        assert not missing, "Missing return annotations:\n" + "\n".join(missing)

    @pytest.mark.parametrize(
        "module_path",
        [
            "utils/logging_config.py",
            "data/gdelt_collector.py",
            "analysis/graph_builder.py",
            "analysis/narrator.py",
            "models/event_classifier.py",
            "dashboard/theme.py",
            "dashboard/geodata.py",
            "dashboard/blocs.py",
            "dashboard/figures.py",
            "main.py",
        ],
    )
    def test_parameters_are_annotated(
        self, project_root: Path, module_path: str
    ) -> None:
        tree = ast.parse(
            (project_root / module_path).read_text(encoding="utf-8"),
            filename=module_path,
        )
        missing: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            arguments = [
                *node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs
            ]
            for argument in arguments:
                if argument.arg in {"self", "cls"}:
                    continue
                if argument.annotation is None:
                    missing.append(f"{module_path}:{node.lineno} {node.name}({argument.arg})")
        assert not missing, "Missing parameter annotations:\n" + "\n".join(missing)


class TestDocumentation:
    """Public callables and classes carry docstrings."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "utils/logging_config.py",
            "data/gdelt_collector.py",
            "analysis/graph_builder.py",
            "analysis/narrator.py",
            "models/event_classifier.py",
            "dashboard/theme.py",
            "dashboard/geodata.py",
            "dashboard/blocs.py",
            "dashboard/figures.py",
            "dashboard/app.py",
            "main.py",
        ],
    )
    def test_module_and_definitions_are_documented(
        self, project_root: Path, module_path: str
    ) -> None:
        tree = ast.parse(
            (project_root / module_path).read_text(encoding="utf-8"),
            filename=module_path,
        )
        assert ast.get_docstring(tree), f"{module_path} has no module docstring"

        # Only module-level definitions and class members form the public
        # surface. Nested closures are implementation detail of their parent.
        definitions: list[ast.AST] = list(tree.body)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                definitions.extend(node.body)

        undocumented = [
            f"{module_path}:{node.lineno} {node.name}"
            for node in definitions
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and not node.name.startswith("_")
            and ast.get_docstring(node) is None
        ]
        assert not undocumented, "Undocumented definitions:\n" + "\n".join(undocumented)


class TestPalette:
    """The dashboard uses the specified enterprise palette."""

    def test_core_colors(self) -> None:
        from dashboard import theme

        assert theme.BACKGROUND == "#0a0e1a"
        assert theme.SURFACE == "#111827"
        assert theme.ACCENT == "#00d4ff"
        assert theme.ACCENT_ALT == "#ff6b35"
        assert theme.POSITIVE == "#22c55e"
        assert theme.NEGATIVE == "#ef4444"

    def test_tone_color_mapping(self) -> None:
        from dashboard import theme

        assert theme.tone_color(0.5) == theme.POSITIVE
        assert theme.tone_color(-0.5) == theme.NEGATIVE
        assert theme.tone_color(0.0) == theme.CAUTION

    def test_stylesheet_defines_the_palette_variables(self) -> None:
        from dashboard import theme

        for variable in ("--bg", "--surface", "--accent", "--positive", "--negative"):
            assert variable in theme.GLOBAL_CSS

"""Read-only, path-confined Markdown retrieval independent of any web framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


class WorkspaceError(RuntimeError):
    """The requested workspace operation was invalid or unsafe."""


class ContextLimitExceeded(WorkspaceError):
    """A tool result would exceed an explicit context ceiling."""


@dataclass(frozen=True)
class WorkspaceLimits:
    """Limits for paths, searches, and text entering model context."""

    max_total_chars: int = 600_000
    max_single_tool_chars: int = 100_000
    max_list_results: int = 500
    max_search_results: int = 80
    max_search_query_chars: int = 300

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass
class ScopedMarkdownWorkspace:
    """Expose Markdown under exactly one resolved root.

    The class has no concept of visibility tiers. A host application creates
    one instance with whichever directory the caller is permitted to read.
    """

    root: Path
    limits: WorkspaceLimits = field(default_factory=WorkspaceLimits)
    source_paths: list[str] = field(default_factory=list, init=False)
    total_chars: int = field(default=0, init=False)
    _seen_paths: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {self.root}")

    def _resolve(self, relative_path: str) -> Path:
        raw = (relative_path or "").strip().replace("\\", "/")
        posix = PurePosixPath(raw)
        if not raw or posix.is_absolute() or ".." in posix.parts:
            raise WorkspaceError("path must be relative and cannot contain '..'")
        try:
            resolved = (self.root / Path(*posix.parts)).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError(f"file does not exist: {raw}") from exc
        if self.root not in resolved.parents or not resolved.is_file():
            raise WorkspaceError("path escaped the workspace root")
        return resolved

    @staticmethod
    def _validate_pattern(pattern: str) -> str:
        clean = (pattern or "*.md").strip().replace("\\", "/")
        posix = PurePosixPath(clean)
        if posix.is_absolute() or ".." in posix.parts:
            raise WorkspaceError("glob must remain inside the workspace")
        return clean

    def _record(self, paths: list[Path]) -> None:
        for path in paths:
            absolute = str(path)
            if absolute not in self._seen_paths:
                self._seen_paths.add(absolute)
                self.source_paths.append(absolute)

    def _charge(self, text: str) -> str:
        if len(text) > self.limits.max_single_tool_chars:
            raise ContextLimitExceeded(
                f"one tool result is {len(text)} characters; limit is "
                f"{self.limits.max_single_tool_chars}. Use search or narrow the query."
            )
        projected = self.total_chars + len(text)
        if projected > self.limits.max_total_chars:
            raise ContextLimitExceeded(
                f"tool results would total {projected} characters; "
                f"run limit is {self.limits.max_total_chars}"
            )
        self.total_chars = projected
        return text

    def _markdown_files(self) -> list[Path]:
        files: list[Path] = []
        for candidate in self.root.rglob("*.md"):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if self.root not in resolved.parents:
                raise WorkspaceError("path escaped the workspace root")
            files.append(resolved)
        return sorted(
            files,
            key=lambda path: (
                path.name != "INDEX.md",
                path.relative_to(self.root).as_posix(),
            ),
        )

    def list_files(self, pattern: str = "*.md") -> str:
        """List matching relative paths without reading file contents."""
        clean = self._validate_pattern(pattern)
        matches: list[str] = []
        for path in self._markdown_files():
            relative = path.relative_to(self.root).as_posix()
            if relative == "INDEX.md" or PurePosixPath(relative).match(clean):
                matches.append(relative)
            if len(matches) >= self.limits.max_list_results:
                break
        capped = len(matches) == self.limits.max_list_results
        suffix = "\n[listing capped]" if capped else ""
        return self._charge("\n".join(matches) + suffix or "[no matching files]")

    def read_file(self, relative_path: str) -> str:
        """Read one complete Markdown file and record it as a source."""
        path = self._resolve(relative_path)
        if path.suffix.lower() != ".md":
            raise WorkspaceError("only Markdown files may be read")
        body = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(self.root).as_posix()
        result = self._charge(f'<file path="{relative}">\n{body}\n</file>')
        self._record([path])
        return result

    def search(self, query: str, pattern: str = "*.md") -> str:
        """Return bounded matching lines and record files whose text is returned."""
        needle = (query or "").strip()
        if not needle:
            raise WorkspaceError("search query is required")
        if len(needle) > self.limits.max_search_query_chars:
            raise WorkspaceError(
                f"search query exceeds {self.limits.max_search_query_chars} characters"
            )
        clean = self._validate_pattern(pattern)
        lowered = needle.casefold()
        lines: list[str] = []
        sources: list[Path] = []

        for path in self._markdown_files():
            relative = path.relative_to(self.root).as_posix()
            if not PurePosixPath(relative).match(clean):
                continue
            matched_file = False
            for number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if lowered not in line.casefold():
                    continue
                lines.append(f"{relative}:{number}: {line[:500]}")
                matched_file = True
                if len(lines) >= self.limits.max_search_results:
                    break
            if matched_file:
                sources.append(path)
            if len(lines) >= self.limits.max_search_results:
                break

        capped = len(lines) == self.limits.max_search_results
        suffix = "\n[search results capped]" if capped else ""
        result = self._charge("\n".join(lines) + suffix or "[no matches]")
        self._record(sources)
        return result

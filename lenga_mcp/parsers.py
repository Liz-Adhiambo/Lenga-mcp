import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lenga_mcp.chunking import HEADING_PATTERNS

try:
    import pypdf as _pypdf
    _PYPDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYPDF_AVAILABLE = False


@dataclass
class Section:
    heading: str
    level: int          # 1=Part/Chapter, 2=Title/Division, 3=Section/Article, 4=Subsection
    content: str
    parent_heading: Optional[str] = None
    children: list = field(default_factory=list)


@dataclass
class ParsedDocument:
    document_id: str
    source_path: str
    file_type: str      # "pdf" | "json" | "csv" | "txt"
    full_text: str
    sections: list[Section]
    metadata: dict


def get_file_metadata(file_path: str) -> dict:
    p = Path(file_path)
    stat = p.stat()
    return {
        "document_id": p.stem,
        "name": p.name,
        "type": p.suffix.lstrip(".").lower(),
        "size_bytes": stat.st_size,
        "last_modified": stat.st_mtime,
    }


def parse_document(file_path: str) -> ParsedDocument:
    """Route to the correct parser based on file extension."""
    ext = Path(file_path).suffix.lower()
    dispatch = {
        ".pdf": parse_pdf,
        ".json": parse_json,
        ".csv": parse_csv,
        ".txt": parse_text,
    }
    parser = dispatch.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(dispatch)}")
    return parser(file_path)


def parse_text(file_path: str) -> ParsedDocument:
    p = Path(file_path)
    text = p.read_text(encoding="utf-8", errors="replace")
    sections = _build_section_tree_from_text(text)
    return ParsedDocument(
        document_id=p.stem,
        source_path=str(p),
        file_type="txt",
        full_text=text,
        sections=sections,
        metadata={"size_bytes": p.stat().st_size},
    )


def parse_pdf(file_path: str) -> ParsedDocument:
    if not _PYPDF_AVAILABLE:
        raise RuntimeError("pypdf is not installed. Run: pip install pypdf")

    p = Path(file_path)
    reader = _pypdf.PdfReader(str(p))

    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        extracted = page.extract_text() or ""
        pages.append(f"--- Page {i + 1} ---\n{extracted}")

    full_text = "\n".join(pages)
    sections = _build_section_tree_from_text(full_text)

    return ParsedDocument(
        document_id=p.stem,
        source_path=str(p),
        file_type="pdf",
        full_text=full_text,
        sections=sections,
        metadata={"page_count": len(reader.pages), "size_bytes": p.stat().st_size},
    )


def parse_json(file_path: str) -> ParsedDocument:
    p = Path(file_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    full_text = json.dumps(data, indent=2)
    sections = _flatten_json_to_sections(data)
    return ParsedDocument(
        document_id=p.stem,
        source_path=str(p),
        file_type="json",
        full_text=full_text,
        sections=sections,
        metadata={"size_bytes": p.stat().st_size},
    )


def parse_csv(file_path: str) -> ParsedDocument:
    p = Path(file_path)
    sections: list[Section] = []
    rows_text: list[str] = []

    with p.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        parent_context = "Columns: " + ", ".join(headers)

        for i, row in enumerate(reader):
            content = "\n".join(f"{k}: {v}" for k, v in row.items())
            first_val = next(iter(row.values()), f"row_{i + 1}")
            heading = f"Row {i + 1}: {first_val}"
            sections.append(Section(
                heading=heading,
                level=1,
                content=content,
                parent_heading=parent_context,
            ))
            rows_text.append(content)

    full_text = f"{parent_context}\n\n" + "\n\n".join(rows_text)
    return ParsedDocument(
        document_id=p.stem,
        source_path=str(p),
        file_type="csv",
        full_text=full_text,
        sections=sections,
        metadata={"row_count": len(sections), "size_bytes": p.stat().st_size},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_heading_level(line: str) -> Optional[int]:
    stripped = line.strip()
    if not stripped:
        return None
    for pattern, level, _ in HEADING_PATTERNS:
        if pattern.match(stripped):
            return level
    return None


def _build_section_tree_from_text(text: str) -> list[Section]:
    """Detect headings via regex, split text into a flat Section list with levels."""
    lines = text.splitlines()
    sections: list[Section] = []
    current_heading: Optional[str] = None
    current_level = 1
    current_lines: list[str] = []

    for line in lines:
        level = _detect_heading_level(line)
        if level is not None:
            if current_heading is not None:
                sections.append(Section(
                    heading=current_heading,
                    level=current_level,
                    content="\n".join(current_lines).strip(),
                ))
            current_heading = line.strip()
            current_level = level
            current_lines = []
        else:
            if line.strip():
                current_lines.append(line)

    # Flush the final section
    if current_heading is not None:
        sections.append(Section(
            heading=current_heading,
            level=current_level,
            content="\n".join(current_lines).strip(),
        ))

    # No headings detected — treat entire text as one section
    if not sections and text.strip():
        sections.append(Section(heading="Document", level=1, content=text.strip()))

    return sections


def _flatten_json_to_sections(
    data, prefix: str = "", level: int = 1
) -> list[Section]:
    """Recursively walk JSON, emitting each leaf as a Section with breadcrumb parent path."""
    sections: list[Section] = []

    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix} > {key}" if prefix else key
            if isinstance(value, (dict, list)):
                sections.extend(_flatten_json_to_sections(value, path, level + 1))
            else:
                sections.append(Section(
                    heading=key,
                    level=level,
                    content=str(value),
                    parent_heading=prefix or None,
                ))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            path = f"{prefix}[{i}]"
            if isinstance(item, (dict, list)):
                sections.extend(_flatten_json_to_sections(item, path, level + 1))
            else:
                sections.append(Section(
                    heading=f"Item {i}",
                    level=level,
                    content=str(item),
                    parent_heading=prefix or None,
                ))

    return sections

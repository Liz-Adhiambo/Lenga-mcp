import re
from dataclasses import dataclass, field

# Ordered most-to-least specific. Each tuple: (compiled_regex, level, name).
# Level 1 = Part/Chapter, 2 = Title/Division, 3 = Section/Article, 4 = Subsection.
HEADING_PATTERNS: list[tuple[re.Pattern, int, str]] = [
    (re.compile(r"^\s*\d+\.\d+\.\d+\s+\S"), 4, "subsubsection"),
    (re.compile(r"^\s*\d+\.\d+\s+\S"), 3, "subsection_numbered"),
    (re.compile(r"^\s*PART\s+[IVXLCDM]+\b", re.IGNORECASE), 1, "roman_part"),
    (re.compile(r"^\s*PART\s+\d+\b", re.IGNORECASE), 1, "numeric_part"),
    (re.compile(r"^\s*CHAPTER\s+\d+\b", re.IGNORECASE), 1, "chapter"),
    (re.compile(r"^\s*SCHEDULE\s+\d+\b", re.IGNORECASE), 1, "schedule"),
    (re.compile(r"^\s*Section\s+\d+[:\-]", re.IGNORECASE), 3, "section_keyword"),
    (re.compile(r"^\s*Article\s+\d+[:\-]", re.IGNORECASE), 3, "article_keyword"),
    (re.compile(r"^\s*\d+\.\s+[A-Z][A-Z\s]{4,}$"), 2, "numbered_title"),
    (re.compile(r"^[A-Z][A-Z\s]{5,}$"), 2, "all_caps_heading"),
]

STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "of", "in",
    "to", "and", "or", "for", "with", "that", "this", "on",
    "at", "by", "from", "be", "has", "have", "had", "it", "as",
    "which", "shall", "any", "such", "may",
})


@dataclass
class Chunk:
    content: str
    heading: str
    parent_path: list[str]
    document_id: str
    score: float = 0.0


def build_chunks_from_sections(sections: list, document_id: str) -> list[Chunk]:
    """Convert a flat Section list into Chunk objects with resolved parent_path ancestry."""
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []  # (level, heading)

    for section in sections:
        # Pop siblings and children we have moved past
        while stack and stack[-1][0] >= section.level:
            stack.pop()

        parent_path = [h for (_, h) in stack]

        # For JSON/CSV sections the stack stays flat (all leaves at same level).
        # Fall back to the breadcrumb stored in parent_heading when available.
        if not parent_path and getattr(section, "parent_heading", None):
            parent_path = [p.strip() for p in section.parent_heading.split(" > ")]

        stack.append((section.level, section.heading))

        chunks.append(Chunk(
            content=section.content,
            heading=section.heading,
            parent_path=parent_path,
            document_id=document_id,
        ))

    return chunks


def _tokenize(text: str) -> set[str]:
    strip_chars = ".,;:()[]\"'—-–/\\"
    return {w.strip(strip_chars) for w in text.lower().split() if w.strip(strip_chars)}


def score_chunks(chunks: list[Chunk], query: str) -> list[Chunk]:
    """Score chunks by keyword overlap against query. Mutates chunk.score, returns sorted list."""
    query_tokens = _tokenize(query) - STOP_WORDS

    if not query_tokens:
        return chunks

    for chunk in chunks:
        combined = f"{chunk.heading} {' '.join(chunk.parent_path)} {chunk.content}"
        chunk_tokens = _tokenize(combined)
        heading_tokens = _tokenize(chunk.heading)
        path_tokens = _tokenize(" ".join(chunk.parent_path))

        overlap = len(query_tokens & chunk_tokens)
        heading_bonus = 2 * len(query_tokens & heading_tokens)
        path_bonus = 1.5 * len(query_tokens & path_tokens)
        chunk.score = (overlap + heading_bonus + path_bonus) / len(query_tokens)

    return sorted(chunks, key=lambda c: c.score, reverse=True)


def get_best_chunk(chunks: list[Chunk], query: str, top_n: int = 1) -> list[Chunk]:
    """Return the top_n highest-scoring chunks. Falls back to positional order if all scores are 0."""
    ranked = score_chunks(chunks, query)
    if all(c.score == 0.0 for c in ranked):
        return chunks[:top_n]
    return ranked[:top_n]


def format_chunk_with_path(chunk: Chunk) -> str:
    """Render a Chunk as a structured plain-text block with full ancestor path."""
    full_path = " > ".join(chunk.parent_path + [chunk.heading]) if chunk.parent_path else chunk.heading
    return (
        f"[PATH] {full_path}\n"
        f"[HEADING] {chunk.heading}\n"
        f"[CONTENT]\n{chunk.content}"
    )

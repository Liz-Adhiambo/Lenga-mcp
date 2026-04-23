import json
from pathlib import Path

from fastmcp import FastMCP

from lenga_mcp.chunking import build_chunks_from_sections, format_chunk_with_path, get_best_chunk
from lenga_mcp.parsers import get_file_metadata, parse_document

LOCAL_CONTEXT_DIR = Path(__file__).parent.parent / "local_context"
SUPPORTED_EXTENSIONS = {".pdf", ".json", ".csv", ".txt"}

NUANCE_WRAPPER_TEMPLATE = """\
=== CONTEXT INTERPRETATION DIRECTIVE ===
Document: {document_id}
Region: Sub-Saharan Africa / East Africa

You are about to receive an extract from a local document repository. Before \
interpreting the content below, apply the following socio-economic lenses:

1. INFORMAL ECONOMY LENS: A significant portion of the working population \
operates outside formal employment structures. Statistics on wages, employment, \
or business formation must be read with informal sector participation in mind.

2. MOBILE-FIRST FINANCE LENS: Financial references (transactions, transfers, \
savings, credit) occur predominantly through mobile money infrastructure \
(M-Pesa, Airtel Money, MTN MoMo). "Bank account" does not represent the \
primary financial access point for most citizens.

3. LAND TENURE LENS: Land rights may be customary, communal, or formally \
registered. Statutory frameworks coexist with traditional systems; community \
and clan ownership patterns are legally and practically significant.

4. MULTILINGUAL CONTEXT LENS: Legal and regulatory language is formal English \
or French, but implementation, dispute resolution, and community understanding \
occur in Swahili, Amharic, Yoruba, Zulu, and many other languages. Terminology \
gaps between official text and lived experience are common.

5. INFRASTRUCTURE CONSTRAINT LENS: References to digital systems, supply chains, \
healthcare delivery, or public services operate under intermittent connectivity, \
unreliable power, and road network constraints that affect policy interpretation.

Interpret all figures, legal language, rights, and obligations through these \
lenses before responding.
=== END DIRECTIVE ===

"""

mcp = FastMCP(
    name="Lenga-MCP",
    instructions=(
        "Context-Bridge: provides grounded access to regional African document repositories. "
        "Use the documents:// resource to browse available files, or documents://{document_id} "
        "to read a specific file. Use get_optimized_context() to retrieve the most relevant "
        "passage for a query, complete with hierarchical context and socio-economic "
        "interpretation guidance. Runs fully offline — no cloud APIs required."
    ),
)


def _resolve_document_path(document_id: str) -> Path:
    """Find the file in LOCAL_CONTEXT_DIR whose stem matches document_id."""
    for ext in SUPPORTED_EXTENSIONS:
        candidate = LOCAL_CONTEXT_DIR / f"{document_id}{ext}"
        if candidate.exists():
            return candidate
    available = [p.stem for p in LOCAL_CONTEXT_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS] \
        if LOCAL_CONTEXT_DIR.exists() else []
    raise FileNotFoundError(
        f"Document '{document_id}' not found. Available documents: {available or 'none (local_context/ is empty)'}"
    )


@mcp.resource(
    "documents://",
    name="document_listing",
    description="Lists all documents in the local_context/ directory with their metadata (id, name, type, size).",
    mime_type="application/json",
)
def list_documents() -> str:
    if not LOCAL_CONTEXT_DIR.exists():
        return json.dumps([])

    docs = []
    for entry in sorted(LOCAL_CONTEXT_DIR.iterdir()):
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append(get_file_metadata(str(entry)))

    return json.dumps(docs, indent=2)


@mcp.resource(
    "documents://{document_id}",
    name="document_content",
    description=(
        "Returns the full plain-text content of a document. "
        "document_id is the filename without extension (e.g. 'employment_act_kenya')."
    ),
    mime_type="text/plain",
)
def get_document(document_id: str) -> str:
    try:
        file_path = _resolve_document_path(document_id)
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from exc

    parsed = parse_document(str(file_path))
    return parsed.full_text


@mcp.tool(
    description=(
        "Parse a document and return the most relevant section for a given query, "
        "with full hierarchical path context (e.g. 'PART III > Section 22: Wages') "
        "and a socio-economic interpretation directive. "
        "document_id is the filename without extension (e.g. 'employment_act_kenya'). "
        "Uses keyword overlap scoring — no embeddings, runs fully offline."
    ),
)
def get_optimized_context(query: str, document_id: str) -> str:
    try:
        file_path = _resolve_document_path(document_id)
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from exc

    parsed = parse_document(str(file_path))
    nuance = NUANCE_WRAPPER_TEMPLATE.format(document_id=document_id)

    if not parsed.sections:
        return nuance + parsed.full_text[:4000]

    chunks = build_chunks_from_sections(parsed.sections, document_id)

    if not chunks:
        return nuance + parsed.full_text[:4000]

    best = get_best_chunk(chunks, query, top_n=1)[0]
    return nuance + format_chunk_with_path(best)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

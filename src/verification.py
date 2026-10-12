import re

# Matches source citations such as [S1] or [S12].
CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


def validate_citations(answer: str, sources: list[dict]) -> dict:
    """Check whether cited source IDs exist in the retrieved sources."""

    available_ids = {source["id"] for source in sources}
    cited_ids = set(CITATION_PATTERN.findall(answer))

    invalid_ids = cited_ids - available_ids

    return {
        "valid": len(invalid_ids) == 0,
        "cited_ids": sorted(cited_ids),
        "invalid_ids": sorted(invalid_ids),
        "has_citations": len(cited_ids) > 0,
    }

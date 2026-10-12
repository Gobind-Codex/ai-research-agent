from src.verification import validate_citations


def test_valid_citations():
    answer = "The paper discusses retrieval [S1] and evaluation [S2]."

    sources = [
        {"id": "S1"},
        {"id": "S2"},
    ]

    result = validate_citations(answer, sources)

    assert result["valid"] is True
    assert result["has_citations"] is True
    assert result["invalid_ids"] == []


def test_invalid_citation():
    answer = "The paper discusses deployment [S9]."

    sources = [
        {"id": "S1"},
        {"id": "S2"},
    ]

    result = validate_citations(answer, sources)

    assert result["valid"] is False
    assert result["invalid_ids"] == ["S9"]


def test_missing_citations():
    answer = "The paper discusses deployment."

    sources = [{"id": "S1"}]

    result = validate_citations(answer, sources)

    assert result["valid"] is True
    assert result["has_citations"] is False

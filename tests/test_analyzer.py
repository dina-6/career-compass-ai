from src.analyzer import extract_skills, skill_gap

def test_extract_skills_is_case_insensitive():
    assert "python" in extract_skills("PYTHON and SQL")

def test_skill_gap_returns_missing_skills():
    result = skill_gap("I know Python and SQL", "Data Scientist")
    assert "python" in result["matched"]
    assert "statistics" in result["missing"]

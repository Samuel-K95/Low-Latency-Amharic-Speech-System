"""
Phase 7 delivery validation (no GPU / no notebook execution).
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT.parent
NB = FINAL / "amharic-speech-to-speech-rag-api-server (11).ipynb"

sys.path.insert(0, str(ROOT))
from server.deps_check import REQUIRED_VERSIONS


def test_notebook_json_valid_utf8():
    raw = NB.read_text(encoding="utf-8")
    nb = json.loads(raw)
    assert nb.get("nbformat") == 4
    assert len(nb["cells"]) >= 14


def test_notebook_code_outputs_cleared():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            assert cell.get("outputs") == [], f"cell {i} still has outputs"
            assert cell.get("execution_count") is None


def test_notebook_dep_pins_match_deps_check():
    text = NB.read_text(encoding="utf-8")
    # Prefer notebook (12) if present
    nb12 = FINAL / "amharic-speech-to-speech-rag-api-server (12).ipynb"
    if nb12.exists():
        text = nb12.read_text(encoding="utf-8")
    for pkg, ver in REQUIRED_VERSIONS.items():
        if pkg == "transformers":
            assert "transformers>=5.3.0" in text
        else:
            assert f"{pkg}=={ver}" in text or ver in text
    assert "deps_check" in text or "assert_dependency_versions" in text
    assert "HiggsAudioV2TokenizerModel" in text
    assert "4.57.1" not in text


def test_notebook_launches_modular_package():
    text = NB.read_text(encoding="utf-8")
    assert "from server.launcher import run_server" in text
    assert "find_project" in text
    assert "server/launcher.py" in text or "launcher.py" in text


def test_package_python_syntax():
    files = (
        list((ROOT / "server").rglob("*.py"))
        + list((ROOT / "src").rglob("*.py"))
        + list((ROOT / "tests").rglob("*.py"))
    )
    assert files
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

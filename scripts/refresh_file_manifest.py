"""Refresh the repository file inventory and source-license inventory.
Run from the repository root after copying the patch overlay.
Uses Python standard library only; does not modify models or source code.
"""
from pathlib import Path
import hashlib, json, csv

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "FILE_MANIFEST.json"
existing = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else []
paths = {row["path"] for row in existing}

# Include all files in newly introduced documentation directories and patch paths.
for rel in [
    "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "AGENTS.md",
    "docs/LICENSE_SCOPE.md", "docs/PUBLICATION_STATUS.md",
    "models/LICENSE.md", "models/README.md", "models/PROVENANCE.json",
    "LICENSES/Apache-2.0.txt",
    "LICENSES/PureJaxRL-Apache-2.0.txt",
    "LICENSES/Gymnax-Apache-2.0.txt",
    "src/network.py.license", "src/instrumented_logic.py.license",
    "scripts/refresh_file_manifest.py", "docs/SOURCE_LICENSE_INVENTORY.csv",
]:
    paths.add(rel)

source = []
for path in sorted(set(ROOT.glob("*.py")) | set(ROOT.glob("scripts/*.py")) |
                   set(ROOT.glob("src/*.py")) | set(ROOT.glob("vendor/**/*.py"))):
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith("vendor/"):
        origin = "third-party upstream"
        license_name = "MIT or Apache-2.0, see THIRD_PARTY_NOTICES.md"
    elif rel == "src/network.py" or rel == "src/instrumented_logic.py":
        origin = "upstream-derived + project modifications"
        license_name = "original licenses + PolyForm Noncommercial for independently owned modifications"
    else:
        origin = "project implementation"
        license_name = "PolyForm Noncommercial 1.0.0 (project-owned portions)"
    source.append([rel, hashlib.sha256(path.read_bytes()).hexdigest(), origin, license_name])

inventory = ROOT / "docs/SOURCE_LICENSE_INVENTORY.csv"
inventory.parent.mkdir(parents=True, exist_ok=True)
with inventory.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "sha256", "origin", "applicable_license"])
    writer.writerows(source)

rows = []
for rel in sorted(paths):
    if rel == "FILE_MANIFEST.json":
        continue
    path = ROOT / rel
    if path.is_file():
        content = path.read_bytes()
        rows.append(dict(path=rel, bytes=len(content),
                         sha256=hashlib.sha256(content).hexdigest()))

MANIFEST.write_text(json.dumps(rows, ensure_ascii=False, indent=2)+"\n",
                    encoding="utf-8")
print(f"Updated FILE_MANIFEST.json: {len(rows)} files")
print(f"Updated SOURCE_LICENSE_INVENTORY.csv: {len(source)} source files")

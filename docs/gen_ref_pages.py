"""Generate the code reference pages and navigation."""

from pathlib import Path
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "GPyEDS"

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src.parent).with_suffix("")
    doc_path = path.relative_to(src.parent).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

import tomllib

# Generate cite.md with current version
with open(root / "pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)
    version = pyproject["project"]["version"]

cite_content = f"""# How to cite

Please cite using the following:

	norberttoth398. (2024). norberttoth398/GPyEDS: GPyEDS v{version} ({version}). Zenodo. https://doi.org/10.5281/zenodo.13837097
"""
with mkdocs_gen_files.open("cite.md", "w") as f:
    f.write(cite_content)

# Generate licence.md from root LICENSE
with open(root / "LICENSE", "r") as f:
    license_content = f.read()

with mkdocs_gen_files.open("licence.md", "w") as f:
    f.write(license_content)


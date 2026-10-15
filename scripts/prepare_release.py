"""Switch only the geographic fix to prepare chronological GitHub releases."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("version", choices=["1.0.0", "1.0.1"])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
path = root / "src/basic_cleaning/run.py"
source = path.read_text()
start_marker = "    # NYC_FILTER_START"
end_marker = "    # NYC_FILTER_END\n"
if start_marker in source:
    start = source.index(start_marker)
    end = source.index(end_marker) + len(end_marker)
    source = source[:start] + source[end:]
if args.version == "1.0.1":
    marker = "    return df\n"
    block = (root / "scripts/nyc_filter.txt").read_text()
    source = source.replace(marker, block + marker, 1)
path.write_text(source)
(root / "VERSION").write_text(args.version + "\n")
print(f"Prepared {args.version}; commit, push, and publish its GitHub release.")

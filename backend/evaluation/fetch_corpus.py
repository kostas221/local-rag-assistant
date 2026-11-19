"""Recover public evaluation PDFs locally; record hashes without committing the PDFs."""
import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import requests


def fetch(destination: Path, sources_path: Path | None = None, pin: bool = False):
    """sources_path: default το corpus_sources.json (7 άρθρα cloud)· το heldout_sources.json για το κλειστό
    σετ (19/11/2026). pin: άρθρο με "sha256": null κατεβαίνει ΜΙΑ φορά και το SHA-256 του γράφεται πίσω
    στο αρχείο πηγών -> από εκεί και πέρα κάθε λήψη ελέγχεται απέναντί του (commit το αρχείο)."""
    sources_path = sources_path or Path(__file__).with_name("corpus_sources.json")
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    pinned_now = []
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for name, source in sources.items():
        url = source["url"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.pdf", name):
            raise ValueError("Invalid corpus filename")
        if not url.startswith("https://"):
            raise ValueError("Corpus downloads require HTTPS")
        target = destination / name
        if target.exists():
            content = target.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if name in manifest and manifest[name]["sha256"] != digest:
                raise ValueError(f"Local file changed: {name}")
        else:
            with requests.get(url, headers={"User-Agent": "Z-AI-evaluation-corpus/1.0"},
                              timeout=(10, 60), stream=True) as response:
                response.raise_for_status()
                buffer = bytearray()
                for chunk in response.iter_content(1024 * 1024):
                    buffer.extend(chunk)
                    if len(buffer) > 30 * 1024 * 1024:
                        raise ValueError(f"Oversized PDF: {name}")
                content = bytes(buffer)
            if len(content) > 30 * 1024 * 1024 or not content.startswith(b"%PDF-"):
                raise ValueError(f"Invalid or oversized PDF: {name}")
            digest = hashlib.sha256(content).hexdigest()
        if not content.startswith(b"%PDF-"):
            raise ValueError(f"Not a PDF: {name}")
        if source.get("sha256") is None:
            if not pin:
                raise ValueError(f"Unpinned source (no sha256): {name} — run once with --pin")
            source["sha256"] = digest
            pinned_now.append(name)
        if digest != source["sha256"]:
            raise ValueError(f"Publisher file differs from the pinned corpus: {name}")
        if not target.exists():
            with target.open("xb") as output:
                output.write(content)
        manifest[name] = {"url": url, "sha256": digest, "bytes": len(content),
                          "checked_utc": datetime.now(UTC).isoformat()}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Verified {name}: {len(content)} bytes", flush=True)
    if pinned_now:
        sources_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Pinned {len(pinned_now)} new sources in {sources_path.name}: {', '.join(pinned_now)}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=Path(__file__).parent / "test_papers" / "cloud")
    parser.add_argument("--sources", type=Path, default=None,
                        help="sources JSON (default: corpus_sources.json; held-out: heldout_sources.json)")
    parser.add_argument("--pin", action="store_true",
                        help="record the SHA-256 of sources that have none yet (first download only)")
    args = parser.parse_args()
    fetch(args.destination, args.sources, args.pin)

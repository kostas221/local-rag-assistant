"""Docling vs pypdf: ΜΟΝΟ σύγκριση. Δεν γράφει τίποτα στη βάση.

Στόχος: να δούμε ΠΡΙΝ αλλάξουμε το ingest αν το Docling διορθώνει τις τρεις
συγκεκριμένες ζημιές που μετρήσαμε με το pypdf:
  1. σπασμένες λέξεις ("distribu ted", "A WS", "cl oud")
  2. ligatures (ﬁ ﬂ ﬀ ﬃ) που δεν ταιριάζουν με το "fi"/"ffi" της ερώτησης
  3. πίνακες που διαλύονται σε μία λέξη ανά γραμμή

Τρέχει σε ΞΕΧΩΡΙΣΤΟ container (βλ. Dockerfile δίπλα) γιατί το docling θέλει
pydantic>=2 και το backend είναι σε pydantic 1.10.

Χρήση:
    python compare_extraction.py                 # οι 2 προβληματικές σελίδες
    python compare_extraction.py --all           # όλο το corpus (αργό σε CPU)
"""
import argparse
import glob
import os
import re
import sys
import time

LIGATURES = "ﬁﬂﬀﬃﬄﬅﬆ"
REAL_SHORT = {
    "a", "i", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is",
    "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we",
    "am", "ok", "ad", "id", "ie", "eg", "et", "al", "vs", "st", "nd", "rd",
    "th", "pp", "ii", "iii", "iv", "vi", "cf", "de", "la", "el", "hz", "kb",
    "mb", "gb", "tb", "ms", "ec", "s3", "vm", "ip", "os", "io",
}

# (αρχείο, index σελίδας 0-based, keywords που ΠΡΕΠΕΙ να υπάρχουν)
TARGETS = [
    ("1902.03383v1.pdf", 12, ["filestore", "ephemeral", "iops"]),
    ("excamera-nsdi17.pdf", 14, ["undershoot-pct", "auto-alt-ref",
                                 "buf-initial-sz"]),
]


def damage(text: str) -> dict:
    toks = re.findall(r"[A-Za-z]+", text)
    broken = [t for t in toks if len(t) <= 2 and t.lower() not in REAL_SHORT]
    lines = [ln for ln in text.split("\n") if ln.strip()]
    lonely = [ln for ln in lines if len(ln.split()) <= 2]
    return {
        "chars": len(text),
        "tokens": len(toks),
        "broken": len(broken),
        "broken_pct": len(broken) / max(len(toks), 1),
        "lig": sum(text.count(c) for c in LIGATURES),
        "hyphen": len(re.findall(r"[a-zA-Z]-\n[a-z]", text)),
        "lines": len(lines),
        "lonely_pct": len(lonely) / max(len(lines), 1),
    }


def show(tag: str, d: dict):
    print(f"  {tag:<9} {d['chars']:>6} χαρ | {d['tokens']:>5} tokens | "
          f"σπασμένα {d['broken']:>4} ({d['broken_pct']:>5.1%}) | "
          f"lig {d['lig']:>3} | συλλ {d['hyphen']:>3} | "
          f"γραμμές ≤2 λέξεων {d['lonely_pct']:>5.1%}")


def pypdf_pages(path: str) -> list:
    from pypdf import PdfReader
    return [(p.extract_text() or "") for p in PdfReader(path).pages]


def docling_pages(path: str) -> list:
    """Markdown ανά σελίδα. Κρατάμε την ανά-σελίδα δομή γιατί όλο το retrieval
    (parent-document expansion) στηρίζεται στο metadata 'page'."""
    from docling.document_converter import DocumentConverter
    doc = DocumentConverter().convert(path).document
    n = len(getattr(doc, "pages", {})) or 1
    out = []
    for pno in range(1, n + 1):          # το docling μετράει σελίδες από 1
        try:
            out.append(doc.export_to_markdown(page_no=pno))
        except TypeError:
            # παλιότερο API χωρίς page_no -> γύρνα ΟΛΟ το κείμενο μία φορά
            return [doc.export_to_markdown()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="όλο το corpus αντί για τις 2 προβληματικές σελίδες")
    ap.add_argument("--pdf-dir", default="/pdfs")
    args = ap.parse_args()

    paths = {os.path.basename(p).split("_", 1)[-1]: p
             for p in sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))}
    if not paths:
        print(f"Δεν βρέθηκαν PDF στο {args.pdf_dir}")
        return 1

    targets = ([(n, None, []) for n in paths] if args.all else TARGETS)

    for name, page_idx, keywords in targets:
        if name not in paths:
            print(f"ΠΑΡΑΛΕΙΨΗ: δεν βρέθηκε {name}")
            continue
        path = paths[name]
        t0 = time.time()
        old_pages = pypdf_pages(path)
        new_pages = docling_pages(path)
        dt = time.time() - t0

        idxs = [page_idx] if page_idx is not None else range(len(old_pages))
        for i in idxs:
            old = old_pages[i] if i < len(old_pages) else ""
            new = new_pages[i] if i < len(new_pages) else ""
            print(f"\n{'=' * 76}\n{name} — σελίδα index {i}  "
                  f"({dt:.0f}s για όλο το αρχείο)\n{'=' * 76}")
            show("pypdf", damage(old))
            show("docling", damage(new))
            if keywords:
                fo = [k for k in keywords if k.lower() in old.lower()]
                fn = [k for k in keywords if k.lower() in new.lower()]
                print(f"  keywords: pypdf {len(fo)}/{len(keywords)} {fo} | "
                      f"docling {len(fn)}/{len(keywords)} {fn}")
            if page_idx is not None:
                print("\n  --- DOCLING (πρώτες 24 γραμμές) ---")
                for ln in new.split("\n")[:24]:
                    print("  |" + ln[:104])
    return 0


if __name__ == "__main__":
    sys.exit(main())

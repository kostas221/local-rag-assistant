"""Φάση 4.1: απογραφή σχημάτων — υπάρχει ΚΑΝ πρόβλημα με τα σχήματα; (0 $, 0 μοντέλα).

ΤΟ ΕΡΩΤΗΜΑ: η εξαγωγή (ai_core.ingest_pdf -> pymupdf get_text) κρατάει ΜΟΝΟ κείμενο. Ό,τι δείχνει ένα
σχήμα (καμπύλες, ράβδοι, βέλη αρχιτεκτονικής) δεν φτάνει ποτέ στο ευρετήριο — μόνο η λεζάντα και όσες
ετικέτες είναι vector κείμενο (άξονες, κουτιά), σκόρπιες. Πριν χτίσουμε multimodal ingestion: ΠΟΣΑ
σχήματα των 7 papers έχουν πληροφορία που ΔΕΝ λέει πουθενά το κείμενο;

ΤΙ ΚΑΝΕΙ: για κάθε PDF του σώματος βρίσκει κάθε λεζάντα «Figure N» / «Table N», μαζεύει τις προτάσεις του
κειμένου που αναφέρονται σε αυτό («as Figure 3 shows ...») και σώζει τη σελίδα ως εικόνα. Η κρίση «το
σχήμα λέει κάτι που το κείμενο δεν λέει» γίνεται με το μάτι, σχήμα-σχήμα, πάνω στην εικόνα + τις προτάσεις.

ΕΞΟΔΟΣ:
    runs/figures_inventory.csv    μία γραμμή ανά σχήμα/πίνακα (paper, σελίδα, είδος, αριθμός, λεζάντα,
                                  εικόνες/vector paths στη σελίδα, πόσες αναφορές στο κείμενο)
    runs/figures_inventory.json   τα ίδια + τις προτάσεις-αναφορές αυτούσιες
    runs/figures_png/             οι σελίδες με σχήμα ως PNG + όλο το εξαγόμενο κείμενο ανά paper
                                  (<paper>_text.txt) — αντίγραφα papers, ΠΟΤΕ commit (.gitignore)

ΚΑΝΟΝΑΣ ΑΠΟΦΑΣΗΣ, ΓΡΑΜΜΕΝΟΣ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (30/9/2026):
    «σχήμα-μόνο γεγονός» = κάτι που θα ρωτούσε λογικά ένας αναγνώστης (αριθμός, τάση, σύγκριση, όνομα
    εξαρτήματος, ροή) που ΦΑΙΝΕΤΑΙ στο σχήμα και ΔΕΝ υπάρχει στο εξαγόμενο κείμενο (σώμα, λεζάντα, ετικέτες).
    ≥ 10 σχήματα με σχήμα-μόνο γεγονός -> Φάση 4.2: σετ ~20 ερωτήσεων, μέτρηση του σημερινού συστήματος,
         μετά δοκιμή περιγραφής σχημάτων από το Gemini στο ingest.
    < 10 -> η Φάση 4 κλείνει εδώ: «το κείμενο κουβαλάει ήδη το μήνυμα των σχημάτων» — χωρίς αλλαγή.
ΠΡΟΒΛΕΨΗ: ~45-60 σχήματα συνολικά (τα περισσότερα γραφήματα σε PyWren / ExCamera / MapReduce / Baldini,
    διαγράμματα στα υπόλοιπα). Το ΜΗΝΥΜΑ του σχήματος («κλιμακώνεται γραμμικά») το λέει το κείμενο στα
    ~70%· οι ΑΚΡΙΒΕΙΣ τιμές (σημεία καμπύλης) λείπουν από το κείμενο στα περισσότερα γραφήματα -> εκτιμώ
    12-20 σχήματα με σχήμα-μόνο γεγονός, άρα συνέχεια στη 4.2 — με την επιφύλαξη ότι «διάβασε μια τιμή
    από την καμπύλη» είναι σπάνια πραγματική ερώτηση χρήστη.

ΑΠΟΤΕΛΕΣΜΑ (30/9/2026, 0 $): 33 σχήματα + 20 πίνακες σε 7 papers· κάθε σχήμα κρίθηκε με το μάτι πάνω
    στη σελίδα ΚΑΙ στο εξαγόμενο κείμενο όλου του paper -> runs/figures_verdict.csv.
        ισχυρό 1 · μέτριο 2 · μέτριο-ασθενές 4 · ασθενές 4 (το κείμενο ουσιαστικά το λέει) · όχι 22
    -> 3 καθαρά, 7 με γενναιόδωρη μέτρηση: ΚΑΤΩ από το 10. Η Φάση 4 ΚΛΕΙΝΕΙ, καμία αλλαγή.
    Γιατί: οι συγγραφείς γράφουν στο κείμενο το μήνυμα ΚΑΙ τους αριθμούς κάθε σχήματος (MapReduce 4/4,
    Berkeley 2009 3/3 πλήρως καλυμμένα· ExCamera: 60-300×, 2%/9%, 9 TFLOPS όλα στο κείμενο). Και τα
    vector διαγράμματα εξάγουν ήδη τις ετικέτες τους ως λέξεις (OpenWhisk: Controller / Invoker /
    Executor … όλα στο κείμενο)· αόρατα μένουν μόνο τα βέλη, τα 6 raster σχήματα και τα γραφήματα με
    κείμενο-ως-μονοπάτια (Berkeley View Σχ. 4). Το καθαρότερο κενό: το κόστος σε $ του sort 1TB (PyWren
    Σχ. 5) υπάρχει ΜΟΝΟ στο γράφημα — στο κείμενο μόνο οι υποδιαιρέσεις του άξονα.
    Πρόβλεψη: λάθος και στα δύο (45-60 -> 33 σχήματα · 12-20 -> 3-7).
    ΟΡΙΟ: κρίση ενός αναθεωρητή, όχι μέτρηση της εφαρμογής· ισχύει για ΑΥΤΟ το σώμα (slides, σκαναρισμένα
    ή αναφορές γεμάτες γραφήματα θα έδιναν άλλη απάντηση).

ΤΡΕΞΙΜΟ (από τη ρίζα του repo, Git Bash):
    docker compose run --rm --no-deps backend python -u evaluation/probe_figures_inventory.py
"""
import csv
import json
import os
import re

import pymupdf

PDF_DIR = "evaluation/test_papers/cloud"
OUT_CSV = "evaluation/runs/figures_inventory.csv"
OUT_JSON = "evaluation/runs/figures_inventory.json"
PNG_DIR = "evaluation/runs/figures_png"
DPI = 100

# λεζάντα = γραμμή που ΞΕΚΙΝΑΕΙ με «Figure 3:» / «Fig. 3.» / «Table 2:» (ή σκέτο «Figure 3» στο τέλος
# γραμμής, ή «Fig. 3 Serverless ...» με κεφαλαίο — στυλ Springer, Baldini). Η αναφορά μέσα σε πρόταση
# («Figure 3 shows») συνεχίζει με πεζό ή δεν ξεκινάει γραμμή.
CAPTION = re.compile(r"^\s*(Figure|Fig\.|Table)\s*(\d+)\s*([:.]|$|(?=[A-Z]))")
MENTION = re.compile(r"\b(Figures?|Figs?\.|Tables?)\s*(\d+)((?:\s*(?:,|and|&)\s*\d+)*)", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _kind(word: str) -> str:
    return "table" if word.lower().startswith("tab") else "figure"


def scan_paper(path: str) -> tuple[list[dict], dict[tuple[str, int], list[str]]]:
    name = os.path.basename(path)
    found: dict[tuple[str, int], dict] = {}
    body: list[str] = []
    with pymupdf.open(path) as doc:
        for pno, page in enumerate(doc, start=1):
            n_img = len(page.get_images(full=True))
            n_draw = len(page.get_drawings())
            for block in page.get_text("blocks"):
                if block[6] != 0:                      # 1 = εικόνα, όχι κείμενο
                    continue
                lines = block[4].strip().splitlines()
                hit = None
                for li, line in enumerate(lines):
                    m = CAPTION.match(line)
                    if m:
                        hit = (li, m)
                        break
                if hit is None:
                    body.append(" ".join(lines))
                    continue
                li, m = hit
                if li:                                 # κείμενο πριν τη λεζάντα στο ίδιο block
                    body.append(" ".join(lines[:li]))
                key = (_kind(m.group(1)), int(m.group(2)))
                caption = " ".join(lines[li:]).strip()
                prev = found.get(key)
                # ίδιος αριθμός δύο φορές: κράτα τη λεζάντα στην ΑΡΧΗ block (η άλλη είναι συνήθως
                # αναφορά σε αλλαγή γραμμής «... in\nFigure 3. The ...»)
                if prev is None or (prev["line_in_block"] > 0 and li == 0):
                    found[key] = {"paper": name, "page": pno, "kind": key[0], "no": key[1],
                                  "caption": caption[:400], "line_in_block": li,
                                  "images_on_page": n_img, "vector_paths_on_page": n_draw}
                    if prev is not None:
                        body.append(prev["caption"])
                else:
                    body.append(caption)

    refs: dict[tuple[str, int], list[str]] = {}
    for sent in SENT_SPLIT.split(" ".join(body)):
        for m in MENTION.finditer(sent):
            nums = [int(m.group(2))] + [int(n) for n in re.findall(r"\d+", m.group(3) or "")]
            for n in nums:
                refs.setdefault((_kind(m.group(1)), n), []).append(sent.strip()[:500])
    return sorted(found.values(), key=lambda r: (r["kind"], r["no"])), refs


def main() -> int:
    os.makedirs(PNG_DIR, exist_ok=True)
    rows = []
    for fn in sorted(os.listdir(PDF_DIR)):
        if not fn.endswith(".pdf"):
            continue
        path = os.path.join(PDF_DIR, fn)
        items, refs = scan_paper(path)
        stem = os.path.splitext(fn)[0]
        with pymupdf.open(path) as doc:
            for p in sorted({it["page"] for it in items if it["kind"] == "figure"}):
                doc[p - 1].get_pixmap(dpi=DPI).save(os.path.join(PNG_DIR, f"{stem}_p{p:02d}.png"))
            # ό,τι «βλέπει» το ευρετήριο (ίδια κλήση με το ingest, χωρίς το normalize): για τον έλεγχο
            # «το λέει κάπου το κείμενο;» σε ΟΛΟ το paper, όχι μόνο στις προτάσεις με «Figure N»
            with open(os.path.join(PNG_DIR, f"{stem}_text.txt"), "w", encoding="utf-8") as fh:
                for pno, page in enumerate(doc, start=1):
                    fh.write(f"\n===== σελίδα {pno} =====\n{page.get_text()}")
        orphans = sorted(k for k in refs if k not in {(it["kind"], it["no"]) for it in items})
        for it in items:
            it["refs"] = list(dict.fromkeys(refs.get((it["kind"], it["no"]), [])))
            it["n_refs"] = len(it["refs"])
            it["png"] = f"{stem}_p{it['page']:02d}.png" if it["kind"] == "figure" else ""
            rows.append(it)
        n_fig = sum(1 for it in items if it["kind"] == "figure")
        n_tab = len(items) - n_fig
        no_ref = [f"{it['kind'][0].upper()}{it['no']}" for it in items if not it["n_refs"]]
        print(f"{fn:<28} σχήματα {n_fig:>3} · πίνακες {n_tab:>2} · χωρίς αναφορά στο κείμενο: "
              f"{', '.join(no_ref) or '-'}"
              + (f" · αναφέρονται αλλά ΔΕΝ βρέθηκε λεζάντα: "
                 f"{', '.join(f'{k[0][0].upper()}{k[1]}' for k in orphans)}" if orphans else ""))

    figs = [r for r in rows if r["kind"] == "figure"]
    print(f"\nΣΥΝΟΛΟ: {len(figs)} σχήματα σε {len({(r['paper'], r['page']) for r in figs})} σελίδες · "
          f"{len(rows) - len(figs)} πίνακες · σχήματα με raster εικόνα στη σελίδα: "
          f"{sum(1 for r in figs if r['images_on_page'])}")

    fields = ["paper", "page", "kind", "no", "n_refs", "images_on_page", "vector_paths_on_page",
              "png", "caption"]
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    print(f"Γράφτηκαν: {OUT_CSV} · {OUT_JSON} · {PNG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

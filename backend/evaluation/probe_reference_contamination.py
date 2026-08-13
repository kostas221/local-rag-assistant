"""Μολύνει η ΒΙΒΛΙΟΓΡΑΦΙΑ τα chunks; — ΜΗΔΕΝ κόστος API, ΜΗΔΕΝ re-ingest.

ΤΟ ΕΥΡΗΜΑ ΠΟΥ ΤΟ ΓΕΝΝΗΣΕ (12/8/2026)
------------------------------------
Η σελίδα-στόχος του `h002` (`excamera-nsdi17.pdf:15`) είναι **1313 χαρακτήρες
σε ΕΝΑ chunk** και περιέχει δύο τελείως διαφορετικά πράγματα:

    [31] YU, Y., AND ANASTASSIOU, D. Software implementation of MPEG-II ...
    [32] ZAHARIA, M., ... Resilient Distributed Datasets ... NSDI (Apr. 2012).

    Appendix: vpxenc command lines
    vpxenc --codec=vp8 \\ --buf-initial-sz=10000 \\ --undershoot-pct=100 \\ ...

Δηλαδή **ουρά βιβλιογραφίας + shell command**. Η υπόθεση εδώ είναι ότι το
κυρίαρχο σημασιολογικό περιεχόμενο του chunk είναι οι αναφορές σε ΑΣΧΕΤΑ paper
(MPEG-II encoding, Spark/RDD) και ότι αυτό εξηγεί ταυτόχρονα:

  · γιατί το dense βάζει τη σελίδα στη **θέση 15** (και κόβεται στο RRF)
  · γιατί ο cross-encoder της δίνει **−1.97** ΑΚΟΜΑ ΚΑΙ με τη σωστή q028

ΓΙΑΤΙ ΑΞΙΖΕΙ ΞΕΧΩΡΙΣΤΗ ΜΕΤΡΗΣΗ
------------------------------
Είναι ο **μόνος** υποψήφιος μηχανισμός με **απαριθμήσιμη ακτίνα επίδρασης**.
Κάθε άλλη ιδέα που δοκιμάστηκε (κατώφλια, prompts, enrichment, decomposition,
RRF slots, μεγαλύτερος reranker) άλλαζε κάτι που ισχύει για **ΚΑΘΕ** ερώτηση,
γι' αυτό και κάθε φορά κάτι άλλο χαλούσε. Ένα καθάρισμα βιβλιογραφίας στο
ingestion αγγίζει **μόνο** τα chunks που περιέχουν βιβλιογραφία — και αυτά τα
μετράει και τα τυπώνει το ΜΕΡΟΣ Α, ώστε να ελεγχθούν με το μάτι.

ΤΙ ΜΕΤΡΑΕΙ
----------
ΜΕΡΟΣ Α — ΑΚΤΙΝΑ: πόσα από τα 418 chunks περιέχουν αναφορές, τι ποσοστό του
κειμένου τους είναι, και ποια είναι ΜΕΙΚΤΑ (βιβλιογραφία + περιεχόμενο) έναντι
ΚΑΘΑΡΩΝ (μόνο βιβλιογραφία). Τα μεικτά είναι το ρίσκο, τα καθαρά είναι σκέτος
θόρυβος στο index.

ΜΕΡΟΣ Β — ΤΟ ΑΠΟΦΑΣΙΣΤΙΚΟ: για τη σελίδα-στόχο, ξαναβαθμολογεί με τον
ΠΡΑΓΜΑΤΙΚΟ cross-encoder το αρχικό κείμενο έναντι του καθαρισμένου, και
ξαναϋπολογίζει τη θέση του στο dense αντικαθιστώντας ΜΟΝΟ αυτή τη γραμμή του
πίνακα embeddings. Καμία εγγραφή, κανένα re-ingest.

ΤΟ ΚΡΙΤΗΡΙΟ ΑΠΟΦΑΣΗΣ
--------------------
Αν το σκορ του cross-encoder ΔΕΝ ανέβει ουσιωδώς, η υπόθεση πεθαίνει εδώ και
ΔΕΝ προχωράμε σε re-ingest. Το πάτωμα θορύβου αυτού του project είναι ±0.04 σε
n=45· εδώ n=1, οπότε μετράει μόνο **μεγάλη** κίνηση (τάξης logits), όχι
δεύτερο δεκαδικό.

    docker compose exec backend python evaluation/probe_reference_contamination.py
    docker compose exec backend python evaluation/probe_reference_contamination.py \\
        --doc excamera-nsdi17.pdf --page 15 --show
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import ai_core

EVAL = "/app/evaluation"

# Ερωτήσεις-στόχοι: ΙΔΙΑ σελίδα, μία κακοδιατυπωμένη και μία σωστή. Η δεύτερη
# είναι το κρίσιμο control — αν ΚΑΙ αυτή είναι στο −1.97, το πρόβλημα δεν είναι
# η διατύπωση της ερώτησης αλλά η αναπαράσταση της σελίδας.
DEFAULT_QUERIES = [
    ("h002", "What exact settings did they use when they ran the encoder?"),
    ("q028", "Which specific vpxenc command-line flags does the ExCamera "
             "appendix list for its encoding runs?"),
]

# Αρχή εγγραφής βιβλιογραφίας: «[31] YU, Y., AND ...». Ο αριθμός σε αγκύλες
# στην αρχή γραμμής είναι το ασφαλέστερο σημάδι — οι ενδοκειμενικές αναφορές
# («όπως στο [12]») δεν ξεκινούν γραμμή.
REF_START = re.compile(r"^\s*\[\d{1,3}\]\s+\S")

# Τέλος του μπλοκ βιβλιογραφίας: επικεφαλίδα ενότητας ή αριθμημένο τμήμα.
SECTION = re.compile(
    r"^\s*(Appendix|APPENDIX|Acknowledg|Abstract|References|REFERENCES|"
    r"\d+(\.\d+)*\.?\s+[A-Z])")


def split_references(text: str):
    """Χωρίζει το κείμενο σε (περιεχόμενο, βιβλιογραφία).

    ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΟΣ ΚΑΙ ΕΛΕΓΞΙΜΟΣ: δουλεύει ανά γραμμή. Μια γραμμή ανήκει στη
    βιβλιογραφία αν ξεκινάει εγγραφή ([NN] ...) ή αν είναι ΣΥΝΕΧΕΙΑ εγγραφής.
    Η συνέχεια σπάει σε επικεφαλίδα ενότητας ή σε κενή γραμμή που ακολουθείται
    από μη-εγγραφή.

    ΓΙΑΤΙ ΟΧΙ ΠΙΟ ΕΞΥΠΝΟ: ό,τι αφαιρείται πρέπει να μπορεί να ελεγχθεί με το
    μάτι (`--show`). Ένας ευρετικός κανόνας που δεν τυπώνεται δεν είναι μέτρηση.
    """
    lines = text.split("\n")
    is_ref = [False] * len(lines)
    in_ref = False
    for i, ln in enumerate(lines):
        if REF_START.match(ln):
            in_ref = True
            is_ref[i] = True
            continue
        if in_ref:
            if SECTION.match(ln):
                in_ref = False
                continue
            if not ln.strip():
                # Κενή γραμμή: συνεχίζει ΜΟΝΟ αν ακολουθεί άλλη εγγραφή.
                nxt = next((lines[j] for j in range(i + 1, len(lines))
                            if lines[j].strip()), "")
                if not REF_START.match(nxt):
                    in_ref = False
                continue
            is_ref[i] = True
    content = "\n".join(ln for ln, r in zip(lines, is_ref) if not r).strip()
    refs = "\n".join(ln for ln, r in zip(lines, is_ref) if r).strip()
    return content, refs


def embed(text: str) -> np.ndarray:
    v = np.asarray(ai_core.sentence_transformer_ef([text])[0], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


def rank_of(dm, row_pos: int, qv: np.ndarray, replacement=None) -> tuple:
    """Θέση του chunk στο dense, προαιρετικά με ΑΝΤΙΚΑΤΑΣΤΑΜΕΝΟ διάνυσμα."""
    M = dm["matrix"]
    sims = M @ qv
    if replacement is not None:
        sims = sims.copy()
        sims[row_pos] = float(replacement @ qv)
    order = np.argsort(-sims, kind="stable")
    return int(np.where(order == row_pos)[0][0]) + 1, float(sims[row_pos])


def main(args) -> int:
    data = ai_core.collection.get(include=["documents", "metadatas"])
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    print(f"\n{len(ids)} chunks στο index\n")

    # ---------------- ΜΕΡΟΣ Α — ΑΚΤΙΝΑ ΕΠΙΔΡΑΣΗΣ ----------------
    print("=" * 78)
    print("ΜΕΡΟΣ Α — ΠΟΣΑ CHUNKS ΑΓΓΙΖΕΙ ΕΝΑ ΚΑΘΑΡΙΣΜΑ ΒΙΒΛΙΟΓΡΑΦΙΑΣ")
    print("=" * 78)
    rows, mixed, pure = [], [], []
    for cid, txt, m in zip(ids, docs, metas):
        content, refs = split_references(txt)
        if not refs:
            continue
        frac = len(refs) / max(len(txt), 1)
        rec = {"id": cid, "doc": m.get("file_name", "?"),
               "page": m.get("page", "?"), "chars": len(txt),
               "ref_chars": len(refs), "ref_frac": round(frac, 3),
               "left_chars": len(content)}
        rows.append(rec)
        (pure if len(content) < args.pure_below else mixed).append(rec)

    print(f"chunks με βιβλιογραφία : {len(rows)} / {len(ids)}  "
          f"({100.0 * len(rows) / len(ids):.1f}%)")
    print(f"  ΚΑΘΑΡΑ (σχεδόν μόνο αναφορές, <{args.pure_below} χαρ. μένουν): "
          f"{len(pure)}   -> σκέτος θόρυβος στο index")
    print(f"  ΜΕΙΚΤΑ (αναφορές + περιεχόμενο)                  : "
          f"{len(mixed)}   -> ΕΔΩ είναι και το κέρδος και το ρίσκο")
    if mixed:
        print(f"\n{'id':<26}{'doc':<26}{'σελ':>5}{'χαρ':>7}"
              f"{'ref%':>7}{'μένουν':>8}")
        print("-" * 78)
        for r in sorted(mixed, key=lambda x: -x["ref_frac"]):
            print(f"{r['id'][:24]:<26}{str(r['doc'])[:24]:<26}"
                  f"{r['page']!s:>5}{r['chars']:>7}"
                  f"{100 * r['ref_frac']:>6.0f}%{r['left_chars']:>8}")

    # ---------------- ΜΕΡΟΣ Β — ΤΟ ΑΠΟΦΑΣΙΣΤΙΚΟ ----------------
    print("\n" + "=" * 78)
    print(f"ΜΕΡΟΣ Β — ΞΑΝΑΒΑΘΜΟΛΟΓΗΣΗ ΤΗΣ {args.doc}:{args.page}")
    print("=" * 78)
    tgt = [(c, t, m) for c, t, m in zip(ids, docs, metas)
           if m.get("file_name") == args.doc and str(m.get("page")) == str(args.page)]
    if not tgt:
        print(f"ΔΕΝ ΒΡΕΘΗΚΕ chunk για {args.doc}:{args.page}")
        return 1

    dm = ai_core._get_dense_matrix()
    queries = DEFAULT_QUERIES if not args.query else [("custom", q) for q in args.query]
    out = []

    for cid, txt, _m in tgt:
        content, refs = split_references(txt)
        print(f"\nchunk {cid}  ({len(txt)} χαρ.)  ->  περιεχόμενο "
              f"{len(content)}  ·  βιβλιογραφία {len(refs)} "
              f"({100.0 * len(refs) / max(len(txt), 1):.0f}%)")
        if args.show:
            print("-" * 78)
            print("ΑΦΑΙΡΕΙΤΑΙ:\n" + (refs or "(τίποτα)"))
            print("-" * 78)
            print("ΜΕΝΕΙ:\n" + content[:args.chars])
            print("-" * 78)
        if not refs:
            print("  Δεν εντοπίστηκε βιβλιογραφία — τίποτα να μετρηθεί.")
            continue

        pos = dm["pos"][cid]
        v_clean = embed(content)
        variants = [("αρχικό", txt, None), ("ΧΩΡΙΣ βιβλιογρ.", content, v_clean)]
        if args.with_refs_only:
            variants.append(("μόνο βιβλιογρ.", refs, embed(refs)))
        # Περιγραφικό πρόθεμα: η ΓΕΦΥΡΑ ΟΡΟΛΟΓΙΑΣ μπαίνει στο ΕΓΓΡΑΦΟ, όχι στο
        # ερώτημα. Ο εμπλουτισμός ερωτήματος απορρίφθηκε 3 φορές γιατί μεγαλώνει
        # ΚΑΘΕ ερώτηση και παραπλανά το gate· ένα στατικό πρόθεμα ανά chunk
        # μπορεί να ελεγχθεί με το μάτι και έχει σταθερή, μετρήσιμη ακτίνα.
        pfx_list = list(args.prefix or [])
        for path in (args.prefix_json or []):
            with open(path, encoding="utf-8") as fh:
                got = json.load(fh).get(cid)
            if got:
                pfx_list.append(got)
            else:
                print(f"  (το {os.path.basename(path)} δεν έχει το {cid})")
        if len(pfx_list) > 1 and args.combine:
            # ΓΙΑΤΙ: περιγραφή και ερωτήσεις τραβάνε ΑΝΤΙΘΕΤΑ — η περιγραφή
            # ανεβάζει το dense (15->4) και αφήνει τον reranker, οι ερωτήσεις
            # ανεβάζουν τον reranker (+3.97) και ρίχνουν το dense (->18).
            pfx_list.append("\n".join(pfx_list))
        for i, pfx in enumerate(pfx_list):
            body = content if args.prefix_on_clean else txt
            newtxt = pfx.rstrip() + "\n" + body
            name = ("ΟΛΑ ΜΑΖΙ" if args.combine and i == len(pfx_list) - 1
                    else f"πρόθεμα {i + 1}")
            variants.append((name, newtxt, embed(newtxt)))

        for qid, q in queries:
            qv = ai_core._embed_query(q)
            print(f"\n  «{q[:66]}»")
            print(f"  {'παραλλαγή':<18}{'rerank':>9}{'dense θέση':>12}"
                  f"{'cos':>8}   gate")
            print("  " + "-" * 74)
            base = None
            for name, text, rep in variants:
                s = float(ai_core.reranker.predict(
                    [[q, text]], batch_size=1)[0])
                rk, cs = rank_of(dm, pos, qv, rep)
                if base is None:
                    base = s
                mark = ("ΠΕΡΝΑΕΙ" if s >= ai_core.MIN_RERANK_SCORE else "κόβει")
                delta = "" if name == "αρχικό" else f"  ({s - base:+.2f})"
                print(f"  {name:<18}{s:>9.2f}{rk:>12}{cs:>8.3f}   "
                      f"{mark}{delta}")
                out.append({"chunk": cid, "query_id": qid, "variant": name,
                            "rerank": round(s, 3), "dense_rank": rk,
                            "cos": round(cs, 4),
                            "passes_gate": s >= ai_core.MIN_RERANK_SCORE})

    print("\n" + "=" * 78)
    print("ΠΩΣ ΔΙΑΒΑΖΕΤΑΙ")
    print(f"  gate = {ai_core.MIN_RERANK_SCORE} (ωμά logits).  Το h002 σήμερα "
          "παίρνει top-1 +1.27 ΑΠΟ ΑΛΛΕΣ σελίδες, άρα δεν κόβεται καν.")
    print("  Το κρίσιμο ΔΕΝ είναι το gate — είναι αν ο στόχος ανεβαίνει αρκετά")
    print("  ώστε (α) το dense να τον βάλει μέσα στους 30 με καλή θέση και")
    print("  (β) ο reranker να τον κατατάξει πάνω από τις άσχετες σελίδες.")
    print("  ΑΝ ΤΟ rerank ΔΕΝ ΚΟΥΝΗΘΕΙ ΠΑΝΩ ΑΠΟ ~1 logit, Η ΥΠΟΘΕΣΗ ΠΕΘΑΙΝΕΙ")
    print("  και ΔΕΝ γίνεται re-ingest.")
    print("=" * 78)

    if args.csv and out:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f"→ γράφτηκε {args.csv}")
    if args.radius_csv and rows:
        os.makedirs(os.path.dirname(args.radius_csv), exist_ok=True)
        with open(args.radius_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"→ γράφτηκε {args.radius_csv}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="μολύνει η βιβλιογραφία τα chunks; (ΜΗΔΕΝ κόστος)")
    ap.add_argument("--doc", default="excamera-nsdi17.pdf")
    ap.add_argument("--page", default="15")
    ap.add_argument("--query", nargs="*", default=None,
                    help="δικές σου ερωτήσεις αντί για h002/q028")
    ap.add_argument("--show", action="store_true",
                    help="τύπωσε ΤΙ ΑΚΡΙΒΩΣ αφαιρείται (έλεγχος με το μάτι)")
    ap.add_argument("--with-refs-only", action="store_true",
                    help="βαθμολόγησε και ΜΟΝΟ τη βιβλιογραφία (control)")
    ap.add_argument("--prefix", nargs="*", default=None,
                    help="περιγραφικό πρόθεμα/-τα να μπουν ΜΠΡΟΣΤΑ στο chunk")
    ap.add_argument("--prefix-json", nargs="*", default=None,
                    help="JSON {chunk_id: πρόθεμα} από το build_chunk_prefixes")
    ap.add_argument("--combine", action="store_true",
                    help="βαθμολόγησε και ΟΛΑ τα προθέματα μαζί")
    ap.add_argument("--prefix-on-clean", action="store_true",
                    help="βάλε το πρόθεμα στο ΚΑΘΑΡΙΣΜΕΝΟ κείμενο")
    ap.add_argument("--pure-below", type=int, default=120,
                    help="χαρακτήρες περιεχομένου κάτω από τους οποίους το "
                         "chunk θεωρείται ΚΑΘΑΡΗ βιβλιογραφία")
    ap.add_argument("--chars", type=int, default=700)
    ap.add_argument("--csv", default="/app/evaluation/runs/refs_rescore.csv")
    ap.add_argument("--radius-csv",
                    default="/app/evaluation/runs/refs_radius.csv")
    raise SystemExit(main(ap.parse_args()))

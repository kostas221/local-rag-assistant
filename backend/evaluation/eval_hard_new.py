"""Φάση 0.2: πόσο συχνά λέει το σύστημα «δεν ξέρω» σε ερώτηση που ΕΧΕΙ απάντηση;

ΤΟ ΕΡΩΤΗΜΑ ΜΕ ΑΠΛΑ ΛΟΓΙΑ:
    Κάθε ερώτηση του golden_hard_new είναι παράφραση ερώτησης που ξέρουμε ότι
    απαντιέται — γραμμένη όπως θα τη γράφε ένας χρήστης (αόριστα, χωρίς ορολογία,
    στα ελληνικά, πολύ σύντομα). Άρα ΚΑΘΕ σιωπή είναι λάθος, και κάθε απάντηση από
    σελίδες χωρίς τις λέξεις-κλειδιά είναι ύποπτη («νομίζω ότι ξέρω»).

ΤΙ ΚΑΝΕΙ:
    • ίδιο απομονωμένο store με το eval_near_ooc (7 cloud PDF, 418 chunks) — ΟΧΙ το
      ευρετήριο της παραγωγής, που 25/9 έχει άλλα αρχεία (λείπει το 1706.03178).
    • δύο περάσματα με τον ΠΡΑΓΜΑΤΙΚΟ search_documents:
        OFF = ENABLE_CORRECTIVE=0 -> ντετερμινιστικό (μόνο φύλακας)
        ON  = όπως η παραγωγή       -> + δεύτερη ευκαιρία (τυχαιότητα: η αναδιατύπωση)
    • ανά ερώτηση, στις σελίδες που γύρισαν: πόσες λέξεις-κλειδιά βρέθηκαν (page-level,
      ίδιο κριτήριο με το verify_corrective).
    • ΕΛΕΓΧΟΙ: (α) το ΠΑΛΙΟ hard set (16) πρέπει να δώσει ό,τι καταγράφεται (13/16
      απαντιούνται)· (β) οι 5 out_of_corpus πρέπει να κόβονται στο OFF.
    Μεταφράσεις: σώζονται στο runs/hard_translations.json -> 2ο τρέξιμο = ίδια ανάκτηση.

ΚΑΤΗΓΟΡΙΕΣ (πέρασμα ON):
    με υλικό     = γύρισαν σελίδες ΚΑΙ ≥1 λέξη-κλειδί
    χωρίς υλικό  = γύρισαν σελίδες, 0 λέξεις-κλειδιά  (ο φύλακας δεν το έπιασε)
    σιωπή        = καμία σελίδα                         (λάθος κοπή — ΟΛΕΣ απαντιούνται)

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (25/9/2026), νέο σετ n=59:
    κόβει ο φύλακας (OFF) 12-20 · σιωπή μετά τον corrective (ON) 6-12
    με υλικό 38-47 (65-80%) · χωρίς υλικό 3-7
    μέση κάλυψη λέξεων ~55-65% (πάτωμα τύχης 17.5%)
    με όνομα συστήματος (20) ≥15 μονάδες καλύτερα από χωρίς (39)
    χειρότεροι τύποι: short και nojargon
    έλεγχος: παλιό hard 13/16 (±1 από την αναδιατύπωση) · ooc 5/5 κομμένα στο OFF

ΑΠΟΤΕΛΕΣΜΑ 1ου ΤΡΕΞΙΜΑΤΟΣ (25/9/2026, runs/hard_new_eval_run1.csv, ΠΑΛΙΕΣ λέξεις):
    κόβει ο φύλακας 8 ✗ · σιωπή 1 ✗ · με υλικό 51 ✗ · χωρίς υλικό 7 ✓ · κάλυψη 78% ✗
    με όνομα 19/20 έναντι 32/39 χωρίς = 13 μονάδες ✗ · χειρότερος τύπος meta 12/15 ✗
    έλεγχοι: παλιό hard 14/16 απαντιούνται (σιωπή h009, h012) ✓ · ooc 5/5 κομμένα ✓
    ΜΙΑ σωστή πρόβλεψη στις επτά — υποτίμησα το σύστημα σε όλα.
    Με το μάτι (inspect_hard_failures.py), από τα 8 λάθη τα 2 ήταν του ΣΕΤ:
        h109 = δύο σωστές απαντήσεις (έχασε το «στο paper») -> set_error, ΜΕΝΕΙ στο σετ
        h111 = οι λέξεις μου ήταν στενές (έφερε τη ΣΩΣΤΗ σελίδα 1902 σ.6) -> νέες λέξεις
    Πραγματικά λάθη 6: σιωπή h158 · λάθος σελίδες h126, h138, h145, h152, h167.
    Το κύριο λάθος ΔΕΝ είναι η σιωπή — είναι η απάντηση από λάθος σελίδες.
    ΣΦΑΛΜΑ ΜΕΤΡΗΣΗΣ που βρέθηκε: φράσεις-κλειδιά ελέγχθηκαν με ενωμένες γραμμές, ενώ
    η αξιολόγηση ψάχνει το κείμενο με τις αλλαγές γραμμής («next decade» -> 0 σελίδες).
    Διορθώθηκαν q001/q008/q009/q018 στο runs/hard_review.json · πάτωμα τύχης 18.3%.

ΞΑΝΑΒΑΘΜΟΛΟΓΗΣΗ ΧΩΡΙΣ ΝΕΑ ΑΝΑΖΗΤΗΣΗ:
    Το CSV σώζει ποιες σελίδες γύρισαν (off_pagelist / on_pagelist). Αν αλλάξουν ΜΟΝΟ
    λέξεις-κλειδιά, το --rescore ξαναμετράει από τις σωσμένες σελίδες: μηδέν Gemini,
    μηδέν αναζήτηση, και η τυχαιότητα της αναδιατύπωσης δεν μπαίνει ξανά στη μέτρηση.

ΚΟΣΤΟΣ: μεταφράσεις των νέων ερωτήσεων (~60 × ~100 tokens, μία φορά) + αναδιατυπώσεις
μόνο όσων κόβονται στο ON. Καμία γέννηση απάντησης. Χρόνος ~1 λεπτό φόρτωση + ~5 λεπτά.

    docker compose exec backend python evaluation/eval_hard_new.py
    docker compose exec backend python evaluation/eval_hard_new.py --limit 5   # δοκιμή
    docker compose exec backend python evaluation/eval_hard_new.py --rescore   # μόνο λέξεις άλλαξαν
"""
import argparse
import asyncio
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_near_ooc as E

NEW_PATH = os.path.join(E.HERE, "golden_hard_new.jsonl")
OLD_PATH = os.path.join(E.HERE, "golden_hard_paraphrase.jsonl")
TRANS_PATH = os.path.join(E.HERE, "runs", "hard_translations.json")
FIELDS = ["set", "id", "parent", "category", "hard_type", "lang", "named", "set_error",
          "question", "translation", "n_kw", "off_pages", "off_cov", "best1", "on_pages",
          "on_cov", "best2", "rewrite", "outcome", "verdict", "files",
          "off_pagelist", "on_pagelist"]


def covered(pages, kws) -> int:
    """Ίδιο με το verify_corrective.covered: λέξεις-κλειδιά στις ΣΕΛΙΔΕΣ που γύρισαν.
    Κείμενο ΟΠΩΣ ΕΙΝΑΙ (με τις αλλαγές γραμμής) — ίδιο με το eval_engine. Μια φράση
    που σπάει σε δύο γραμμές ΔΕΝ βρίσκεται· γι' αυτό οι φράσεις ελέγχονται με raw κείμενο."""
    blob = "\n".join(text for text, _meta in pages).lower()
    return sum(1 for k in kws if k.lower() in blob)


def verdict(n_pages: int, cov: int) -> str:
    if not n_pages:
        return "σιωπή"
    return "με υλικό" if cov else "χωρίς υλικό"


def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n:.0%})" if n else "—"


def pagelist(pages) -> str:
    return ";".join(f"{m.get('file_name', '?')}:{m.get('page')}" for _x, m in pages)


def load_tests(limit: int = 0):
    new = E.load_jsonl(NEW_PATH)[:limit or None]
    old = E.load_jsonl(OLD_PATH)
    ooc = [r for r in E.load_jsonl(E.GOLDEN_50) if r.get("category") == "out_of_corpus"]
    for t in old + ooc:
        t.setdefault("hard_type", "")
    tests = [("νέο", t) for t in new] + [("παλιό", t) for t in old] + [("ooc", t) for t in ooc]
    return new, old, ooc, tests


async def run_pass(ai_core, tests, enabled: bool, saved: dict) -> dict:
    ai_core.ENABLE_CORRECTIVE = enabled
    out = {}
    for t in tests:
        pages, b1, b2, rw, outcome = await E.retrieve(ai_core, t["question"])
        tr = ai_core._translation_cache.get(t["question"], "")
        if tr:
            saved[t["question"]] = tr
        out[t["id"]] = {"pages": pages, "b1": b1, "b2": b2, "rw": rw, "outcome": outcome,
                        "cov": covered(pages, t["keywords"]) if pages else 0}
        print(f"  {'ON ' if enabled else 'OFF'} {t['id']:<6} {outcome:<18} best1 {E.fmt(b1):>6}"
              f"  best2 {E.fmt(b2):>6}  σελ {len(pages)}  kw {out[t['id']]['cov']}/"
              f"{len(t['keywords'])}", flush=True)
    return out


async def run_retrieval(args, tests) -> dict:
    flat = [t for _s, t in tests]
    ai_core, _near = E.open_store()
    saved: dict = {}
    if os.path.exists(TRANS_PATH):
        with open(TRANS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        ai_core._translation_cache.update(saved)
        print(f"Μεταφράσεις hard από προηγούμενο τρέξιμο: {len(saved)}")
    print(f"gate {ai_core.MIN_RERANK_SCORE} · corrective {ai_core.CORRECTIVE_MIN_SCORE}")

    print(f"\n===== ΠΕΡΑΣΜΑ OFF (ντετερμινιστικό) — {len(flat)} ερωτήσεις =====")
    off = await run_pass(ai_core, flat, False, saved)
    os.makedirs(os.path.dirname(TRANS_PATH), exist_ok=True)
    with open(TRANS_PATH, "w", encoding="utf-8") as f:      # σώζεται ΠΡΙΝ το ON
        json.dump(saved, f, ensure_ascii=False, indent=1)
    print(f"\n===== ΠΕΡΑΣΜΑ ON (όπως η παραγωγή) — {len(flat)} ερωτήσεις =====")
    on = await run_pass(ai_core, flat, True, saved)

    rows = {}
    for s, t in tests:
        o, n = off[t["id"]], on[t["id"]]
        rows[t["id"]] = {
            "set": s, "id": t["id"], "parent": t.get("parent", ""),
            "category": t.get("category", ""), "hard_type": t.get("hard_type", ""),
            "lang": t.get("lang", ""), "named": int(bool(t.get("names_entity"))),
            "set_error": int(bool(t.get("set_error"))),
            "question": t["question"], "translation": saved.get(t["question"], ""),
            "n_kw": len(t["keywords"]), "off_pages": len(o["pages"]), "off_cov": o["cov"],
            "best1": E.fmt(n["b1"]), "on_pages": len(n["pages"]), "on_cov": n["cov"],
            "best2": E.fmt(n["b2"]), "rewrite": n["rw"], "outcome": n["outcome"],
            "verdict": verdict(len(n["pages"]), n["cov"]),
            "files": ";".join(sorted({m.get("file_name", "?") for _x, m in n["pages"]})),
            "off_pagelist": pagelist(o["pages"]), "on_pagelist": pagelist(n["pages"])}
    return rows


def store_pages() -> dict:
    """{'αρχείο:σελίδα': κείμενο} από το απομονωμένο store, ΟΠΩΣ το _expand_to_pages:
    chunks σε σειρά index, ενωμένα με '\\n'. Χωρίς ai_core -> δευτερόλεπτα."""
    import re

    import chromadb
    col = chromadb.PersistentClient(path=E.TEST_DB).get_collection("eval_near_ooc")
    got = col.get(include=["documents", "metadatas"])
    by: dict = {}
    for cid, doc, m in zip(got["ids"], got["documents"], got["metadatas"]):
        idx = re.search(r"_c(\d+)_", cid)
        by.setdefault(f"{m.get('file_name')}:{m.get('page')}", []).append(
            (int(idx.group(1)) if idx else 0, doc or ""))
    return {k: "\n".join(d for _i, d in sorted(v)) for k, v in by.items()}


def rescore(args, tests) -> dict:
    """ΜΗΔΕΝ ανάκτηση: ξαναμετρά την κάλυψη στις ΣΩΣΜΕΝΕΣ σελίδες με τις ΤΡΕΧΟΥΣΕΣ
    λέξεις-κλειδιά. Για αλλαγές keywords χωρίς νέο τρέξιμο (και χωρίς νέα τυχαιότητα
    από την αναδιατύπωση)."""
    with open(args.csv, encoding="utf-8") as f:
        old_rows = {r["id"]: r for r in csv.DictReader(f)}
    if "on_pagelist" not in next(iter(old_rows.values())):
        raise SystemExit("!! Το CSV είναι από παλιό τρέξιμο χωρίς λίστα σελίδων — "
                         "χρειάζεται ένα πλήρες τρέξιμο πρώτα")
    text = store_pages()
    rows = {}
    for s, t in tests:
        r = dict(old_rows[t["id"]])
        for side in ("off", "on"):
            keys = [k for k in r[f"{side}_pagelist"].split(";") if k]
            missing = [k for k in keys if k not in text]
            if missing:
                raise SystemExit(f"!! {t['id']}: σελίδες που δεν υπάρχουν στο store {missing}")
            r[f"{side}_pages"] = len(keys)
            r[f"{side}_cov"] = covered([(text[k], {}) for k in keys], t["keywords"]) if keys else 0
        r.update(set=s, n_kw=len(t["keywords"]), set_error=int(bool(t.get("set_error"))),
                 verdict=verdict(r["on_pages"], r["on_cov"]))
        changed = r["verdict"] != old_rows[t["id"]]["verdict"]
        if changed:
            print(f"  {t['id']}: {old_rows[t['id']]['verdict']} -> {r['verdict']}")
        rows[t["id"]] = r
    return rows


def summarize(label: str, tests: list[dict], rows: dict) -> None:
    n = len(tests)
    v = Counter(rows[t["id"]]["verdict"] for t in tests)
    off_cut = sum(int(rows[t["id"]]["off_pages"]) == 0 for t in tests)
    saved = sum(int(rows[t["id"]]["off_pages"]) == 0 and rows[t["id"]]["verdict"] == "με υλικό"
                for t in tests)
    kw = sum(int(rows[t["id"]]["on_cov"]) / int(rows[t["id"]]["n_kw"]) for t in tests) / max(1, n)
    print(f"\n{label} (n={n})")
    print(f"  φύλακας (OFF) έκοψε {pct(off_cut, n)} · ο corrective έσωσε με υλικό {saved}")
    print(f"  ON:  με υλικό {pct(v['με υλικό'], n)} · χωρίς υλικό {pct(v['χωρίς υλικό'], n)}"
          f" · σιωπή {pct(v['σιωπή'], n)}")
    print(f"  μέση κάλυψη λέξεων-κλειδιών (ON): {kw:.1%}")
    for bad in ("σιωπή", "χωρίς υλικό"):
        print(f"  άνω όριο 95% «{bad}»: {E.upper_bound(v[bad], n):.1%}")


def breakdown(title: str, tests: list[dict], rows: dict, key) -> None:
    print(f"\n  ανά {title}:")
    groups: dict = {}
    for t in tests:
        groups.setdefault(key(t), []).append(t)
    for g, ts in sorted(groups.items(), key=lambda x: str(x[0])):
        v = Counter(rows[t["id"]]["verdict"] for t in ts)
        print(f"    {g!s:<14} n={len(ts):<3} με υλικό {v['με υλικό'] / len(ts):>4.0%} · "
              f"χωρίς {v['χωρίς υλικό']:>2} · σιωπή {v['σιωπή']:>2}")


def report(new, old, ooc, rows) -> None:
    print("\n" + "#" * 78)
    print("ΕΛΕΓΧΟΙ")
    cut = [t["id"] for t in ooc if int(rows[t["id"]]["off_pages"]) == 0]
    print(f"  out_of_corpus κομμένα στο OFF: {len(cut)}/{len(ooc)} "
          f"{'OK' if len(cut) == len(ooc) else '!! ΑΠΕΤΥΧΕ — ύποπτο το store'}")
    leak = [t["id"] for t in ooc if int(rows[t["id"]]["on_pages"])]
    print(f"  out_of_corpus που πέρασαν στο ON: {leak or 'κανένα'}  (γνωστό: ~1/10 για q048)")
    summarize("ΠΑΛΙΟ hard set — καταγραφή: 13/16 απαντιούνται", old, rows)

    print("\n" + "#" * 78)
    summarize("ΝΕΟ hard set", new, rows)
    clean = [t for t in new if not t.get("set_error")]
    if len(clean) < len(new):
        summarize(f"ΝΕΟ hard set ΧΩΡΙΣ τα {len(new) - len(clean)} σφάλματα σετ", clean, rows)
    breakdown("τύπο", new, rows, lambda t: t["hard_type"])
    breakdown("γλώσσα", new, rows, lambda t: t["lang"])
    breakdown("κατηγορία", new, rows, lambda t: t["category"])
    breakdown("όνομα συστήματος", new, rows,
              lambda t: "με όνομα" if t.get("names_entity") else "χωρίς όνομα")

    print("\n--- σιωπές και «χωρίς υλικό» (νέο σετ) ---")
    for t in new:
        r = rows[t["id"]]
        if r["verdict"] != "με υλικό":
            mark = "  [ΣΦΑΛΜΑ ΣΕΤ]" if t.get("set_error") else ""
            print(f"  {t['id']} {t['hard_type']:<9} {r['verdict']:<12} best1 {r['best1']:>6}"
                  f"  best2 {r['best2']:>6}  {t['question'][:70]}{mark}")
    print("\nΠΡΟΒΛΕΨΗ (γραμμένη πριν): OFF κόβει 12-20 · ON σιωπή 6-12 · με υλικό 38-47 · "
          "χωρίς υλικό 3-7 · κάλυψη ~55-65% · με όνομα ≥15 μονάδες καλύτερα · "
          "χειρότεροι short/nojargon")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="μόνο οι πρώτες N νέες (δοκιμή)")
    ap.add_argument("--rescore", action="store_true",
                    help="ΧΩΡΙΣ ανάκτηση: ξαναβαθμολόγηση των σωσμένων σελίδων με τα τρέχοντα keywords")
    ap.add_argument("--csv", default=os.path.join(E.HERE, "runs", "hard_new_eval.csv"))
    args = ap.parse_args()
    new, old, ooc, tests = load_tests(args.limit)

    if args.rescore:
        rows = rescore(args, tests)
    else:
        if not E.API_KEY:
            print("!! Λείπει GEMINI_API_KEY — σταματάω")
            return 1
        rows = await run_retrieval(args, tests)
    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows.values())
    report(new, old, ooc, rows)
    print(f"\nΓράφτηκε: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

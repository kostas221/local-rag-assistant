"""Τελική επαλήθευση του corrective retrieval — στον ΠΡΑΓΜΑΤΙΚΟ κώδικα.

ΓΙΑΤΙ ΞΕΧΩΡΙΣΤΑ ΑΠΟ ΤΟ probe_corrective_rewrite.py:
το probe ΑΝΑΠΑΡΗΓΑΓΕ το pipeline. Αυτό καλεί το ίδιο το search_documents(), άρα
πιάνει και λάθη ενσωμάτωσης που το probe δεν μπορεί να δει (λάθος ορίσματα, το
retry να μη συνδέεται με το gate, το expand να τρέχει σε λάθος λίστα).

ΤΙ ΜΕΤΡΑΕΙ ανά ερώτηση:
  - απάντησε ή σιώπησε το σύστημα (κενή λίστα = σιωπή)
  - αν απάντησε: περιέχουν οι ΣΕΛΙΔΕΣ που γύρισε τα keywords; (page-level, όπως
    το eval_engine). Απάντηση ΧΩΡΙΣ keyword = ψευδαίσθηση, ΧΕΙΡΟΤΕΡΗ από σιωπή.

Τρέχει δύο φορές, με ENABLE_CORRECTIVE on/off, ώστε η σύγκριση να έχει ΜΙΑ
μεταβλητή. Το on/off γίνεται με monkeypatch του module global — ο κώδικας
παραγωγής μένει ανέπαφος.

ΚΟΣΤΟΣ: ~15 κλήσεις rewrite στο πέρασμα "on". Το "off" είναι δωρεάν.

    docker compose exec backend python evaluation/verify_corrective.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")
import ai_core
from evaluation.eval_engine import GOLDEN_CORPUS

SETS = [
    "/app/evaluation/golden_hard_paraphrase.jsonl",
    "/app/evaluation/golden_set_50.jsonl",
    "/app/evaluation/golden_multihop_new.jsonl",
]


def load():
    tests, seen = [], set()
    for p in SETS:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    if t["id"] not in seen:
                        seen.add(t["id"])
                        tests.append(t)
    return tests


def covered(pages, kws):
    """Πόσα keywords εμφανίζονται στις σελίδες που γύρισε — page-level, όπως
    το eval_engine (το search_documents τελειώνει με _expand_to_pages)."""
    blob = "\n".join(text for text, _meta in pages).lower()
    return sum(1 for k in kws if k.lower() in blob)


async def run_all(tests, enabled: bool):
    ai_core.ENABLE_CORRECTIVE = enabled
    out = {}
    for t in tests:
        pages = await ai_core.search_documents(t["question"], GOLDEN_CORPUS)
        out[t["id"]] = (len(pages), covered(pages, t["keywords"]) if pages else 0)
    return out


async def main():
    tests = load()
    print(f"{len(tests)} ερωτήσεις · CORRECTIVE_MIN_SCORE = "
          f"{ai_core.CORRECTIVE_MIN_SCORE}\n")

    print("--- πέρασμα 1/2: ENABLE_CORRECTIVE=0 (δωρεάν) ---", flush=True)
    off = await run_all(tests, False)
    print("--- πέρασμα 2/2: ENABLE_CORRECTIVE=1 (~15 rewrites) ---\n", flush=True)
    on = await run_all(tests, True)

    hard = [t for t in tests if t["id"].startswith("h")]
    main_set = [t for t in tests if not t["id"].startswith("h")]

    print(f"{'id':<6} {'category':<14} {'off':>14} {'on':>14}  {'έκβαση'}")
    print("-" * 78)
    changed = []
    for t in tests:
        no, co = off[t["id"]]
        nn, cn = on[t["id"]]
        if (no, co) == (nn, cn):
            continue
        changed.append(t)
        n_kw = len(t["keywords"])
        ooc = t.get("category") == "out_of_corpus"
        if ooc and nn:
            verdict = "ΔΙΑΡΡΟΗ"
        elif not no and nn and cn:
            verdict = "ΣΩΘΗΚΕ"
        elif not no and nn and not cn:
            verdict = "ΨΕΥΔΑΙΣΘΗΣΗ"
        else:
            verdict = "άλλαξε"
        print(f"{t['id']:<6} {t.get('category',''):<14} "
              f"{f'{no} σελ, {co}/{n_kw} kw':>14} {f'{nn} σελ, {cn}/{n_kw} kw':>14}"
              f"  {verdict}")

    def summarize(label, group):
        silent_off = [t for t in group if off[t["id"]][0] == 0]
        silent_on = [t for t in group if on[t["id"]][0] == 0]
        saved = [t for t in group
                 if off[t["id"]][0] == 0 and on[t["id"]][0] > 0 and on[t["id"]][1] > 0]
        halluc = [t for t in group
                  if off[t["id"]][0] == 0 and on[t["id"]][0] > 0 and on[t["id"]][1] == 0]
        print(f"\n  {label}: σιωπηλές {len(silent_off)} -> {len(silent_on)}"
              f"  ·  ΣΩΘΗΚΑΝ {len(saved)}"
              + (f" ({', '.join(t['id'] for t in saved)})" if saved else "")
              + f"  ·  ΨΕΥΔΑΙΣΘΗΣΕΙΣ {len(halluc)}"
              + (f" ({', '.join(t['id'] for t in halluc)})" if halluc else ""))

    print("\n" + "=" * 78)
    print("ΑΠΟΤΕΛΕΣΜΑ")
    print("=" * 78)
    summarize("hard set (16)", hard)
    summarize("κύριο σετ", [t for t in main_set
                            if t.get("category") != "out_of_corpus"])

    ooc = [t for t in tests if t.get("category") == "out_of_corpus"]
    leaked = [t for t in ooc if on[t["id"]][0] > 0]
    print(f"\n  out_of_corpus: {len(ooc)-len(leaked)}/{len(ooc)} σιωπηλά "
          + ("— ΤΟ ΚΡΙΤΗΡΙΟ ΚΡΑΤΗΣΕ"
             if not leaked else f"— ΔΙΑΡΡΟΗ: {', '.join(t['id'] for t in leaked)}"))

    untouched = [t for t in main_set if off[t["id"]] == on[t["id"]]]
    print(f"\n  κύριο σετ ανέπαφο: {len(untouched)}/{len(main_set)} "
          f"({'ΟΚ — μηδέν παρενέργεια' if len(untouched) == len(main_set) else 'ΠΡΟΣΟΧΗ'})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

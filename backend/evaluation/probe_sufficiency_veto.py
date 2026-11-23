"""Βήμα 2β: κριτής επάρκειας στη δεύτερη αναζήτηση. Προκαταχώριση: docs/SUFFICIENCY_VETO_PREREG_2026-11-23.md.

ΤΙ ΚΑΝΕΙ: παίρνει τις ΗΔΗ αποθηκευμένες αναδιατυπώσεις του 10ου τρεξίματος (runs/corrective_ab.csv,
prompt «old» = το σημερινό), ξαναβρίσκει με την ίδια αναζήτηση τις σελίδες που θα πήγαιναν στην απάντηση,
και για όσες περνούν σήμερα ρωτά τον κριτή (sufficiency.py) αν απαντούν την ΑΡΧΙΚΗ ερώτηση. Δύο
κληρώσεις. Καμία νέα αναδιατύπωση, καμία νέα μετάφραση: μόνο οι κλήσεις του κριτή (51 × 2).

ΠΑΓΩΜΑ: runs/sufficiency_veto_calls.json, κλειδί = μοντέλο + κλήρωση + prompt. Το 2ο τρέξιμο = 0 κλήσεις.
Το αρχείο κρατά μόνο την ετυμηγορία και την ερώτηση, όχι κείμενο άρθρων (δημόσιο repo).

    docker compose -f docker-compose.yml -f docker-compose.upgrade.yml run --rm --no-deps \\
        -e DATABASE_URL=sqlite:// -e GEMINI_MODEL=gemini-3.8-flash -v eval_tmp:/tmp backend \\
        python -u evaluation/probe_sufficiency_veto.py
"""
import argparse
import asyncio
import csv
import hashlib
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
from probe_corrective_ab import kind, load_tests
from probe_corrective_attempts import _retrieve

import gemini_rest
import sufficiency as SF

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
CALLS_PATH = os.path.join(RUNS, "sufficiency_veto_calls.json")
LOCK = "/tmp/scoreboard.lock"  # noqa: S108  ίδιο store με το scoreboard -> ίδια κλειδαριά
CONCURRENCY = 4

# Κανόνας απόφασης — ΙΔΙΟΣ με την προκαταχώριση (γραμμένος πριν από το τρέξιμο).
MAX_LEAKS = 0
MIN_SAVED_MEAN = 8.0
MIN_AGREEMENT = 0.90
MAX_ERRORS = 2


def _key(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def load_calls() -> dict:
    if not os.path.exists(CALLS_PATH):
        return {}
    with open(CALLS_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_calls(data: dict) -> None:
    tmp = CALLS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, CALLS_PATH)


def summarize(rows: list[dict], draws: int) -> dict:
    """Ανά κλήρωση και προσπάθεια: πόσα περνούν σήμερα και πόσα μετά το βέτο."""
    out = {}
    attempts = sorted({r["attempt"] for r in rows})
    for d in range(1, draws + 1):
        per = []
        for a in attempts:
            sel = [r for r in rows if r["attempt"] == a]
            kept = [r for r in sel if not SF.vetoes(r[f"v{d}"])]

            def count(rs, k, cov=None):
                return sum(r["kind"] == k and (cov is None or (r["cov"] > 0) == cov) for r in rs)

            per.append({"attempt": a,
                        "leaks": (count(sel, "ooc"), count(kept, "ooc")),
                        "near": (count(sel, "near"), count(kept, "near")),
                        "saved": (count(sel, "answerable", True), count(kept, "answerable", True)),
                        "cov0": (count(sel, "answerable", False), count(kept, "answerable", False))})
        out[d] = per
    return out


async def run(args) -> int:
    ai_core, _near = E.open_store(False)
    if ai_core.collection.count() != 418:
        print(f"!! Το store έχει {ai_core.collection.count()} chunks, όχι 418 — σταματάω")
        return 1
    tests = {(t["_set"], t["id"]): t for t in load_tests(args.run)}
    with open(args.ab, encoding="utf-8") as f:
        stored = [r for r in csv.DictReader(f) if r["prompt"] == "old" and r["passed"] == "1"]
    allowed_ids = ai_core.collection.get(where=ai_core._build_where(None, None), include=[])["ids"]
    idx, dm = ai_core._get_bm25_index(), ai_core._get_dense_matrix()
    print(f"{len(stored)} προσπάθειες περνούν σήμερα τη δεύτερη αναζήτηση · κριτής {ai_core.GEMINI_MODEL} · "
          f"{args.draws} κληρώσεις\n")

    # 1) Σελίδες: ξανά από την αποθηκευμένη αναδιατύπωση, με έλεγχο ότι η βαθμολογία είναι ίδια.
    items, mismatched = [], []
    for r in stored:
        t = tests[(r["set"], r["id"])]
        ranked = await _retrieve(r["rewrite"], allowed_ids, idx, dm)
        best = round(float(ranked[0][0]), 3)
        if abs(best - float(r["best"])) > 1e-3:
            mismatched.append(f"{r['id']}#{r['attempt']} {r['best']}->{best}")
            continue
        pages = await asyncio.to_thread(ai_core._expand_to_pages, ranked[:ai_core.EXPAND_INPUT],
                                        ai_core.MAX_PAGES)
        items.append({"set": r["set"], "id": r["id"], "kind": kind(t), "attempt": int(r["attempt"]),
                      "best": best, "cov": float(r["cov"]), "question": t["_q"],
                      "prompt": SF.build_prompt(t["_q"], pages)})
    if mismatched:
        print(f"!! Άλλη βαθμολογία από το CSV -> εκτός ({len(mismatched)}): {mismatched}")

    # 2) Ο κριτής, με πάγωμα ανά κλήρωση.
    calls = load_calls()
    stats = {"new": 0, "frozen": 0, "errors": [], "seconds": []}
    sem = asyncio.Semaphore(CONCURRENCY)

    async def judge(item, draw):
        k = _key(ai_core.GEMINI_MODEL, str(draw), item["prompt"])
        if k in calls:
            stats["frozen"] += 1
            return calls[k]["out"]
        async with sem:
            t0 = time.perf_counter()
            try:
                out = await gemini_rest.generate_once(item["prompt"], model=ai_core.GEMINI_MODEL,
                                                      api_key=ai_core.GEMINI_API_KEY)
            except Exception as e:  # ένα σφάλμα δεν ρίχνει τη μέτρηση· μετριέται
                stats["errors"].append(f"{item['id']}#{item['attempt']}/{draw}: {type(e).__name__}")
                return None
            sec = round(time.perf_counter() - t0, 2)
        stats["new"] += 1
        stats["seconds"].append(sec)
        calls[k] = {"model": ai_core.GEMINI_MODEL, "draw": draw, "out": out, "seconds": sec,
                    "question": item["question"]}
        return out

    for d in range(1, args.draws + 1):
        raws = await asyncio.gather(*(judge(it, d) for it in items))
        for it, raw in zip(items, raws):
            it[f"raw{d}"] = raw
            it[f"v{d}"] = SF.parse_verdict(raw)
        save_calls(calls)

    # 3) Αποτελέσματα
    print(f"{'ερώτηση':<14}{'είδος':<11}{'βαθμ.':>7}{'κάλ.':>6}  " +
          "  ".join(f"κλ.{d}" for d in range(1, args.draws + 1)))
    for it in sorted(items, key=lambda x: (x["kind"], x["id"], x["attempt"])):
        print(f"{it['id'] + '#' + str(it['attempt']):<14}{it['kind']:<11}{it['best']:>+7.2f}{it['cov']:>6.0f}  " +
              "  ".join(f"{it[f'v{d}'] or '??':>4}" for d in range(1, args.draws + 1)))

    sm = summarize(items, args.draws)
    print("\nΑΝΑ ΚΛΗΡΩΣΗ ΚΑΙ ΠΡΟΣΠΑΘΕΙΑ: σήμερα -> με τον κριτή")
    ok_all = True
    for d, per in sm.items():
        print(f"  κλήρωση {d}")
        for p in per:
            print(f"    προσπάθεια {p['attempt']}: άσχετες {p['leaks'][0]}->{p['leaks'][1]} · "
                  f"κοντινές {p['near'][0]}->{p['near'][1]} · σωσμένες {p['saved'][0]}->{p['saved'][1]} · "
                  f"κάλυψη 0 {p['cov0'][0]}->{p['cov0'][1]}")
        leaks = sum(p["leaks"][1] for p in per)
        saved = statistics.mean(p["saved"][1] for p in per)
        r1, r2 = leaks <= MAX_LEAKS, saved >= MIN_SAVED_MEAN
        ok_all &= r1 and r2
        print(f"    κανόνας 1 (άσχετες = 0): {leaks} {'✓' if r1 else '✗'} · "
              f"κανόνας 2 (σωσμένες ≥ {MIN_SAVED_MEAN}): {saved:.2f} {'✓' if r2 else '✗'}")
    if args.draws >= 2:
        same = sum(it["v1"] == it["v2"] for it in items)
        agree = same / max(1, len(items))
        r3 = agree >= MIN_AGREEMENT
        ok_all &= r3
        print(f"  κανόνας 3 (ίδια ετυμηγορία ≥ {MIN_AGREEMENT:.0%}): {same}/{len(items)} = {agree:.1%} "
              f"{'✓' if r3 else '✗'}")
    unparsed = sum(it[f"v{d}"] is None for it in items for d in range(1, args.draws + 1))
    r4 = unparsed <= MAX_ERRORS
    ok_all &= r4
    print(f"  κανόνας 4 (αποτυχίες/ακατανόητες ≤ {MAX_ERRORS}): {unparsed} {'✓' if r4 else '✗'}"
          + (f" · σφάλματα: {stats['errors'][:3]}" if stats["errors"] else ""))
    if stats["seconds"]:
        print(f"  χρόνος κλήσης κριτή: διάμεσος {statistics.median(stats['seconds']):.1f} s · "
              f"μέγιστος {max(stats['seconds']):.1f} s")
    print(f"  κλήσεις κριτή: {stats['new']} νέες / {stats['frozen']} παγωμένες")
    print(f"\nΑΠΟΦΑΣΗ: {'ΥΙΟΘΕΤΕΙΤΑΙ' if ok_all and not mismatched else 'ΔΕΝ ΥΙΟΘΕΤΕΙΤΑΙ'}")

    if args.csv:
        cols = ["set", "id", "kind", "attempt", "best", "cov"] + [f"v{d}" for d in range(1, args.draws + 1)]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(items)
        print(f"→ γράφτηκε {args.csv}")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ab", default=os.path.join(RUNS, "corrective_ab.csv"))
    ap.add_argument("--run", default=os.path.join(RUNS, "scoreboard", "g38b.json"),
                    help="από εκεί οι μεταφράσεις των ελληνικών ερωτήσεων")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--csv", default=os.path.join(RUNS, "sufficiency_veto.csv"))
    args = ap.parse_args()
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"!! Τρέχει ήδη scoreboard/probe στο ίδιο store (κλειδαριά {LOCK}) — σταματάω")
        return 1
    try:
        return asyncio.run(run(args))
    finally:
        os.close(fd)
        os.remove(LOCK)


if __name__ == "__main__":
    sys.exit(main())

"""Βήμα 6: δοκιμή αποδοχής του «απόσπασμα πηγής» στις απαντήσεις του όλου κειμένου.
Προκαταχώριση: docs/SOURCE_PASSAGES_PREREG_2026-11-24.md.

ΤΙ ΚΑΝΕΙ:
- Παίρνει 20 αποθηκευμένες απαντήσεις του «όλου κειμένου» από τη Φάση Α (runs/longctx/answers_phaseA.json,
  κλήρωση 1): τις 10 πρώτες (κατά id) δύο άρθρων και τις 10 πρώτες τριών άρθρων.
- Ξαναχτίζει τις σελίδες με τις ίδιες ετικέτες [S#].
- Καλεί τον ΠΡΑΓΜΑΤΙΚΟ ai_core.get_source_passages, την ίδια συνάρτηση με το κουμπί της εφαρμογής.

ΚΑΝΟΝΑΣ: εύρεση ≥ 85% (στις προτάσεις με παραπομπή, όχι απουσίας) · p90 χρόνου ≤ 10 s · ≤ 1/20 αποτυχίες.

ΔΗΜΟΣΙΟ REPO: τα χωρία είναι αυτούσιο κείμενο άρθρων. Το runs/source_passages_calls.json (έξοδος μοντέλου)
είναι στο .gitignore, και το runs/source_passages.json κρατά μόνο καταστάσεις και χρόνους.

    docker compose -f docker-compose.yml -f docker-compose.upgrade.yml run --rm --no-deps \\
        -e DATABASE_URL=sqlite:// -e GEMINI_MODEL=gemini-3.8-flash -v eval_tmp:/tmp backend \\
        python -u evaluation/probe_source_passages.py
"""
import asyncio
import collections
import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
import lc_vs_rag_spec as SPEC

import full_text
import gemini_rest
import source_passages as SP

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
CALLS_PATH = os.path.join(RUNS, "source_passages_calls.json")
OUT_PATH = os.path.join(RUNS, "source_passages.json")
ANSWERS = os.path.join(RUNS, "longctx", "answers_phaseA.json")
PER_ARM = 10
MIN_FOUND, MAX_P90, MAX_FAIL = 0.85, 10.0, 1


class Calls:
    """Πάγωμα: κλειδί = SHA-256 του prompt· κρατά ΜΟΝΟ την έξοδο και τον χρόνο της πρώτης φοράς."""

    def __init__(self, real):
        self.real = real
        self.data = {}
        if os.path.exists(CALLS_PATH):
            with open(CALLS_PATH, encoding="utf-8") as f:
                self.data = json.load(f)
        self.new = self.hits = 0
        self.last_sec = None

    async def __call__(self, prompt, **kw):
        k = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if k in self.data:
            self.hits += 1
            self.last_sec = self.data[k]["sec"]
            return self.data[k]["out"]
        t0 = time.perf_counter()
        out = await self.real(prompt, **kw)
        self.last_sec = round(time.perf_counter() - t0, 3)
        self.data[k] = {"out": out, "sec": self.last_sec}
        self.new += 1
        return out

    def save(self):
        tmp = CALLS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, CALLS_PATH)


def pick_rows() -> list[dict]:
    with open(ANSWERS, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    out = []
    for arm in ("two", "three"):
        sel = sorted((r for r in rows if r["arm"] == arm and r["system"] == "full" and r["draw"] == 1),
                     key=lambda r: r["id"])
        out += sel[:PER_ARM]
    return out


def questions() -> dict:
    qs = {}
    for name in ("golden_multihop_v2.jsonl", "golden_three_papers.jsonl"):
        for t in SPEC.load_jsonl(name):
            qs[t["id"]] = t["question"]
    return qs


def p90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(0.9 * len(xs)) - 1)] if xs else None


async def run() -> int:
    ai_core, _saved = E.open_store()
    calls = Calls(gemini_rest.generate_once)
    gemini_rest.generate_once = calls
    metas = {}
    for m in ai_core._get_bm25_index()["metas"]:
        metas.setdefault(m["file_name"], m["doc_id"])
    qs = questions()
    results, times, failures = [], [], []
    for r in pick_rows():
        doc_ids = [metas[d] for d in r["docs"].split(";")]
        where = ai_core._build_where(None, E.TEST_USER, target_doc_ids=doc_ids)
        got = ai_core.collection.get(where=where, include=["documents", "metadatas"])
        pages = full_text.assemble_pages(got["ids"], got["documents"], got["metadatas"], doc_ids)
        sources = [{"label": f"S{i}", "file": m["file_name"], "page": m["page"], "doc_id": m["doc_id"]}
                   for i, (_t, m) in enumerate(pages, 1)]
        if len(pages) != r["n_pages"]:
            print(f"!! {r['id']}: {len(pages)} σελίδες, στο πείραμα {r['n_pages']} — οι ετικέτες δεν ταιριάζουν")
            return 1
        calls.last_sec = None
        try:
            res = await ai_core.get_source_passages(qs[r["id"]], r["answer"], sources, E.TEST_USER)
        except Exception as e:
            failures.append(f"{r['id']}: {type(e).__name__}")
            print(f"  {r['id']:<6} ΑΠΟΤΥΧΙΑ {type(e).__name__}", flush=True)
            continue
        finally:
            calls.save()
        if calls.last_sec is not None:
            times.append(calls.last_sec)
        st = collections.Counter(it["status"] for it in res.get("items", []))
        results.append({"id": r["id"], "arm": r["arm"], "status": res.get("status"),
                        "sec": calls.last_sec, "counts": dict(st)})
        print(f"  {r['id']:<6} {r['arm']:<5} {dict(st)} · {calls.last_sec} s", flush=True)

    tot = collections.Counter()
    for x in results:
        tot.update(x["counts"])
    checked = tot[SP.FOUND] + tot[SP.FOUND_ELSEWHERE] + tot[SP.NOT_FOUND]
    found = (tot[SP.FOUND] + tot[SP.FOUND_ELSEWHERE]) / checked if checked else 0.0
    t90 = p90(times)
    ok1, ok2, ok3 = found >= MIN_FOUND, t90 is not None and t90 <= MAX_P90, len(failures) <= MAX_FAIL
    print("\n" + "#" * 80)
    print(f"Προτάσεις: {dict(tot)}")
    print(f"1. εύρεση {tot[SP.FOUND] + tot[SP.FOUND_ELSEWHERE]}/{checked} = {found:.1%} ≥ {MIN_FOUND:.0%} "
          f"{'✓' if ok1 else '✗'}  (σε άλλη σελίδα από την παρατεθειμένη: {tot[SP.FOUND_ELSEWHERE]})")
    print(f"2. χρόνος p90 {t90} s ≤ {MAX_P90:g} {'✓' if ok2 else '✗'}  (διάμεσος "
          f"{sorted(times)[len(times) // 2] if times else None} s, n={len(times)})")
    print(f"3. αποτυχίες {len(failures)}/{len(results) + len(failures)} ≤ {MAX_FAIL} {'✓' if ok3 else '✗'} {failures}")
    print(f"ΑΠΟΦΑΣΗ: {'ΠΕΡΝΑ — το κουμπί ισχύει και στο όλο κείμενο' if ok1 and ok2 and ok3 else 'ΔΕΝ περνά — μόνο για απαντήσεις RAG'}")
    print(f"Κλήσεις: {calls.new} νέες / {calls.hits} παγωμένες")
    print("#" * 80)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"rows": results, "totals": dict(tot), "found_rate": round(found, 4), "p90_s": t90,
                   "failures": failures, "pass": ok1 and ok2 and ok3}, f, ensure_ascii=False, indent=1)
    print(f"→ {OUT_PATH}")
    return 0


def main() -> int:
    if not E.API_KEY:
        print("!! Λείπει GEMINI_API_KEY")
        return 1
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())

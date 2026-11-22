"""Βήμα 5: στήριξη ανά ισχυρισμό, ΕΚΤΟΣ του chat. Προκαταχώριση: docs/CLAIM_SUPPORT_PREREG_2026-11-22.md.

ΤΙ ΚΑΝΕΙ: για τις ΑΠΟΘΗΚΕΥΜΕΝΕΣ απαντήσεις ενός --label (runs/scoreboard/answers_<label>_<set>.json)
ξαναβρίσκει τις σελίδες που πήρε το μοντέλο (ίδια παγωμένη ανάκτηση με το scoreboard -> 0 κλήσεις,
και έλεγχος ότι οι σελίδες είναι ΙΔΙΕΣ με τον πίνακα του label), χωρίζει κάθε απάντηση σε
προτάσεις και τρέχει τον έλεγχο του `claim_support.py`: ΜΙΑ κλήση ανά απάντηση. Η απάντηση δεν
ξαναγράφεται.

ΤΡΙΑ ΣΥΝΟΛΑ (ορισμοί στην προκαταχώριση):
    P   μάρτυρες: απαντήσεις του 3.8 που ο GPT-4.1 έκρινε σωστές -> πρέπει να ΜΗ σημαδεύονται
    R   αρνήσεις: οι κοντινές εκτός θέματος -> οι προτάσεις άρνησης δεν είναι ισχυρισμοί
    N   αρνητικά: (α) τεχνητές αλλοιώσεις αριθμού / έτους / μονάδας σε προτάσεις του P που
        στηρίχθηκαν στην 1η κλήρωση (--mutate), (β) τα πραγματικά λάθη του 2.5 (--label baseline)

ΠΑΓΩΜΑ: οι κλήσεις του ελέγχου σε runs/claim_support_calls.json, κλειδί = μοντέλο + κλήρωση +
prompt. Το 2ο τρέξιμο = 0 κλήσεις. ΔΗΜΟΣΙΟ REPO: τα χωρία είναι αυτούσιο κείμενο άρθρων, άρα
μένουν ΜΟΝΟ στο gitignored `*_detail.csv` και στο αρχείο κλήσεων· το CSV που μπαίνει στο git δεν
τα έχει.

    # 1) μάρτυρες + αρνήσεις, 2 κληρώσεις (σταθερότητα), 3.8
    docker compose -f docker-compose.yml -f docker-compose.upgrade.yml run --rm --no-deps -v eval_tmp:/tmp \\
        -e DATABASE_URL=sqlite:// -e GEMINI_MODEL=gemini-3.8-flash backend \\
        python evaluation/probe_claim_support.py --label g38b --draws 2
    # 2) τεχνητές αλλοιώσεις (χρειάζεται το 1)
    ... python evaluation/probe_claim_support.py --label g38b --mutate
    # 3) τα πραγματικά λάθη του 2.5 (παγωμένη ανάκτηση του 2.5, ο ΕΛΕΓΧΟΣ τρέχει με 3.8)
    ... -e GEMINI_MODEL=gemini-2.5-flash ... --label baseline --sets mh_new,near_ooc \\
        --ids n024,n038,m039,m064,m071,m079
    # 4) τυφλό δείγμα για ανθρώπινη κρίση, και μετά τη συμπλήρωσή του η βαθμολόγηση
    ... --label g38b --blind 30          /   ... --score-blind
"""
import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
import scoreboard as S

import claim_support as CS
import gemini_rest

CHECK_MODEL = os.getenv("CHECK_MODEL", "gemini-3.8-flash")
CALLS_PATH = os.path.join(S.RUNS, "claim_support_calls.json")
JUDGE2_PATH = os.path.join(S.RUNS, "compare", "judge2_gpt-4.1_g38b.json")
SETS = ("main", "mh_new", "near_ooc", "tables")
SET_FILES = {"main": "golden_set_50.jsonl", "mh_new": "golden_multihop_v2.jsonl",
             "near_ooc": "golden_near_ooc.jsonl", "tables": "golden_tables.jsonl"}
BLIND_PATH = os.path.join(S.RUNS, "claim_support_blind.csv")          # gitignored
BLIND_KEY = os.path.join(S.RUNS, "claim_support_blind_key.json")      # gitignored
SEED = 20261119
CONCURRENCY = 4
_GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")


def _key(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


class Calls:
    def __init__(self, path: str):
        self.path = path
        self.data = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        self.hits = self.misses = 0
        self.errors: list[str] = []

    async def get(self, prompt: str, draw: int, call) -> tuple[str, float]:
        k = _key(CHECK_MODEL, str(draw), prompt)
        if k in self.data:
            self.hits += 1
            return self.data[k]["out"], self.data[k]["seconds"]
        t0 = time.perf_counter()
        out = await call(prompt)
        sec = round(time.perf_counter() - t0, 2)
        self.misses += 1
        self.data[k] = {"model": CHECK_MODEL, "draw": draw, "chars": len(prompt), "out": out,
                        "seconds": sec}
        return out, sec

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
        for _ in range(5):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.5)
        os.replace(tmp, self.path)


# ----------------------------------------------------------------------------- σύνολα
def positive_ids(label: str) -> dict[str, set]:
    """P: όσα ο GPT-4.1 έκρινε σωστά (main correct, mh και τα δύο correct), πίνακες με όλες τις
    αυστηρές τιμές. R: κοντινές με σωστή άρνηση."""
    out: dict[str, set] = defaultdict(set)
    if os.path.exists(JUDGE2_PATH):
        for r in json.load(open(JUDGE2_PATH, encoding="utf-8"))["rows"]:
            if r["system"] != label:
                continue
            if r["set"] == "main" and r.get("label") == "correct":
                out["main"].add(r["id"])
            elif r["set"] == "mh_new" and r.get("label_1") == r.get("label_2") == "correct":
                out["mh_new"].add(r["id"])
            elif r["set"] == "near_ooc" and r.get("label") == "correct":
                out["near_ooc"].add(r["id"])
    p = os.path.join(S.OUT_DIR, f"answers_{label}_tables.json")
    if os.path.exists(p):
        for r in json.load(open(p, encoding="utf-8"))["rows"]:
            if r.get("strict_total") and r.get("strict_ok") == r.get("strict_total"):
                out["tables"].add(r["id"])
    return out


def judge_evidence(row: dict) -> list[str]:
    """Η πρόταση της απάντησης που ο κριτής 1 είπε ότι έκρινε τη συμπεριφορά (μόνο στα παλιά)."""
    return [row[k] for k in ("evidence", "evidence_1", "evidence_2") if row.get(k)]


def _words(s: str) -> set:
    return set(re.findall(r"\w{3,}", CS.normalize_for_match(re.sub(r"\[[^\]]*\]", "", s))))


def matches_evidence(claim: str, evidence: list[str]) -> bool:
    cw = _words(claim)
    return any(cw and (len(cw & _words(ev)) / max(1, min(len(cw), len(_words(ev))))) >= 0.6
               for ev in evidence)


# ----------------------------------------------------------------------------- αλλοιώσεις
_YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)")
_UNIT_SWAPS = [(r"\bmilliseconds\b", "seconds"), (r"\bseconds\b", "minutes"), (r"\bminutes\b", "hours"),
               (r"\bhours\b", "minutes"), (r"(?<=\d)\s?ms\b", " s"), (r"\bMB\b", "GB"), (r"\bGB\b", "TB"),
               (r"\bKB\b", "MB"), (r"\bTB\b", "GB"), (r"δευτερόλεπτα", "λεπτά"),
               (r"(?<!\w)λεπτά", "ώρες"), (r"(?<!\w)ώρες", "λεπτά"), (r"\bGHz\b", "MHz"),
               (r"\bGbps\b", "Mbps"), (r"\bMbps\b", "Gbps")]


def mutate(kind: str, text: str, cited: str) -> str | None:
    """Μία αλλοίωση που αλλάζει το νόημα και δεν υπάρχει ήδη στις σελίδες. None αν δεν γίνεται."""
    have = CS.numbers_in(cited)
    if kind == "year":
        m = _YEAR.search(text)
        if not m:
            return None
        for d in (3, -3, 5, -5):
            y = str(int(m.group(1)) + d)
            if y not in have:
                return text[:m.start()] + y + text[m.end():]
        return None
    if kind == "number":
        for m in CS._NUM_RE.finditer(text):
            raw = m.group(0)
            if _YEAR.fullmatch(raw) or raw.replace(",", ".") not in have:
                continue
            if re.fullmatch(r"\d+", raw):
                n = int(raw)
                cands = [str(n * 2), str(n + 3), str(n + 7)] if n < 1000 else [str(n + n // 4)]
            else:
                sep = "," if "," in raw else "."
                head, tail = raw.rsplit(sep, 1)
                cands = [f"{head}{sep}{tail[:-1]}{(int(tail[-1]) + k) % 10}" for k in (3, 5, 7)]
            for c in cands:
                if c.replace(",", ".") not in have and c != raw:
                    return text[:m.start()] + c + text[m.end():]
        return None
    if kind == "unit":
        for pat, rep in _UNIT_SWAPS:
            if re.search(pat, text):
                new = re.sub(pat, rep, text, count=1)
                if new != text:
                    return new
        return None
    raise ValueError(kind)


# ----------------------------------------------------------------------------- ανάκτηση
async def reproduce_pages(ai_core, label: str, sets, ids) -> dict:
    """(set, id) -> (t, row, pages) μόνο όπου οι σελίδες είναι ΙΔΙΕΣ με τον πίνακα του label."""
    import eval_engine
    base_path = os.path.join(S.OUT_DIR, f"{label}.json")
    base = {(r["set"], r["id"]): r for r in json.load(open(base_path, encoding="utf-8"))["rows"]}
    out, skipped = {}, []
    for s in sets:
        ans = {r["id"]: r for r in
               json.load(open(os.path.join(S.OUT_DIR, f"answers_{label}_{s}.json"), encoding="utf-8"))["rows"]}
        for t in E.load_jsonl(os.path.join(E.HERE, SET_FILES[s])):
            if (ids and t["id"] not in ids) or t["id"] not in ans:
                continue
            res = await E.retrieve(ai_core, t["question"])
            r0 = S.make_row(s, t, t["question"], (*res, 0.0), ai_core, eval_engine.calculate_mrr,
                            eval_engine.calculate_ndcg)
            b = base.get((s, t["id"]))
            if not res[0]:
                continue                              # κομμένη: σταθερή άρνηση, τίποτα να ελεγχθεί
            if not b or b["pages"] != r0["pages"]:
                skipped.append(f"{s}/{t['id']}")
                continue
            out[(s, t["id"])] = (t, ans[t["id"]], res[0])
    if skipped:
        print(f"!! Άλλες σελίδες από τον πίνακα {label} -> εκτός: {', '.join(skipped)}")
    return out


# ----------------------------------------------------------------------------- έλεγχος
async def check_one(calls, call, draw, question, claims, texts):
    prompt = CS.build_check_prompt(question, claims, texts)
    if prompt is None:
        return CS.apply_check(claims, texts, None, question), None
    try:
        raw, sec = await calls.get(prompt, draw, call)
    except Exception as e:          # noqa: BLE001  ένα σφάλμα δεν ρίχνει όλο το τρέξιμο
        calls.errors.append(f"{type(e).__name__}: {e}"[:200])
        return CS.apply_check(claims, texts, None, question), None
    return CS.apply_check(claims, texts, CS.parse_check(raw), question), sec


def claim_rows(group, s, t, draw, claims, sec, extra=None):
    rows = []
    for c in claims:
        rows.append({"group": group, "set": s, "id": t["id"], "lang": "el" if _GREEK.search(t["question"]) else "en",
                     "draw": draw, "i": c.i, "kind": c.kind, "labels": "|".join(c.labels),
                     "status": c.status, "reason": c.reason, "support": c.support or "",
                     "quote_found": int(c.quote_found), "cite_ok": int(c.cite_ok),
                     "numbers_missing": "|".join(c.numbers_missing), "missing": c.missing,
                     "seconds": sec if c.i == 1 else "", **(extra or {}),
                     "claim": c.text, "quote": c.quote or "", "quote_label": c.quote_label or ""})
    return rows


PUBLIC_DROP = ("quote",)          # αυτούσιο κείμενο άρθρων -> ΟΧΙ στο CSV του git


def write_csv(path: str, rows: list[dict], public: bool) -> None:
    if not rows:
        return
    cols = [k for k in rows[0] if not (public and k in PUBLIC_DROP)]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict]) -> None:
    claims = [r for r in rows if r["kind"] == "claim"]
    by = defaultdict(list)
    for r in claims:
        by[(r["group"], r["draw"], r["lang"])].append(r)
        by[(r["group"], r["draw"], "όλα")].append(r)
    print(f"\n{'ομάδα':<16}{'κλ.':>3} {'γλ.':<5}{'προτ.':>6}{'(α) παρ.':>10}{'(β) σημείο':>11}"
          f"{'(γ) στηρ.':>10}{'σημαδ.':>8}  λόγοι")
    for (g, d, lang), rs in sorted(by.items()):
        n = len(rs)
        found = [r for r in rs if r["quote_found"]]
        flag = [r for r in rs if r["status"] == "flagged"]
        reasons = Counter(r["reason"] for r in flag)
        print(f"{g:<16}{d:>3} {lang:<5}{n:>6}{sum(bool(r['labels']) for r in rs) / n:>10.0%}"
              f"{(sum(r['cite_ok'] for r in found) / len(found) if found else 0):>11.0%}"
              f"{sum(r['status'] == 'supported' for r in rs) / n:>10.0%}{len(flag):>8}  "
              f"{', '.join(f'{k} {v}' for k, v in reasons.most_common())}")
    ans = defaultdict(set)
    for r in claims:
        if r["status"] == "flagged":
            ans[(r["group"], r["draw"])].add((r["set"], r["id"]))
    tot = defaultdict(set)
    for r in rows:
        tot[(r["group"], r["draw"])].add((r["set"], r["id"]))
    print("\nαπαντήσεις με ≥1 σημαδεμένη πρόταση: " + " · ".join(
        f"{g}/{d}: {len(ans[(g, d)])}/{len(v)}" for (g, d), v in sorted(tot.items())))
    secs = [float(r["seconds"]) for r in rows if r["seconds"] not in ("", None)]
    if secs:
        secs.sort()
        print(f"χρόνος κλήσης ελέγχου: διάμεσος {statistics.median(secs):.1f} s · "
              f"p95 {secs[int(0.95 * (len(secs) - 1))]:.1f} s · κλήσεις {len(secs)}")
    draws = sorted({r["draw"] for r in claims if r["group"] == "P"})
    if len(draws) >= 2:
        a = {(r["set"], r["id"], r["i"]): r["status"] for r in claims if r["group"] == "P" and r["draw"] == draws[0]}
        b = {(r["set"], r["id"], r["i"]): r["status"] for r in claims if r["group"] == "P" and r["draw"] == draws[1]}
        common = a.keys() & b.keys()
        same = sum(a[k] == b[k] for k in common)
        print(f"σταθερότητα P (κλήρωση {draws[0]} vs {draws[1]}): ίδια κρίση {same}/{len(common)} "
              f"= {same / max(1, len(common)):.1%}")
    muts = [r for r in claims if r["group"].startswith("N_")]
    if muts:
        print("\nαλλοιώσεις: πιάστηκαν (σύνολο · μόνο από την κρίση του μοντέλου, χωρίς τον έλεγχο αριθμών)")
        for g in sorted({r["group"] for r in muts}):
            rs = [r for r in muts if r["group"] == g]
            caught = sum(r["status"] == "flagged" for r in rs)
            by_model = sum(r["support"] in ("partial", "none") for r in rs)
            print(f"  {g:<12} {caught}/{len(rs)} = {caught / len(rs):.0%} · {by_model}/{len(rs)}")
    real = [r for r in claims if r["group"] == "N_real"]
    if real:
        print("\nπραγματικά λάθη (2.5): η πρόταση που έδειξε ο κριτής 1")
        for (s, qid) in sorted({(r["set"], r["id"]) for r in real}):
            rs = [r for r in real if (r["set"], r["id"]) == (s, qid)]
            ev = [r for r in rs if r.get("judge_evidence") == 1]
            print(f"  {qid}: σημαδεμένες {sum(r['status'] == 'flagged' for r in rs)}/{len(rs)} · "
                  f"πρόταση κριτή: " + (", ".join(f"#{r['i']} {r['status']}/{r['reason'] or r['support']}"
                                                 for r in ev) or "δεν ταυτίστηκε"))


# ----------------------------------------------------------------------------- τυφλό δείγμα
def make_blind(detail_rows: list[dict], n: int) -> None:
    """n προτάσεις του P, μισές σημαδεμένες και μισές όχι, μισές el και μισές en, ανακατεμένες. Ο
    άνθρωπος βλέπει ερώτηση, πρόταση, αρχείο και σελίδα — ΟΧΙ την κρίση ούτε το χωρίο."""
    rng = random.Random(SEED)
    pool = [r for r in detail_rows if r["group"] == "P" and r["draw"] == 1 and r["kind"] == "claim"]
    pick = []
    for status in ("flagged", "supported"):
        for lang in ("el", "en"):
            cand = [r for r in pool if r["status"] == status and r["lang"] == lang]
            rng.shuffle(cand)
            pick += cand[: n // 4]
    rng.shuffle(pick)
    key = []
    with open(BLIND_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "ερώτηση", "πρόταση", "σελίδες που παραπέμπει", "στηρίζεται (ναι/μερικώς/όχι)"])
        for k, r in enumerate(pick, 1):
            w.writerow([k, r["question"], r["claim"], r["pages"], ""])
            key.append({"n": k, "set": r["set"], "id": r["id"], "i": r["i"], "status": r["status"],
                        "reason": r["reason"], "lang": r["lang"]})
    json.dump(key, open(BLIND_KEY, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Τυφλό δείγμα: {len(pick)} προτάσεις -> {BLIND_PATH}\nΣυμπλήρωσε την τελευταία στήλη "
          f"ανοίγοντας τα PDF στις σελίδες, και μετά: --score-blind")


def score_blind() -> int:
    key = {k["n"]: k for k in json.load(open(BLIND_KEY, encoding="utf-8"))}
    with open(BLIND_PATH, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))[1:]
    agree, total, table = 0, 0, Counter()
    for row in rows:
        human = row[4].strip().lower()
        if not human:
            continue
        h = "supported" if human.startswith(("ν", "y")) else "flagged"
        k = key[int(row[0])]
        total += 1
        agree += h == k["status"]
        table[(k["status"], h, k["lang"])] += 1
    print(f"Συμφωνία ανθρώπου–ελέγχου: {agree}/{total}")
    for (c, h, lang), v in sorted(table.items()):
        print(f"  έλεγχος {c:<10} άνθρωπος {h:<10} {lang}: {v}")
    return 0


# ----------------------------------------------------------------------------- main
async def run(args) -> int:
    if args.score_blind:
        return score_blind()
    real_once = gemini_rest.generate_once

    async def call(prompt):
        return await real_once(prompt, model=CHECK_MODEL, api_key=E.API_KEY, temperature=0.0,
                               max_output_tokens=8192, thinking_budget=1024)

    sets = [s for s in args.sets.split(",") if s]
    ids = set(args.ids.split(",")) if args.ids else None
    real_mode = args.label != "g38b"
    ai_core, _near = E.open_store()
    if ai_core.collection.count() != S.EXPECTED_CHUNKS:
        print(f"!! Το store δεν έχει {S.EXPECTED_CHUNKS} chunks — σταματάω")
        return 1
    ai_core._translation_cache.clear()
    ai_core.ENABLE_CORRECTIVE = True
    frozen = S.Frozen(S.FROZEN_PATH, fresh=False)
    frozen.install()
    print(f"Ανάκτηση (παγωμένη, {E.MODEL}) για τις απαντήσεις του «{args.label}»· έλεγχος με {CHECK_MODEL}")
    items = await reproduce_pages(ai_core, args.label, sets, ids)
    if frozen.misses:
        print(f"!! {frozen.misses} κλήσεις ανάκτησης ΔΕΝ ήταν παγωμένες")
    pos = positive_ids(args.label)
    calls = Calls(CALLS_PATH)
    sem = asyncio.Semaphore(CONCURRENCY)
    jobs = []           # (group, s, t, draw, claims, texts, extra)
    detail = _load_detail(args.label) if args.mutate else []
    for (s, qid), (t, row, pages) in items.items():
        texts = CS.label_texts(pages)
        pages_str = "; ".join(f"{m.get('file_name')} σ.{m.get('page')}" for _x, m in pages)
        if real_mode:
            group = "N_real"
        elif s == "near_ooc":
            group = "R" if qid in pos["near_ooc"] else "other"
        else:
            group = "P" if qid in pos[s] else "other"
        if args.mutate:
            if group != "P":
                continue
            prev = {(r["i"]): r for r in detail if r["group"] == "P" and r["set"] == s
                    and r["id"] == qid and r["draw"] == 1 and r["status"] == "supported"}
            base = CS.split_claims(row["answer"], E.REFUSAL)
            for kind in ("number", "year", "unit"):
                mclaims = []
                for c in base:
                    if c.i not in prev:
                        continue
                    new = mutate(kind, c.text, "\n".join(texts.get(lab, "") for lab in c.labels))
                    if new:
                        mclaims.append(CS.Claim(i=len(mclaims) + 1, text=new, labels=c.labels))
                if mclaims:
                    jobs.append((f"N_{kind}", s, t, 1, mclaims, texts, {"pages": pages_str}))
            continue
        for draw in range(1, (args.draws if group == "P" else 1) + 1):
            claims = CS.split_claims(row["answer"], E.REFUSAL)
            extra = {"pages": pages_str}
            if real_mode:
                ev = judge_evidence(row)
                jobs.append((group, s, t, draw, claims, texts, extra | {"_ev": ev}))
            else:
                jobs.append((group, s, t, draw, claims, texts, extra))

    async def one(job):
        group, s, t, draw, claims, texts, extra = job
        async with sem:
            done, sec = await check_one(calls, call, draw, t["question"], claims, texts)
        ev = extra.pop("_ev", None)
        rows = claim_rows(group, s, t, draw, done, sec, {**extra, "question": t["question"]})
        if ev is not None:
            for r in rows:
                r["judge_evidence"] = int(matches_evidence(r["claim"], ev))
        return rows

    print(f"{len(jobs)} κλήσεις ελέγχου (το πολύ)…")
    results = await asyncio.gather(*(one(j) for j in jobs))
    calls.save()
    rows = [r for rs in results for r in rs]
    tag = args.label + ("_mutate" if args.mutate else "")
    write_csv(os.path.join(S.RUNS, f"claim_support_{tag}_detail.csv"), rows, public=False)
    write_csv(os.path.join(S.RUNS, f"claim_support_{tag}.csv"), rows, public=True)
    summarize(rows)
    print(f"\nκλήσεις ελέγχου: {calls.misses} νέες / {calls.hits} παγωμένες")
    if calls.errors:
        print(f"!! ΣΦΑΛΜΑΤΑ ({len(calls.errors)}): {calls.errors[:3]}")
    if args.blind:
        make_blind(rows, args.blind)
    return 1 if calls.errors else 0


def _load_detail(label: str) -> list[dict]:
    p = os.path.join(S.RUNS, f"claim_support_{label}_detail.csv")
    if not os.path.exists(p):
        raise SystemExit(f"!! Λείπει το {p}: τρέξε πρώτα χωρίς --mutate")
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["draw"], r["i"] = int(r["draw"]), int(r["i"])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="g38b")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--ids", default="")
    ap.add_argument("--draws", type=int, default=1, help="κληρώσεις για το P (σταθερότητα)")
    ap.add_argument("--mutate", action="store_true")
    ap.add_argument("--blind", type=int, default=0)
    ap.add_argument("--score-blind", action="store_true")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())

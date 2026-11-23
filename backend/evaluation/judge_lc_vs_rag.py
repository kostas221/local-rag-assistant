"""RAG ή ΟΛΟ το κείμενο των ΕΠΙΛΕΓΜΕΝΩΝ άρθρων; Μέρος 2 από 2: κρίση, πίνακας, κανόνας, διάγνωση.
Προκαταχώριση: docs/LONGCONTEXT_VS_RAG_PREREG_2026-11-23.md. Μέρος 1 (οι απαντήσεις): probe_lc_vs_rag.py.

ΤΡΕΧΕΙ ΣΤΟΝ ΥΠΟΛΟΓΙΣΤΗ (μόνο βασική βιβλιοθήκη της Python), όπως το compare_judge2: κλειδί OPENAI_API_KEY από
το περιβάλλον ή από το .env της ρίζας. Πάγωμα: runs/longctx/judge_cache.json (κλειδί μοντέλο + prompt) ->
ξανατρέξιμο = 0 κλήσεις. Φύλακας κόστους --max-usd (default 3).

ΕΛΕΓΧΟΣ ΠΛΗΡΟΤΗΤΑΣ: χωρίς --smoke, ελέγχει ΠΡΙΝ καλέσει τον κριτή ή τυπώσει απόφαση ότι το answers.json
έχει meta.complete=True, ακριβώς τις 654 μοναδικές προκαταχωρισμένες γραμμές (μαζί με τις σιωπές του φύλακα)
και — αν είναι ενεργό — ακριβώς τις 56 μοναδικές γραμμές του χωριστού σκέλους distractor.

ΚΡΙΣΗ (ίδιοι κανόνες με το compare_judge2, ως προς την απάντηση αναφοράς, χωρίς context):
    κύριο       MAIN_J2: correct · partial · refused · wrong                     σωστή = correct
    πίνακες     table_scoring.check_strict, χωρίς κριτή                         σωστή = όλες οι τιμές
    2 άρθρα     MH_J2, ανά άρθρο                                                σωστή = και τα δύο correct
    3 άρθρα     MH3_J2 (το MH_J2 για τρία άρθρα)                                σωστή = και τα τρία correct
    κοντινές    ο κριτής της Φάσης 0.1 αυτούσιος (NEAR_J)                       σωστή άρνηση = correct
    άσχετες     MAIN_J2 (η αναφορά λέει ότι τα έγγραφα δεν απαντούν)            σωστή άρνηση = correct|refused
    Σιωπή του φύλακα (RAG): στο κύριο «refused», στα 2/3 άρθρα «declared_missing», στην ασφάλεια σωστή.

    python backend/evaluation/judge_lc_vs_rag.py --smoke     # η δοκιμή (answers_smoke.json)
    python backend/evaluation/judge_lc_vs_rag.py             # όλο (answers.json)
"""
import argparse
import json
import math
import os
import statistics
import sys

import compare_judge2 as C
import lc_vs_rag_spec as SPEC
import table_scoring as TS

HERE = os.path.dirname(os.path.abspath(__file__))
LC_DIR = SPEC.LC_DIR
C.CMP = LC_DIR                                         # ο Judge γράφει το πάγωμά του εδώ, όχι στο compare/
C.CACHE_PATH = os.path.join(LC_DIR, "judge_cache.json")
GOLD = {"main": "golden_set_50.jsonl", "ooc": "golden_set_50.jsonl", "tables": "golden_tables.jsonl",
        "mh_new": "golden_multihop_v2.jsonl", "t3p": "golden_three_papers.jsonl",
        "near_ooc": "golden_near_ooc.jsonl"}
SYSTEMS = (("rag", "RAG"), ("full", "Όλο το κείμενο"))
SIZES = (1, 2, 3)
QUALITY_ARM = {1: "one", 2: "two", 3: "three"}
# Ο κανόνας της προκαταχώρισης
MIN_GAIN, SAFETY_SLACK, MAX_COST_RATIO, MAX_P90_TTFT, ROUND_TOKENS = 2.0, 1, 3.0, 15.0, 5000

MH3_J2 = """You are grading one answer of a question-answering system that must answer ONLY from a \
fixed set of research papers.

The QUESTION needs information from THREE papers:
  PAPER 1 = {name1}
  PAPER 2 = {name2}
  PAPER 3 = {name3}

QUESTION: {question}
REFERENCE ANSWER (written by reading the three complete papers; may be in Greek): {reference}
Key terms expected in the part about PAPER 1: {kw1}
Key terms expected in the part about PAPER 2: {kw2}
Key terms expected in the part about PAPER 3: {kw3}

SYSTEM ANSWER (may be in Greek; markers like [S3] are citations and do not matter):
<<<
{answer}
>>>

For EACH paper, label the part of the SYSTEM ANSWER about that paper with exactly one of:
- "correct": it gives that paper's part of the reference answer.
- "partial": it gives only some of that part, or related information from that paper that misses \
the main point of the reference.
- "declared_missing": it clearly says that the documents do not contain that part.
- "omitted": it does not address that paper's part and does not say that it is missing.
- "wrong": it states something about that paper's part that contradicts the reference, or \
attributes to that paper something that belongs to another paper.
If the answer gives some useful information for a part AND says the rest is missing, the label is \
"partial". Judge ONLY against the REFERENCE ANSWER. Ignore length, style, language and citations.

Return ONLY JSON, no prose:
{{"paper_1": "<label>", "paper_2": "<label>", "paper_3": "<label>",
 "evidence_1": "<ONE sentence copied VERBATIM from the SYSTEM ANSWER that decided the label of PAPER 1, or empty>",
 "evidence_2": "<the same for PAPER 2>",
 "evidence_3": "<the same for PAPER 3>"}}
"""


def lab(j: dict, field: str) -> str:
    return str(j.get(field, "")).strip().lower()


def judge_row(judge: C.Judge, r: dict, gold: dict) -> dict:
    t = gold[r["set"]][r["id"]]
    ans, docs, cut = r["answer"], r["docs"].split(";"), bool(r["cut"])
    keep_fields = (
        "arm", "set", "id", "size", "lang", "docs", "draw", "system", "cut",
        "in_tokens", "cached_tokens", "out_tokens", "usd", "ttft", "search_s",
    )
    opt_fields = (
        "cache_role", "e2e_ttft", "e2e_live", "prep_s",
        "ret_live_calls", "ret_frozen_calls", "ret_live_s", "ret_usd", "target_doc",
    )
    out = {k: r[k] for k in keep_fields}
    for k in opt_fields:
        if k in r:
            out[k] = r[k]
    if r["set"] in ("main", "ooc"):
        if cut:
            label = "cut" if r["set"] == "ooc" else "refused"
        else:
            label = lab(judge(C.MAIN_J2.format(question=t["question"], reference=t["reference_answer"],
                                               answer=ans)), "label")
        ok_labels = ("correct", "refused", "cut") if r["set"] == "ooc" else ("correct",)
        out |= {"labels": [label], "ok": label in ok_labels,
                "refused": label == "refused"}
    elif r["set"] == "tables":
        strict = [] if cut else TS.check_strict(ans, t["facts"])
        out |= {"labels": [f"{sum(strict)}/{len(t['facts'])}"], "ok": bool(strict) and all(strict),
                "refused": cut or (bool(r["refusal_re"]) and not any(strict))}
    elif r["set"] in ("mh_new", "t3p"):
        if cut:
            labels = ["declared_missing"] * len(docs)
        else:
            names = {f"name{i}": C.NAMES[d] for i, d in enumerate(docs, 1)}
            kws = {f"kw{i}": ", ".join(t["keywords_by_doc"][d]) for i, d in enumerate(docs, 1)}
            tmpl, question = (C.MH_J2, t["question_en"]) if r["set"] == "mh_new" else (MH3_J2, t["question"])
            j = judge(tmpl.format(question=question, reference=t["reference_answer"], answer=ans,
                                  **names, **kws))
            labels = [lab(j, f"paper_{i}") for i in range(1, len(docs) + 1)]
        got = set(r["pages"].split(";")) if r["system"] == "rag" else None
        ev = [1 if got is None else int(any(ep in got for ep in t["evidence_pages"] if ep.startswith(d + ":")))
              for d in docs]
        out |= {"labels": labels, "ok": all(x == "correct" for x in labels),
                "refused": all(x == "declared_missing" for x in labels), "ev": ev}
    elif r["set"] == "near_ooc":
        if cut:
            label = "cut"
        else:
            label = lab(judge(C.NEAR_J.format(focus=t["focus"], note=t.get("review_note") or "",
                                              question=t["question"], question_en=t["question_en"],
                                              answer=ans)), "label")
        out |= {"labels": [label], "ok": label in ("correct", "cut"), "refused": False}
    else:
        raise ValueError(r["set"])
    return out


def p90(xs: list) -> float | None:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(0.9 * len(xs)) - 1)] if xs else None


def stats(rows: list[dict]) -> dict:
    """Ανά (μέγεθος, σύστημα): ποιότητα από το προκαταχωρισμένο σκέλος του μεγέθους (one/two/three),
    ασφάλεια από το σκέλος safety, κόστος/χρόνος από τις κλήσεις του σκέλους ποιότητας.
    Το χωριστό σκέλος distractor ΔΕΝ μπαίνει ποτέ εδώ."""
    med = statistics.median
    S = {}
    for s in SIZES:
        for sysk, _name in SYSTEMS:
            q = [r for r in rows if r["arm"] == QUALITY_ARM[s] and r["system"] == sysk]
            sa = [r for r in rows if r["arm"] == "safety" and r["size"] == s and r["system"] == sysk]
            called = [r for r in q if not r["cut"]]
            draws = len({r["draw"] for r in q}) or 1
            tok = [r["in_tokens"] for r in called if r["in_tokens"]]
            tt = [r["ttft"] for r in called if r["ttft"] is not None]
            S[s, sysk] = {
                "n": len(q) // draws, "draws": draws, "ok": sum(r["ok"] for r in q) / draws,
                "refused": sum(r["refused"] for r in q) / draws,
                "safe": sum(r["ok"] for r in sa), "n_safe": len(sa),
                "tok": med(tok) if tok else None, "max_tok": max(tok) if tok else None,
                "cached": sum(r["cached_tokens"] for r in called) / max(1, sum(tok)) if tok else None,
                "usd": med(r["usd"] for r in called) if called else None,
                "ttft": med(tt) if tt else None, "p90": p90(tt),
                "search": med(r["search_s"] for r in q if r.get("search_s") is not None)
                          if any(r.get("search_s") is not None for r in q) else None,
            }
    return S


def art(s: int) -> str:
    return f"{s} άρθρ{'ο' if s == 1 else 'α'}"


def fmt(x, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def print_table(S: dict) -> None:
    head = f"{'':<40}" + "".join(f"{art(s):>14}" for s in SIZES)
    print("\n" + "#" * 82 + "\n" + head)

    def block(title: str, cell) -> None:
        print(title)
        for sysk, name in SYSTEMS:
            print(f"  {name:<38}" + "".join(f"{cell(S[s, sysk]):>14}" for s in SIZES))

    block("Ποιότητα: σωστές ανά κλήρωση", lambda d: f"{d['ok']:.1f}/{d['n']}" if d["n"] else "-")
    block("Αρνήσεις σε ερωτήσεις με απάντηση", lambda d: f"{d['refused']:.1f}" if d["n"] else "-")
    block("Ασφάλεια: σωστές αρνήσεις", lambda d: f"{d['safe']}/{d['n_safe']}" if d["n_safe"] else "-")
    block("Tokens εισόδου ανά ερώτηση (διάμεσος)", lambda d: fmt(d["tok"], ",.0f"))
    block("  από το cache του Gemini", lambda d: fmt(d["cached"], ".0%"))
    block("Κόστος ανά ερώτηση (διάμεσος, $)", lambda d: fmt(d["usd"], ".4f"))
    block("TTFT διάμεσος / p90 (s)",
          lambda d: f"{d['ttft']:.1f} / {d['p90']:.1f}" if d["ttft"] is not None else "-")
    print("  (RAG: + αναζήτηση, διάμεσος s)" + " " * 9
          + "".join(f"{fmt(S[s, 'rag']['search'], '.2f'):>14}" for s in SIZES))
    print("#" * 82)


def _role_stats(rs: list[dict]) -> str:
    if not rs:
        return "-"
    med_u = statistics.median(r["usd"] for r in rs)
    tok = sum(r["in_tokens"] or 0 for r in rs)
    cache = sum(r["cached_tokens"] or 0 for r in rs) / max(1, tok) if tok else 0.0
    return f"{med_u:.4f} ({cache:.0%}, n={len(rs)})"


def print_cost_breakdown(rows: list[dict]) -> None:
    """Κόστος πρώτης ερώτησης στα PDF, επόμενης διαφορετικής ερώτησης στα ίδια PDF και επανάληψης (κλ. 2)."""
    print("\nΚΟΣΤΟΣ & CACHE ΑΝΑ ΡΟΛΟ ΕΡΩΤΗΣΗΣ (διάμεσο $, % cache, n · μόνο κλήσεις ποιότητας)")
    roles = (
        ("first", "1η ερώτηση στα συγκεκριμένα PDF"),
        ("next", "Επόμενη διαφορετική ερώτηση στα ίδια PDF"),
        ("draw1", "1η + επόμενη (κλ. 1, χωρίς επανάληψη)"),
        ("repeat", "Επανάληψη ίδιας ερώτησης (κλ. 2)"),
    )
    head = f"{'':<42}" + "".join(f"{art(s):>20}" for s in SIZES)
    print(head)
    for rkey, title in roles:
        print(title)
        for sysk, name in SYSTEMS:
            cells = []
            for s in SIZES:
                q = [r for r in rows if r["arm"] == QUALITY_ARM[s] and r["system"] == sysk and not r["cut"]]
                if rkey == "draw1":
                    sub = [r for r in q if r.get("cache_role") in ("first", "next") or (
                        "cache_role" not in r and r["draw"] == 1
                    )]
                else:
                    sub = [r for r in q if r.get("cache_role") == rkey]
                cells.append(f"{_role_stats(sub):>20}")
            print(f"  {name:<40}" + "".join(cells))


def print_wait_time(rows: list[dict], live_timing: list[dict]) -> None:
    """Χρόνος αναμονής χρήστη (έναρξη ερώτησης -> 1ο ορατό κείμενο), με p90 στους συνολικούς χρόνους."""
    med = statistics.median
    print("\nΧΡΟΝΟΣ ΑΝΑΜΟΝΗΣ ΧΡΗΣΤΗ (έναρξη ερώτησης -> 1ο ορατό κείμενο · διάμεσος / p90 σε s)")
    if live_timing:
        print("  Ζωντανό δείγμα (χωρίς παγωμένες κλήσεις ανάκτησης — περιλαμβάνει πραγματικό δίκτυο):")
        for s in SIZES:
            sub = [r for r in live_timing if r["size"] == s]
            if not sub:
                continue
            r_tot = [r["rag_e2e_ttft"] for r in sub if r.get("rag_e2e_ttft") is not None]
            f_tot = [r["full_e2e_ttft"] for r in sub if r.get("full_e2e_ttft") is not None]
            r_srch = [r["rag_search_s"] for r in sub if r.get("rag_search_s") is not None]
            api_n = sum(r.get("rag_ret_calls", 0) for r in sub)
            print(
                f"    {art(s)} (n={len(sub)}, ζωντανές κλήσεις ανάκτησης {api_n}): "
                f"RAG {med(r_tot):.2f} / p90 {p90(r_tot):.2f} s (αναζήτηση διάμ. {med(r_srch):.2f} s) · "
                f"Όλο {med(f_tot):.2f} / p90 {p90(f_tot):.2f} s"
            )
        r_all = [r["rag_e2e_ttft"] for r in live_timing if r.get("rag_e2e_ttft") is not None]
        f_all = [r["full_e2e_ttft"] for r in live_timing if r.get("full_e2e_ttft") is not None]
        if r_all and f_all:
            print(
                f"    Σύνολο ζωντανού δείγματος (n={len(live_timing)}): "
                f"RAG {med(r_all):.2f} / p90 {p90(r_all):.2f} s · "
                f"Όλο {med(f_all):.2f} / p90 {p90(f_all):.2f} s"
            )
    else:
        print("  (δεν υπάρχει ζωντανό δείγμα στο αρχείο)")

    # Συνολικοί χρόνοι ανά ερώτηση από το κύριο τρέξιμο (p90 πάνω στο per-question e2e_ttft)
    has_e2e = any(r.get("e2e_ttft") is not None for r in rows)
    if has_e2e:
        print("  Κύριο τρέξιμο (per-question search_s/prep_s + TTFT · p90 στους συνολικούς χρόνους):")
        for s in SIZES:
            parts = []
            for sysk, name in SYSTEMS:
                q = [r for r in rows if r["arm"] == QUALITY_ARM[s] and r["system"] == sysk
                     and r.get("e2e_ttft") is not None]
                live_q = [r for r in q if r.get("e2e_live")]
                if not q:
                    continue
                vals = [r["e2e_ttft"] for r in q]
                msg = f"{name} {med(vals):.2f} / p90 {p90(vals):.2f} s"
                if sysk == "rag" and live_q:
                    lv = [r["e2e_ttft"] for r in live_q]
                    msg += f" (χωρίς παγ. ανάκτηση n={len(lv)}: {med(lv):.2f} / p90 {p90(lv):.2f} s)"
                parts.append(msg)
            if parts:
                print(f"    {art(s)}: " + " · ".join(parts))


def print_distractor(rows: list[dict]) -> None:
    """Χωριστό διαγνωστικό σκέλος: μονοαρθρικές ερωτήσεις με +1 ή +2 επιπλέον επιλεγμένα PDF."""
    dist = [r for r in rows if r["arm"] == "distract"]
    if not dist:
        return
    qids = sorted({r["id"] for r in dist})
    print("\n" + "-" * 82)
    print(f"ΔΙΑΓΝΩΣΤΙΚΟ ΣΚΕΛΟΣ DISTRACTOR (χωριστό — ΔΕΝ μπαίνει στον προκαταχωρισμένο κανόνα, n={len(qids)})")
    print("  Γνωστές μονοαρθρικές ερωτήσεις: μόνο το άρθρο τους (1) vs +1 επιπλέον PDF (2) vs +2 (3)")
    cols = ((1, "1 άρθρο (one)"), (2, "2 άρθρα (+1)"), (3, "3 άρθρα (+2)"))
    head = f"{'':<36}" + "".join(f"{label:>16}" for _s, label in cols)
    print(head)

    def subset(s: int, sysk: str) -> list[dict]:
        if s == 1:
            return [r for r in rows if r["arm"] == "one" and r["id"] in qids and r["system"] == sysk]
        return [r for r in dist if r["size"] == s and r["system"] == sysk]

    for title, fn in (
        ("Σωστές (ανά κλήρωση)", lambda rs: (
            f"{sum(r['ok'] for r in rs) / (len({r['draw'] for r in rs}) or 1):.1f}/"
            f"{len(rs) // (len({r['draw'] for r in rs}) or 1)}"
            if rs else "-"
        )),
        ("Αρνήσεις (ανά κλήρωση)", lambda rs: (
            f"{sum(r['refused'] for r in rs) / (len({r['draw'] for r in rs}) or 1):.1f}"
            if rs else "-"
        )),
        ("Κόστος διάμ. ($)", lambda rs: (
            f"{statistics.median(r['usd'] for r in rs if not r['cut']):.4f}"
            if any(not r["cut"] for r in rs) else "-"
        )),
        ("Cache (%)", lambda rs: (
            f"{sum(r['cached_tokens'] for r in rs if not r['cut']) / max(1, sum(r['in_tokens'] or 0 for r in rs if not r['cut'])):.0%}"
            if any(not r["cut"] and r.get("in_tokens") for r in rs) else "-"
        )),
    ):
        print(title)
        for sysk, name in SYSTEMS:
            print(f"  {name:<34}" + "".join(f"{fn(subset(s, sysk)):>16}" for s, _l in cols))

    by = {(r["id"], r["size"], r["system"]): r for r in dist}
    for s in (2, 3):
        won, lost = [], []
        for qid in qids:
            f_row, r_row = by.get((qid, s, "full")), by.get((qid, s, "rag"))
            if not f_row or not r_row or f_row["ok"] == r_row["ok"]:
                continue
            (won if f_row["ok"] else lost).append(qid)
        print(f"  ζευγαρωτά {art(s)}: όλο κερδίζει {len(won)} / χάνει {len(lost)} "
              f"(p={C.binom_p(len(won), len(lost)):.2f})  κερδίζει: {won}  χάνει: {lost}")
    print("-" * 82)


def decide(S: dict, smoke: bool) -> None:
    print("\nΚΑΝΟΝΑΣ (προκαταχώριση): το «όλο το κείμενο» περνά σε ένα μέγεθος αν ισχύουν ΚΑΙ τα τέσσερα")
    passed = {}
    for s in SIZES:
        rag, full = S[s, "rag"], S[s, "full"]
        if not (rag["n"] and full["n"] and rag["n_safe"] and full["n_safe"]):
            print(f"  {art(s)}: λείπουν δεδομένα")
            continue
        c1 = full["ok"] >= rag["ok"] + MIN_GAIN
        c2 = full["safe"] >= rag["safe"] - SAFETY_SLACK
        c3 = rag["usd"] is not None and full["usd"] is not None and full["usd"] <= MAX_COST_RATIO * rag["usd"]
        c4 = full["p90"] is not None and full["p90"] <= MAX_P90_TTFT
        passed[s] = c1 and c2 and c3 and c4
        ratio = full["usd"] / rag["usd"] if (rag["usd"] and full["usd"]) else None

        def m(ok):
            return "✓" if ok else "✗"
        print(f"  {art(s)}: 1 ποιότητα {full['ok']:.1f} ≥ {rag['ok']:.1f}+{MIN_GAIN:g} {m(c1)} · "
              f"2 ασφάλεια {full['safe']} ≥ {rag['safe']}−{SAFETY_SLACK} {m(c2)} · "
              f"3 κόστος {fmt(ratio, '.1f')}× ≤ {MAX_COST_RATIO:g}× {m(c3)} · "
              f"4 p90 {fmt(full['p90'], '.1f')} s ≤ {MAX_P90_TTFT:g} {m(c4)}  ->  {'ΠΕΡΝΑ' if passed[s] else 'όχι'}")
    if smoke:
        print("  (δοκιμή: λίγες ερωτήσεις — ο κανόνας ΔΕΝ εφαρμόζεται, μόνο ελέγχεται ότι όλα δουλεύουν)")
        return
    if len(passed) != len(SIZES):
        print("!! Λείπουν δεδομένα για ένα ή περισσότερα μεγέθη — ΔΕΝ εκδίδεται απόφαση.")
        return
    ok_sizes = [s for s in SIZES if passed.get(s)]
    if not ok_sizes:
        print("ΑΠΟΦΑΣΗ: σε κανένα μέγεθος -> μένει το RAG.")
        return
    top = max(ok_sizes)
    T = int(S[top, "full"]["max_tok"] // ROUND_TOKENS * ROUND_TOKENS)
    lo = min(ok_sizes)
    gaps = [s for s in SIZES if lo <= s <= top and not passed.get(s)]
    print(f"ΑΠΟΦΑΣΗ: περνά στα μεγέθη {ok_sizes} -> όλο το κείμενο όταν τα επιλεγμένα έχουν μαζί ≤ {T:,} tokens"
          + (f" και είναι ≥ {lo} άρθρα" if lo > 1 else "") + ", αλλιώς RAG."
          + (f" ΠΡΟΣΟΧΗ: ενδιάμεσο μέγεθος {gaps} δεν πέρασε — γράφεται, δεν εξομαλύνεται." if gaps else ""))
    print("  Πριν από την εφαρμογή: ξανά τα σετ άρνησης στον νέο δρόμο· τελική επιβεβαίωση στο κλειστό σετ.")


def paired(rows: list[dict]) -> None:
    print("\nΖΕΥΓΑΡΩΤΑ (ίδια ερώτηση, ίδια κλήρωση): όλο το κείμενο κερδίζει / χάνει")
    by = {(r["arm"], r["id"], r["size"], r["draw"], r["system"]): r for r in rows}
    for arm in ("one", "two", "three", "safety"):
        w = l_ = 0
        lost, won = [], []
        for (a, qid, size, draw, sysk), r in by.items():
            if a != arm or sysk != "full":
                continue
            o = by.get((a, qid, size, draw, "rag"))
            if o is None or r["ok"] == o["ok"]:
                continue
            if r["ok"]:
                w += 1
                won.append(f"{qid}" + (f"/{size}" if arm == "safety" else "") + f"·{draw}")
            else:
                l_ += 1
                lost.append(f"{qid}" + (f"/{size}" if arm == "safety" else "") + f"·{draw}")
        print(f"  {arm:<7} {w} / {l_}  (p={C.binom_p(w, l_):.2f})  κερδίζει: {won}  χάνει: {lost}")


def diagnose(rows: list[dict]) -> None:
    """Προκαθορισμένη: στα 2/3 άρθρα, όπου ένα σύστημα σωστό και το άλλο όχι — είχε έρθει στο RAG η
    σελίδα-τεκμήριο κάθε άρθρου; Ανά άρθρο: λείπει -> φταίει η αναζήτηση · ήταν εκεί -> το μοντέλο."""
    print("\nΔΙΑΓΝΩΣΗ (2 και 3 άρθρα): ετικέτα RAG -> όλο το κείμενο · ev = ήρθε στο RAG η σελίδα-τεκμήριο")
    by = {(r["arm"], r["id"], r["draw"], r["system"]): r for r in rows if r["arm"] in ("two", "three")}
    tally = {"full": [0, 0], "rag": [0, 0]}     # σύστημα που κέρδισε το ΜΕΡΟΣ -> [ev λείπει, ev υπήρχε]
    for (arm, qid, draw, sysk), full in sorted(by.items()):
        if sysk != "full":
            continue
        rag = by.get((arm, qid, draw, "rag"))
        if rag is None or full["ok"] == rag["ok"]:
            continue
        parts = []
        for i, d in enumerate(full["docs"].split(";")):
            a, b, ev = rag["labels"][i], full["labels"][i], rag["ev"][i]
            if (a == "correct") != (b == "correct"):
                tally["full" if b == "correct" else "rag"][ev] += 1
            parts.append(f"{d.split('.')[0][:12]}: {a}->{b} ev{'✓' if ev else '✗'}")
        print(f"  {qid} κλ.{draw} {'ΟΛΟ' if full['ok'] else 'RAG'} σωστό · " + " | ".join(parts))
    for who, (miss, had) in tally.items():
        name = "στο «όλο το κείμενο»" if who == "full" else "στο RAG"
        print(f"  μέρη σωστά μόνο {name}: η σελίδα-τεκμήριο έλειπε από το RAG σε {miss}, ήταν εκεί σε {had}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):      # κονσόλα Windows (cp1253): ≥ ✓ × δεν τυπώνονται αλλιώς
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="κρίνει το answers_smoke.json (δοκιμή)")
    ap.add_argument("--input", default="", help="προαιρετική διαδρομή answers JSON αντί για το default")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--max-usd", type=float, default=3.0)
    args = ap.parse_args(argv)
    src = args.input or os.path.join(LC_DIR, "answers_smoke.json" if args.smoke else "answers.json")
    if not os.path.exists(src):
        print(f"!! Λείπει το {src} — τρέξε πρώτα το probe_lc_vs_rag.py")
        return 1
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    # Έλεγχος πληρότητας ΠΡΙΝ από οποιαδήποτε πληρωμένη κλήση κριτή ή έκδοση απόφασης
    errs = SPEC.validate_answers(data, smoke=args.smoke)
    if errs:
        print(f"!! ΑΚΥΡΟ/ΗΜΙΤΕΛΕΣ {src} — ΔΕΝ κρίνεται και ΔΕΝ εκδίδεται απόφαση:")
        for e in errs:
            print(f"   - {e}")
        return 1

    key = C.api_key()
    if not key:
        print("!! Λείπει OPENAI_API_KEY (περιβάλλον ή .env της ρίζας)")
        return 1
    gold = {sk: C.load_jsonl(fn) for sk, fn in GOLD.items()}
    judge = C.Judge(args.model, key, args.max_usd)
    out = []
    try:
        for n, r in enumerate(data["rows"], 1):
            out.append(judge_row(judge, r, gold))
            if judge.new and judge.new % 20 == 0:
                judge.save()
            if n % 25 == 0 or n == len(data["rows"]):
                print(f"  κρίθηκαν {n}/{len(data['rows'])} · νέες {judge.new} · ~{judge.usd:.3f} $", flush=True)
    finally:
        judge.save()
    S = stats(out)
    print_table(S)
    decide(S, args.smoke)
    print_cost_breakdown(out)
    print_wait_time(out, data.get("live_timing") or [])
    print_distractor(out)
    paired(out)
    diagnose(out)
    m = data.get("meta") or {}
    print(f"\nGemini (μέρος 1): ~{m.get('spent_usd')} $ · κριτής: {judge.new} νέες / {judge.hits} παγωμένες "
          f"κρίσεις · ~{judge.usd:.3f} $ τώρα")
    dst = src.replace("answers", "judged") if "answers" in os.path.basename(src) else src + ".judged.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(
            {"meta": m | {"judge": args.model}, "rows": out, "live_timing": data.get("live_timing") or []},
            f,
            ensure_ascii=False,
            indent=1,
        )
    print(f"→ {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

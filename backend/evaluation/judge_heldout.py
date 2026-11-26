"""Βήμα 7: κρίση, αναφορά και φύλλο ανθρώπινου ελέγχου (host, Git Bash, μόνο stdlib).

    python backend/evaluation/judge_heldout.py [--smoke] [--max-usd 3]      # κρίση + αναφορά + φύλλο
    python backend/evaluation/judge_heldout.py [--smoke] --agreement        # μετά τον έλεγχό σου

ΚΡΙΤΗΣ: GPT-4.1 ως προς την απάντηση αναφοράς, οι ίδιες οδηγίες με το compare_judge2 (MAIN_J2, NEAR_J).
Στις ερωτήσεις δύο άρθρων το MH_J2 παίρνει, στη θέση των λέξεων-κλειδιών που δεν υπάρχουν στο κλειστό
σετ, τα αποσπάσματα-τεκμήρια και τη λίστα «S3 = ποιο άρθρο, ποια σελίδα». Οι πίνακες: αυτόματος,
αυστηρός έλεγχος του αριθμού (heldout_common.check_table_values) ΚΑΙ ο κριτής για μονάδα και γραμμή.
Η cache του κριτή είναι ιδιωτική (private/judge_cache.json), όχι η κοινή του compare_judge2 που είναι
στο git.

ΕΞΟΔΟΣ: μόνο αριθμοί και id. Τα αποτελέσματα χωρίς κείμενα: runs/heldout*/results.json (στο git). Το
φύλλο ελέγχου και οι κρίσεις με κείμενα: private/ (εκτός git).
"""
import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import compare_judge2 as C2  # noqa: E402
import heldout_common as H  # noqa: E402

MAIN_LABELS = ("correct", "partial", "refused", "wrong")
MH_LABELS = ("correct", "partial", "declared_missing", "omitted", "wrong")
NEAR_LABELS = ("correct", "soft_leak", "leak")

_MH_OLD = ("Key terms expected in the part about PAPER 1: {kw1}\n"
           "Key terms expected in the part about PAPER 2: {kw2}\n")
_MH_NEW = ("Passage of PAPER 1 that supports its part (verbatim from the paper, may be empty): {kw1}\n"
           "Passage of PAPER 2 that supports its part (verbatim from the paper, may be empty): {kw2}\n"
           "Source labels cited in the SYSTEM ANSWER: {labels}\n")
assert _MH_OLD in C2.MH_J2, "το MH_J2 άλλαξε: έλεγξε την προσαρμογή"
MH_HELDOUT = C2.MH_J2.replace(_MH_OLD, _MH_NEW)


# -------------------------------------------------------------------------------------------- φόρτωση

def load(mode: str):
    cfg = H.MODES[mode]
    items = {it["id"]: it for it in H.load_items(cfg["csv"])}
    with open(os.path.join(H.private_dir(mode), "answers.json"), encoding="utf-8") as f:
        answers = json.load(f)
    if cfg["sources"]:
        with open(cfg["sources"], encoding="utf-8") as f:
            src = json.load(f)
        titles = {k: v["title"] for k, v in src.items()}
        topics = {k: v.get("topic", "") for k, v in src.items()}
    else:
        titles = dict(C2.NAMES)
        topics = dict.fromkeys(titles, "cloud")
    return items, answers, titles, topics


def label_list(item: dict, row: dict, titles: dict) -> str:
    if not row.get("sources"):
        return "(the answer has no sources)"
    by_label = {s["label"]: s for s in row["sources"]}
    parts = []
    for lab in row.get("cited") or []:
        s = by_label.get(lab)
        if not s:
            continue
        if s["file"] == item["doc1"]:
            who = "PAPER 1"
        elif s["file"] == item["doc2"]:
            who = "PAPER 2"
        else:
            who = f"another paper ({titles.get(s['file'], s['file'])})"
        parts.append(f"{lab} = {who}, page {s['page']}")
    return "; ".join(parts) or "(none)"


def _label(out: dict, key: str, allowed: tuple) -> str:
    v = str((out or {}).get(key, "")).strip().lower()
    return v if v in allowed else f"invalid:{v[:20]}"


def judge_row(judge, item: dict, arm: str, row: dict, titles: dict) -> dict:
    """Οι κρίσεις μιας απάντησης, με την αντιστοίχιση της προκαταχώρισης."""
    q, ref, ans, t = item["question"], item["reference"], row["answer"], item["type"]
    j, raw = {}, {}
    if t in ("single", "table") or (t == "near" and item["behaviour"] == "partial"):
        raw["main"] = judge(C2.MAIN_J2.format(question=q, reference=ref, answer=ans))
        j["main"] = _label(raw["main"], "label", MAIN_LABELS)
    if t == "two_papers":
        raw["mh"] = judge(MH_HELDOUT.format(
            name1=titles.get(item["doc1"], item["doc1"]), name2=titles.get(item["doc2"], item["doc2"]),
            question=q, reference=ref, kw1=item["quote1"] or "(not given)", kw2=item["quote2"] or "(not given)",
            labels=label_list(item, row, titles), answer=ans))
        j["mh"] = {"paper_1": _label(raw["mh"], "paper_1", MH_LABELS),
                   "paper_2": _label(raw["mh"], "paper_2", MH_LABELS)}
    if t == "near" and arm != "closed":
        raw["near"] = judge(C2.NEAR_J.format(focus=item["missing"], note=ref, question=q, question_en=q,
                                             answer=ans))
        j["near"] = _label(raw["near"], "label", NEAR_LABELS)
    if t == "table":
        j["table"] = H.check_table_values(item["values"], ans)
    return {"j": j, "raw": raw}


# -------------------------------------------------------------------------------------------- αναφορά

def rate(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(r["correct"] is True for r in rows)
    tech = sum(r["correct"] is None for r in rows)
    lo, hi = H.wilson(k, n)
    return {"correct": k, "not_correct": n - k - tech, "technical": tech, "n": n,
            "rate": round(k / n, 3) if n else None,
            "wilson95": [round(lo, 3), round(hi, 3)] if n else None}


def fmt(r: dict) -> str:
    if not r["n"]:
        return "—"
    w = r["wilson95"]
    t = f", τεχν. {r['technical']}" if r["technical"] else ""
    return f"{r['correct']}/{r['n']} ({100 * r['rate']:.0f}%, Wilson {100 * w[0]:.0f}–{100 * w[1]:.0f}%{t})"


def build_report(items, answers, judged, topics, mode: str, model_versions) -> tuple[dict, list[dict]]:
    pub = []
    for key, row in answers["rows"].items():
        it = items[row["id"]]
        arm = row["arm"]
        j = judged.get(key, {}).get("j", {})
        d = H.decide(it, arm, row, j)
        p = row.get("passages") or {}
        pub.append({
            "key": key, "arm": arm, "id": row["id"], "type": it["type"], "behaviour": it["behaviour"],
            "lang": it["lang"], "topic": topics.get(it["doc1"], ""), "category": H.category_of(it["type"], arm),
            "status": row["status"], "attempts": row.get("attempts"), "correct": d["correct"], "why": d["why"],
            "judge": j, "false_refusal": H.false_refusal(it, row, j),
            "outcome": row.get("outcome"), "full_text": row.get("full_text"),
            "second_search": row.get("second_search"), "n_pages": row.get("n_pages"),
            "evidence": row.get("evidence"), "n_cited": len(row.get("cited") or []),
            "other_papers_cited": row.get("other_papers_cited") or [],
            "passages_status": p.get("status"),
            "passage_statuses": dict(Counter(x.get("status") for x in p.get("items") or [])),
            "numbers_missing": len(p.get("numbers_missing") or []),
            "critical": H.critical_candidates(it, arm, row, j),
            "flags": H.review_flags(it, arm, row, d["correct"]),
            "wall_s": row.get("wall_s"), "generation_s": row.get("generation_s"),
            "prompt_tokens": row.get("prompt_tokens"), "completion_tokens": row.get("completion_tokens"),
            "thinking_tokens": row.get("thinking_tokens"), "est_usd": row.get("est_usd"),
        })

    def sel(arm=None, **kw):
        return [r for r in pub if (arm is None or r["arm"] == arm) and all(r[k] == v for k, v in kw.items())]

    alpha = sel("alpha")
    rep = {"mode": mode, "when": time.strftime("%Y-%m-%d %H:%M"), "judge_models": sorted(model_versions),
           "config": answers["meta"].get("config"), "headline_alpha": rate(alpha)}
    rep["by_category"] = {c: rate([r for r in pub if r["category"] == c]) for c in H.CATEGORIES}
    rep["alpha_by_lang"] = {g: rate(sel("alpha", lang=g)) for g in sorted({r["lang"] for r in alpha})}
    rep["alpha_by_topic"] = {g: rate(sel("alpha", topic=g)) for g in sorted({r["topic"] for r in alpha})}
    rep["alpha_by_category_lang"] = {f"{c}/{g}": rate([r for r in alpha if r["category"] == c and r["lang"] == g])
                                     for c in H.CATEGORIES for g in sorted({r["lang"] for r in alpha})
                                     if any(r["category"] == c for r in alpha)}

    # (β) ζευγαρωμένη με την (α), στις ίδιες ερωτήσεις
    a2 = {r["id"]: r["correct"] for r in sel("alpha", type="two_papers")}
    b2 = {r["id"]: r["correct"] for r in sel("beta")}
    both = sorted(set(a2) & set(b2))
    rep["beta_vs_alpha"] = {
        "n": len(both),
        "improved": [i for i in both if b2[i] is True and a2[i] is not True],
        "worsened": [i for i in both if a2[i] is True and b2[i] is not True],
        "both_correct": sum(a2[i] is True and b2[i] is True for i in both),
        "neither": sum(a2[i] is not True and b2[i] is not True for i in both),
        "beta_full_text": sum(bool(r["full_text"]) for r in sel("beta")),
    }

    # χωρίς έγγραφα: τι ξέρει το μοντέλο από μνήμη (διαγνωστικό)
    closed = {r["id"]: r for r in sel("closed")}
    rep["closed_book"] = {t: rate([r for r in closed.values() if r["type"] == t])
                          for t in ("single", "two_papers", "table")}
    rep["closed_book_by_topic"] = {g: rate([r for r in closed.values() if r["topic"] == g])
                                   for g in sorted({r["topic"] for r in closed.values()})}
    ax = {r["id"]: r for r in alpha if r["id"] in closed}
    rep["closed_vs_alpha"] = {
        "n": len(ax),
        "both": sum(ax[i]["correct"] is True and closed[i]["correct"] is True for i in ax),
        "only_rag": sum(ax[i]["correct"] is True and closed[i]["correct"] is not True for i in ax),
        "only_closed": sum(ax[i]["correct"] is not True and closed[i]["correct"] is True for i in ax),
        "neither": sum(ax[i]["correct"] is not True and closed[i]["correct"] is not True for i in ax),
    }

    rag = [r for r in pub if r["arm"] in ("alpha", "beta") and r["status"] == "ok"]
    rep["false_refusals"] = {arm: {"count": sum(r["false_refusal"] for r in sel(arm)),
                                   "answerable": sum(r["type"] != "near" for r in sel(arm)),
                                   "ids": [r["id"] for r in sel(arm) if r["false_refusal"]]}
                             for arm in ("alpha", "beta")}
    rep["routes"] = {arm: dict(Counter(r["outcome"] for r in sel(arm))) for arm in ("alpha", "beta")}

    def lat(rows, key):
        xs = [r[key] for r in rows if r.get(key) is not None]
        return {"median": round(statistics.median(xs), 2) if xs else None,
                "p95": round(H.percentile(xs, 95), 2) if xs else None, "n": len(xs)}
    rep["latency_s"] = {arm: {"wall": lat(sel(arm, status="ok"), "wall_s"),
                              "generation": lat(sel(arm, status="ok"), "generation_s")}
                        for arm in ("alpha", "beta")}
    rep["cost_estimate_usd"] = {arm: {"total": round(sum(r["est_usd"] or 0 for r in sel(arm)), 3),
                                      "per_answer": round(statistics.mean([r["est_usd"] or 0 for r in sel(arm)]), 4)
                                      if sel(arm) else None,
                                      "prompt_tokens_median": lat(sel(arm, status="ok"), "prompt_tokens")["median"]}
                                for arm in ("alpha", "beta", "closed")}

    sts = Counter()
    for r in rag:
        sts.update(r["passage_statuses"])
    sentences = sum(v for k, v in sts.items() if k != "absence")
    cited = sentences - sts.get("no_citation", 0)
    rep["citations"] = {"sentences": sentences, "with_citation": cited,
                        "passage_found_on_cited_page": sts.get("found", 0),
                        "found_on_other_cited_page": sts.get("found_elsewhere", 0),
                        "not_found": sts.get("not_found", 0),
                        "pct_with_citation": round(cited / sentences, 3) if sentences else None,
                        "pct_found_of_cited": round(sts.get("found", 0) / cited, 3) if cited else None,
                        "answers_with_number_missing_from_cited_page": sum(r["numbers_missing"] > 0 for r in rag)}

    ev_rows = [(r, e) for r in rag for e in (r["evidence"] or []) if e]
    rep["evidence_pages"] = {arm: {"pages": sum(1 for r, _e in ev_rows if r["arm"] == arm),
                                   "reached_model": sum(e["in_final"] for r, e in ev_rows if r["arm"] == arm),
                                   "cited": sum(e["cited"] for r, e in ev_rows if r["arm"] == arm)}
                             for arm in ("alpha", "beta")}
    stage = Counter()
    for r, e in ev_rows:
        if r["arm"] != "alpha" or r["correct"] is not False:
            continue
        if e["in_final"]:
            stage["reached_model"] += 1
        elif e["in_candidates_any"]:
            stage["dropped_after_reranker"] += 1
        else:
            stage["not_in_candidates"] += 1
    rep["failure_stage_alpha_evidence_pages"] = dict(stage)

    rep["critical_candidates"] = {c: [r["key"] for r in pub if c in r["critical"]] for c in ("c1", "c2", "c3", "c4")}
    rep["review_flags"] = {f: [r["key"] for r in pub if f in r["flags"]]
                           for f in ("other_paper_cited", "non_plain_value", "passage_check_missing")}
    rep["technical_failures"] = [r["key"] for r in pub if r["status"] != "ok"]
    return rep, pub


def print_report(rep: dict) -> None:
    print(f"\n=== {rep['mode']}: αποτελέσματα κριτή ({', '.join(rep['judge_models']) or '—'}) ===")
    print(f"Κύριος αριθμός, (α) χωρίς επιλογή: {fmt(rep['headline_alpha'])}")
    for c in H.CATEGORIES:
        print(f"  {c:<14} {fmt(rep['by_category'][c])}")
    print("  ανά γλώσσα (α): " + " · ".join(f"{g} {fmt(v)}" for g, v in rep["alpha_by_lang"].items()))
    print("  ανά θέμα (α):   " + " · ".join(f"{g} {fmt(v)}" for g, v in rep["alpha_by_topic"].items()))
    b = rep["beta_vs_alpha"]
    print(f"(β) έναντι (α), {b['n']} ίδιες ερωτήσεις: βελτιώθηκαν {len(b['improved'])} {b['improved']}, "
          f"χειροτέρεψαν {len(b['worsened'])} {b['worsened']}, σωστές και στις δύο {b['both_correct']}, "
          f"σε καμία {b['neither']} · όλο το κείμενο {b['beta_full_text']}/{b['n']}")
    print("Χωρίς έγγραφα: " + " · ".join(f"{t} {fmt(v)}" for t, v in rep["closed_book"].items()))
    cv = rep["closed_vs_alpha"]
    print(f"  ίδιες ερωτήσεις με (α): και τα δύο {cv['both']} · μόνο RAG {cv['only_rag']} · "
          f"μόνο χωρίς έγγραφα {cv['only_closed']} · κανένα {cv['neither']} (από {cv['n']})")
    for arm, v in rep["false_refusals"].items():
        print(f"Αδικαιολόγητες αρνήσεις ({arm}): {v['count']}/{v['answerable']} {v['ids']}")
    for arm, v in rep["latency_s"].items():
        print(f"Χρόνος ({arm}): διάμεσος {v['wall']['median']} s · p95 {v['wall']['p95']} s · "
              f"κόστος/απάντηση ~{rep['cost_estimate_usd'][arm]['per_answer']} $ (εκτίμηση)")
    c = rep["citations"]
    print(f"Παραπομπές: {c['with_citation']}/{c['sentences']} προτάσεις με [S#] · χωρίο στη σελίδα που "
          f"παρατίθεται {c['passage_found_on_cited_page']}/{c['with_citation']} · δεν βρέθηκε {c['not_found']}")
    for arm, v in rep["evidence_pages"].items():
        print(f"Σελίδες-τεκμήρια ({arm}): έφτασαν στο μοντέλο {v['reached_model']}/{v['pages']} · "
              f"παρατέθηκαν {v['cited']}/{v['pages']}")
    print(f"Πού χάθηκε το τεκμήριο στις αποτυχίες (α): {rep['failure_stage_alpha_evidence_pages']}")
    print(f"Διαδρομές: {rep['routes']}")
    print("Υποψήφια κρίσιμα λάθη: " + " · ".join(f"{k} {len(v)}" for k, v in rep["critical_candidates"].items()))
    print("Σημάδια για έλεγχο: " + " · ".join(f"{k} {len(v)}" for k, v in rep["review_flags"].items()))
    if rep["technical_failures"]:
        print(f"Τεχνικές αποτυχίες: {rep['technical_failures']}")


# ------------------------------------------------------------------------------------- φύλλο ελέγχου

HUMAN_HELP = """# Ανθρώπινος έλεγχος — φάση 1 (χωρίς απάντηση αναφοράς)

Για κάθε R: διάβασε την ερώτηση και την απάντηση. Κάτω από κάθε απάντηση υπάρχει, για κάθε πρόταση, το χωρίο
από το οποίο βγήκε, αντιγραμμένο αυτούσιο από τη σελίδα (το ίδιο που δείχνει το κουμπί «Source passages»).
Αν το χωρίο σού αρκεί, δεν χρειάζεται το PDF. Άνοιξε το PDF (`backend/evaluation/test_papers/heldout/`) όταν
κάτι δεν στέκει, όταν λείπει χωρίο, ή όταν η απάντηση λέει ότι κάτι δεν υπάρχει στο άρθρο.

Προχώρα ΜΕ ΤΗ ΣΕΙΡΑ R01 → R20. Η σειρά είναι τυχαία, και όσες προλάβεις μετράνε με τον πραγματικό
παρονομαστή. ΜΗΝ ανοίξεις το phase2 πριν συμπληρώσεις το `phase1_form.csv` για όσες έκανες.

Στήλες του phase1_form.csv:
- `verdict`: για ερωτήσεις με απάντηση: correct | partial | refused | wrong.
  Για δύο άρθρα: `verdict` = το σκέλος του 1ου άρθρου, `verdict_paper2` = του 2ου, με τιμές
  correct | partial | declared_missing | omitted | wrong.
  Για «το άρθρο δεν το λέει» (σημειώνεται στην κεφαλίδα): correct (λέει καθαρά ότι λείπει) |
  soft_leak (δίνει σχετικά χωρίς να πει ότι λείπει) | leak (δίνει το στοιχείο που λείπει σαν αληθινό).
- `unsupported_claim`: yes/no — υπάρχει ισχυρισμός που ΔΕΝ στηρίζεται στις πηγές;
- `wrong_value`: yes/no/- — (πίνακες) δηλώνεται λάθος αριθμός, μονάδα, πρόσημο, γραμμή ή στήλη;
- `wrong_article_citation`: yes/no — παραπομπή σε άρθρο που δεν σχετίζεται με τον ισχυρισμό;
- `citation_supports`: yes/partly/no/- — στηρίζουν οι παραπομπές αυτό που λέει η απάντηση (δεκτό και
  ισοδύναμο χωρίο από άλλη σελίδα);
- `valid_from_other_article`: yes/no/- — η απάντηση στηρίζεται σωστά σε ΑΛΛΟ άρθρο από αυτό της
  ερώτησης, και η ερώτηση δεν περιορίζεται σε συγκεκριμένο άρθρο;
- `notes`: ελεύθερο κείμενο.
"""

PHASE2_HELP = """# Ανθρώπινος έλεγχος — φάση 2 (αποκάλυψη της απάντησης αναφοράς)

Άνοιξέ το ΜΟΝΟ αφού συμπληρώσεις το phase1_form.csv. Δεν αλλάζεις τη φάση 1. Στο phase2_form.csv:
- `set_error`: none | ambiguous_other_article (σωστή απάντηση από άλλο άρθρο, η ερώτηση δεν περιορίζεται)
  | wrong_reference (λάθος απάντηση αναφοράς/τιμή/«τι λείπει») | other
- `notes`: ελεύθερο κείμενο.
"""


_PASSAGE_STATUS = {
    "found_elsewhere": "χωρίο από άλλη σελίδα της απάντησης",
    "not_found": "δεν βρέθηκε αυτούσιο χωρίο",
    "no_citation": "χωρίς παραπομπή",
    "absence": "λέει τι λείπει (δεν υπάρχει χωρίο)",
}


def passages_md(a: dict) -> str:
    """Το χωρίο κάθε πρότασης, όπως το δείχνει το κουμπί «Source passages» (βήμα 6) στην εφαρμογή."""
    p = a.get("passages") or {}
    if p.get("status") not in ("done", "no_pages") or not p.get("items"):
        return "  - (ο έλεγχος χωρίων δεν ολοκληρώθηκε: έλεγξε στο PDF)"
    out = []
    for x in p["items"]:
        sent = " ".join((x.get("sentence") or "").split())
        if len(sent) > 220:
            sent = sent[:217] + "…"
        st = x.get("status")
        if st in ("found", "found_elsewhere") and x.get("quote"):
            extra = f" ({_PASSAGE_STATUS[st]})" if st == "found_elsewhere" else ""
            out.append(f"  - *{sent}*\n    → [{x.get('label')}] {x.get('file')}, σελ. {x.get('page')}{extra}: "
                       f"«{' '.join(x['quote'].split())}»")
        else:
            out.append(f"  - *{sent}*\n    → {_PASSAGE_STATUS.get(st, st or '—')}")
    return "\n".join(out)


def build_review(mode: str, items, answers, pub, titles) -> dict:
    rows = {r["key"]: r for r in pub}
    flagged = sorted(k for k, r in rows.items()
                     if r["arm"] in ("alpha", "beta") and r["status"] == "ok" and (r["critical"] or r["flags"]))
    pool = defaultdict(list)
    for k, r in rows.items():
        if r["arm"] in ("alpha", "beta") and r["status"] == "ok" and k not in flagged:
            pool[r["category"]].append(k)
    rnd = H.sample_random(pool)
    keys = flagged + rnd
    random.Random(H.REVIEW_SEED + 1).shuffle(keys)
    pdir = H.private_dir(mode)
    key_map, p1, p2 = {}, [HUMAN_HELP], [PHASE2_HELP]
    f1 = [["R", "verdict", "verdict_paper2", "unsupported_claim", "wrong_value", "wrong_article_citation",
           "citation_supports", "valid_from_other_article", "notes"]]
    f2 = [["R", "set_error", "notes"]]
    for n, key in enumerate(keys, 1):
        rid = f"R{n:02d}"
        r, a = rows[key], answers["rows"][key]
        it = items[r["id"]]
        key_map[rid] = {"key": key, "flagged": key in flagged, "critical": r["critical"], "flags": r["flags"],
                        "category": r["category"], "judge": r["judge"], "judge_correct": r["correct"]}
        kind = {"single": "ενός άρθρου", "two_papers": "δύο άρθρων", "table": "πίνακα",
                "near": "«το άρθρο δεν το λέει (όλο ή μέρος)»"}[it["type"]]
        arts = ", ".join(f"{d} ({titles.get(d, '')})" for d in (it["doc1"], it["doc2"]) if d)
        srcs = {s["label"]: s for s in a.get("sources") or []}
        cited = "\n".join(f"  - {lab} = {srcs[lab]['file']}, σελίδα {srcs[lab]['page']}"
                          for lab in a.get("cited") or [] if lab in srcs) or "  - (καμία)"
        p1.append(f"\n---\n## {rid} · ερώτηση {kind}\n\n**Άρθρο(α) της ερώτησης:** {arts}\n\n"
                  f"**Ερώτηση:** {it['question']}\n\n**Απάντηση του συστήματος:**\n\n{a['answer']}\n\n"
                  f"**Παραπομπές:**\n{cited}\n\n**Χωρία ανά πρόταση:**\n{passages_md(a)}\n")
        vals = "; ".join(f"{v['raw']} · {v['unit']} · {v['label']}" for v in it["values"])
        p2.append(f"\n---\n## {rid}\n\n**Σωστή συμπεριφορά:** {it['behaviour']}\n\n"
                  f"**Απάντηση αναφοράς:** {it['reference']}\n\n"
                  + (f"**Τεκμήριο 1:** {it['doc1']} σ. {it['page1']}: «{it['quote1']}»\n\n" if it["page1"] else "")
                  + (f"**Τεκμήριο 2:** {it['doc2']} σ. {it['page2']}: «{it['quote2']}»\n\n" if it["page2"] else "")
                  + (f"**Τι λείπει:** {it['missing']}\n\n" if it["missing"] else "")
                  + (f"**Τιμές πίνακα:** {vals}\n" if vals else ""))
        f1.append([rid] + [""] * 8)
        f2.append([rid, "", ""])
    with open(os.path.join(pdir, "review_phase1.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(p1))
    with open(os.path.join(pdir, "review_phase2_REVEAL.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(p2))
    for name, rows_ in (("phase1_form.csv", f1), ("phase2_form.csv", f2)):
        path = os.path.join(pdir, name)
        if os.path.exists(path):
            print(f"  (το {name} υπάρχει ήδη: δεν το αντικαθιστώ)")
            continue
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerows(rows_)
    H.write_json_atomic(os.path.join(pdir, "review_key.json"), key_map)
    return {"flagged": len(flagged), "random": len(rnd), "total": len(keys)}


# ----------------------------------------------------------------------------------- συμφωνία ανθρώπου

def _read_form(path: str) -> dict:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["R"].strip(): {k: (v or "").strip().lower() for k, v in r.items()} for r in csv.DictReader(f)}


def human_judgments(it: dict, h: dict) -> dict:
    """Οι ανθρώπινες κρίσεις στη μορφή του decide()."""
    v = h.get("verdict", "")
    if it["type"] == "two_papers":
        return {"mh": {"paper_1": v, "paper_2": h.get("verdict_paper2", "")}}
    if it["type"] == "near" and it["behaviour"] == "refuse":
        return {"near": v}
    if it["type"] == "near":
        # μερική: «correct» του ανθρώπου = απάντησε ό,τι υπάρχει ΚΑΙ είπε τι λείπει
        return {"near": "correct" if v == "correct" else v, "main": v}
    if it["type"] == "table":
        return {"main": v, "table": {"plain_ok": h.get("wrong_value") != "yes"}}
    return {"main": v}


def agreement(mode: str) -> int:
    pdir = H.private_dir(mode)
    items, answers, _titles, _topics = load(mode)
    with open(os.path.join(pdir, "review_key.json"), encoding="utf-8") as f:
        key_map = json.load(f)
    f1 = _read_form(os.path.join(pdir, "phase1_form.csv"))
    f2 = _read_form(os.path.join(pdir, "phase2_form.csv"))
    out = {"flagged": Counter(), "random": Counter()}
    confirmed = {c: [0, 0] for c in ("c1", "c2", "c3", "c4")}   # [επιβεβαιώθηκαν, υποψήφια]
    field = {"c1": "unsupported_claim", "c2": "wrong_value", "c3": None, "c4": "wrong_article_citation"}
    set_errors, missing = [], []
    for rid, km in sorted(key_map.items()):
        h = f1.get(rid)
        if not h or not h.get("verdict"):
            missing.append(rid)
            continue
        row = answers["rows"][km["key"]]
        it = items[row["id"]]
        hd = H.decide(it, row["arm"], row, human_judgments(it, h))["correct"]
        grp = "flagged" if km["flagged"] else "random"
        out[grp]["reviewed"] += 1
        out[grp]["agree" if hd == km["judge_correct"] else "revised"] += 1
        if hd != km["judge_correct"]:
            out[grp]["revised_to_correct" if hd else "revised_to_not_correct"] += 1
        for c in km["critical"]:
            confirmed[c][1] += 1
            ok = h.get("verdict") == "leak" if c == "c3" else h.get(field[c]) == "yes"
            confirmed[c][0] += bool(ok)
        se = (f2.get(rid) or {}).get("set_error", "")
        if se and se != "none":
            set_errors.append({"R": rid, "key": km["key"], "kind": se})
    if missing:
        print(f"Λείπουν κρίσεις στο phase1_form.csv για: {', '.join(missing)}")
    res_path = os.path.join(H.MODES[mode]["out"], "results.json")
    with open(res_path, encoding="utf-8") as f:
        res = json.load(f)
    alpha_rows = [r for r in res["rows"] if r["arm"] == "alpha"]
    excl = {e["key"] for e in set_errors}
    kept = [r for r in alpha_rows if r["key"] not in excl]
    res["human"] = {"reviewed": {g: dict(v) for g, v in out.items()},
                    "critical_confirmed": {c: {"confirmed": v[0], "candidates": v[1]} for c, v in confirmed.items()},
                    "set_errors": set_errors,
                    "headline_alpha_excluding_set_errors": rate(kept),
                    "note": "Τα λάθη του σετ αναζητήθηκαν κυρίως στις σημαδεμένες ερωτήσεις (συχνότερα αποτυχίες): "
                            "η εξαίρεσή τους τείνει να ανεβάζει το ποσοστό. Δημοσιεύονται και οι δύο αριθμοί."}
    H.write_json_atomic(res_path, res)
    for g, v in out.items():
        print(f"{g}: ελέγχθηκαν {v['reviewed']} · ίδια κρίση {v['agree']} · άλλαξαν {v['revised']} "
              f"(→σωστό {v['revised_to_correct']}, →όχι σωστό {v['revised_to_not_correct']})")
    print("Κρίσιμα λάθη που επιβεβαιώθηκαν: "
          + " · ".join(f"{c} {v[0]}/{v[1]}" for c, v in confirmed.items()))
    print(f"Λάθη του σετ: {len(set_errors)} {[e['key'] for e in set_errors]}")
    print(f"(α) χωρίς τα λάθη του σετ: {fmt(res['human']['headline_alpha_excluding_set_errors'])} "
          f"· με όλα: {fmt(res['headline_alpha'])}")
    return 0


# -------------------------------------------------------------------------------------------- κύριο

def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")       # Git Bash στα Windows: αλλιώς σπάει στα ελληνικά/«→»
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--max-usd", type=float, default=3.0, help="όριο κόστους του κριτή")
    ap.add_argument("--agreement", action="store_true", help="συμφωνία ανθρώπου–κριτή, μετά τα forms")
    ap.add_argument("--review-only", action="store_true",
                    help="ξαναφτιάχνει ΜΟΝΟ το φύλλο ελέγχου από τα σωσμένα αποτελέσματα· κανένας κριτής, το "
                         "results.json δεν αλλάζει, τα συμπληρωμένα forms δεν αντικαθίστανται")
    args = ap.parse_args()
    mode = "smoke" if args.smoke else "heldout"
    if args.agreement:
        return agreement(mode)
    if args.review_only:
        items, answers, titles, _topics = load(mode)
        with open(os.path.join(H.MODES[mode]["out"], "results.json"), encoding="utf-8") as f:
            pub = json.load(f)["rows"]
        rv = build_review(mode, items, answers, pub, titles)
        print(f"Φύλλο ελέγχου ξαναφτιάχτηκε: {rv['total']} ερωτήσεις ({rv['flagged']} σημαδεμένες, "
              f"{rv['random']} τυχαίες) · 0 κλήσεις · το results.json δεν άλλαξε")
        return 0

    items, answers, titles, topics = load(mode)
    key = C2.api_key()
    if not key:
        print("Λείπει το OPENAI_API_KEY (περιβάλλον ή .env).")
        return 1
    pdir = H.private_dir(mode)
    C2.CMP = pdir                                         # ιδιωτική cache, όχι η κοινή του git
    C2.CACHE_PATH = os.path.join(pdir, "judge_cache.json")
    versions = set()
    orig_http = C2.http

    def http_spy(url, k, body=None):
        d = orig_http(url, k, body)
        if isinstance(d, dict) and d.get("model"):
            versions.add(d["model"])
        return d

    C2.http = http_spy
    judge = C2.Judge(args.model, key, args.max_usd)
    judged_path = os.path.join(pdir, "judgments.json")
    judged = {}
    todo = [(k, r) for k, r in answers["rows"].items() if r["status"] == "ok"]
    print(f"Κριτής {args.model} · {len(todo)} απαντήσεις · όριο {args.max_usd} $")
    try:
        for n, (k, row) in enumerate(todo, 1):
            judged[k] = judge_row(judge, items[row["id"]], row["arm"], row, titles)
            if n % 10 == 0:
                judge.save()
                print(f"  {n}/{len(todo)} · νέες κλήσεις {judge.new} · ~{judge.usd:.2f} $", flush=True)
    finally:
        judge.save()
        H.write_json_atomic(judged_path, judged)
    if not versions and judge.hits:
        versions.add(f"{args.model} (από cache)")
    print(f"Κρίσεις: νέες {judge.new}, από cache {judge.hits}, ~{judge.usd:.2f} $")
    invalid = [k for k, v in judged.items() for lab in [v["j"].get("main"), v["j"].get("near"),
               *((v["j"].get("mh") or {}).values())] if isinstance(lab, str) and lab.startswith("invalid")]
    if invalid:
        print(f"ΠΡΟΣΟΧΗ: άκυρες ετικέτες κριτή σε {sorted(set(invalid))}")

    rep, pub = build_report(items, answers, judged, topics, mode, versions)
    rep["rows"] = pub
    H.write_json_atomic(os.path.join(H.MODES[mode]["out"], "results.json"), rep)
    print_report(rep)
    rv = build_review(mode, items, answers, pub, titles)
    print(f"\nΦύλλο ελέγχου (ιδιωτικό): {rv['total']} ερωτήσεις ({rv['flagged']} σημαδεμένες, "
          f"{rv['random']} τυχαίες, ανακατεμένες) στο {os.path.relpath(pdir, H.BACKEND)}")
    print("Αποτελέσματα χωρίς κείμενα: " + os.path.relpath(os.path.join(H.MODES[mode]["out"], "results.json"),
                                                            H.BACKEND))
    return 0


if __name__ == "__main__":
    sys.exit(main())

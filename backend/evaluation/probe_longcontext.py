"""Φάση 3.1: RAG έναντι «ΟΛΑ τα PDF στο prompt» — ποιότητα, ασφάλεια, κόστος, χρόνος.

ΤΟ ΕΡΩΤΗΜΑ: τα 7 papers είναι ~140k tokens και χωράνε ΟΛΟΚΛΗΡΑ στο Gemini (1M). Γιατί RAG;

ΤΙ ΚΑΝΕΙ: ο ΠΡΑΓΜΑΤΙΚΟΣ ask_ai (ίδιες οδηγίες, ίδιοι κανόνες «μόνο από το κείμενο / πες ότι λείπει»,
ίδιες παραπομπές [S#], ίδιο μοντέλο και ρυθμίσεις) με precomputed = ΟΛΕΣ οι 122 σελίδες του σώματος
(ίδιο κείμενο με τις σελίδες της ανάκτησης, αρχείο/σελίδα με τη σειρά) αντί για τις 8-12 της ανάκτησης.
Αλλάζει ΜΟΝΟ το τι διαβάζει το μοντέλο. Η πλευρά RAG = οι ΗΔΗ μετρημένες απαντήσεις του σημερινού
συστήματος (answers_baseline_main · answers_perdoc_mh_new · answers_baseline_near_ooc) — 0 νέες κλήσεις.
Η ερώτηση είναι στο ΤΕΛΟΣ του prompt -> η αρχή (οδηγίες + 140k σώμα) είναι ΙΔΙΑ σε κάθε κλήση και το
αυτόματο cache του Gemini μπορεί να χρεώσει μόνο ~25% της εισόδου· το usageMetadata λέει αν έγινε.

ΜΕΤΡΙΚΕΣ — ΤΙ ΚΡΙΝΕΤΑΙ ΚΑΙ ΜΕ ΤΙ (ίδια μέτρηση ΚΑΙ στις δύο πλευρές):
    κύριο (45 + 5 ooc)      λέξεις-κλειδιά ΜΕΣΑ στην απάντηση · αρνήσεις σε ερωτήσεις με απάντηση · ooc 5/5;
    multi_hop νέες (29)     μισά με λέξη του paper τους μέσα στην απάντηση · «και τα δύο»
    κοντινές ooc (42)       ο κριτής της Φάσης 0.1 ΑΥΤΟΥΣΙΟΣ (δεν βλέπει context -> φθηνός): σωστό /
                            ήπια διαρροή (β) / διαρροή (γ)· στο RAG η σιωπή του φύλακα μετράει ως σωστό
    Οι λέξεις-κλειδιά είναι ΠΡΟΣΕΓΓΙΣΗ του κριτή (οι κριτές κύριου/multi_hop διαβάζουν ΟΛΟ το context =
    +140k ανά κρίση, ακριβό). Ισχύει ίδια και για τις δύο πλευρές.
    κόστος                  tokens εισόδου (+ πόσα από cache) / εξόδου ανά ερώτηση -> $, με τις τιμές του
                            measure_cost (0.30 / 2.50 ανά 1M)· cache = 25% της τιμής εισόδου
    χρόνος                  χρόνος μέχρι την 1η λέξη (TTFT) και συνολικός, ΜΟΝΟ σε φρέσκες κλήσεις· για το
                            RAG ένα μικρό δείγμα ξαναγεννιέται φρέσκο (χωρίς πάγωμα) μόνο για χρονομέτρηση
ΦΥΛΑΚΑΣ ΚΟΣΤΟΥΣ: --max-usd (default 3.0) — σταματάει πριν τον ξεπεράσει· ό,τι πληρώθηκε σώζεται.

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (30/9/2026):
    κύριο: λέξεις στην απάντηση ίδιες ± 5% · αρνήσεις σε ερωτήσεις με απάντηση: όσες και στο RAG ή λιγότερες
           (το q025 που κόβει ο φύλακας εδώ απαντιέται) · ooc 5/5
    multi_hop νέες: μισά με λέξη όλα στο prompt ΠΑΝΩ από το RAG κατά +3..+10 (βλέπει ΚΑΙ τα δύο papers
           ολόκληρα — αυτό που η Φάση 2 έφτιαξε μόνο εν μέρει)
    κοντινές ooc: σωστό 32-39/42 (RAG 40/42) — περισσότερο «σχετικό» υλικό = περισσότερες ήπιες διαρροές
    κόστος ανά ερώτηση: RAG ~0.005-0.007 $ · όλα στο prompt 0.012-0.018 $ με cache (~0.045 χωρίς) -> 2-8×
    TTFT: RAG 1-3 s · όλα στο prompt 4-12 s
ΤΙ ΘΑ ΓΡΑΦΤΕΙ (κανόνας ερμηνείας, γραμμένος πριν):
    όλα στο prompt ≥ RAG + 5 μισά ΚΑΙ κοντινές όχι χειρότερες -> «σε 7 papers το πλήρες context κερδίζει σε
        ποιότητα· η αξία του RAG είναι κόστος / χρόνος / κλίμακα» (και θα φανεί στα νούμερα)
    ίση ή χειρότερη ποιότητα -> «σε αυτό το μέγεθος το RAG κερδίζει παντού»
    κοντινές χειρότερες κατά ≥ 3 -> το εύρημα ασφαλείας μπαίνει ΠΡΩΤΟ στην περιγραφή

ΑΠΟΤΕΛΕΣΜΑ (30/9/2026, runs/longcontext.csv · 121 ερωτήσεις · 1.42 $ + 0.13 $ η δοκιμή · m079 εκτός: η
απάντηση RAG είχε μείνει σε σφάλμα 402):
                                                        RAG (σήμερα)     ΟΛΑ ΣΤΟ PROMPT
    κύριο (45): λέξεις-κλειδιά στην απάντηση /133            78                81
    κύριο: «αρνήσεις» (μοτίβο) σε ερωτήσεις με απάντηση       10                10   <- το μοτίβο πιάνει και
                                                                                     επιφυλάξεις, όχι μόνο αρνήσεις
    κύριο ooc: αρνήσεις                                      5/5               5/5
    multi_hop νέες (28): μισά με λέξη του paper /56          20                22   (καλύτερα 3, χειρότερα 3)
      «και τα δύο»                                            4                 7
    κοντινές ooc (42): σωστό · (β) · (γ)                 40 · 1 · 1        41 · 1 · 0   (μόνη αλλαγή: n038,
                                                                                     γνωστή κλήρωση της 28/9)
    tokens εισόδου ανά ερώτηση (διάμεσος)                 10.318           132.293
    από το cache του Gemini                                    -             98% (114/115 κλήσεις)
    κόστος ανά ερώτηση (διάμεσος)                          0.0048 $         0.0120 $  -> 2.5× (χωρίς cache ~8.6×)
    TTFT (διάμεσος)                                      3.24 s + ανάκτηση   3.85 s   (p90 10.7 s)
                                                         (~0.8 s στο Ryzen)
    ΣΥΜΠΕΡΑΣΜΑ (κατά τον κανόνα): +2 μισά < +5 -> ΙΣΗ ΠΟΙΟΤΗΤΑ· κοντινές όχι χειρότερες. «Σε 7 papers (~132k
        tokens) το πλήρες context ΔΕΝ απαντά καλύτερα από το RAG μετά τη Φάση 2 — και κοστίζει 2.5×.» Η διαφορά
        είναι ΜΙΚΡΟΤΕΡΗ από όσο νόμιζα: με το cache το πλήρες context είναι σχεδόν εξίσου γρήγορο στον διάμεσο
        (χειρότερη ουρά) και μόλις 2.5× ακριβότερο. Τα ΠΡΑΓΜΑΤΙΚΑ πλεονεκτήματα του RAG εδώ δεν είναι η
        ποιότητα: είναι η ΚΛΙΜΑΚΑ (πάνω από ~1M tokens το πλήρες context δεν χωράει καν), το ΚΟΣΤΟΣ χωρίς
        εγγυημένο cache (8.6×), η ΑΠΟΜΟΝΩΣΗ ΧΡΗΣΤΩΝ (κάθε χρήστης βλέπει μόνο τα δικά του έγγραφα — το πλήρες
        context θα ήθελε άλλο prompt ανά χρήστη, άρα και καθόλου κοινό cache) και ο ΦΥΛΑΚΑΣ (σιωπή πριν τη γέννηση).
    ⚠️ ΟΡΙΟ ΜΕΤΡΗΣΗΣ: οι λέξεις-κλειδιά είναι ΧΟΝΔΡΗ προσέγγιση — στο ίδιο mh_new ο κριτής δίνει στο RAG 44/58
        σωστά μισά, οι λέξεις 20/56 (ακριβείς αριθμοί, ελληνικές απαντήσεις). Η σύγκριση κριτή προς κριτή στα
        multi_hop (~1.2 $, ο κριτής διαβάζει όλο το σώμα) ΔΕΝ έγινε.
    ΠΡΟΒΛΕΨΗ: κύριο ±5% ✓ · αρνήσεις ίδιες ✓ (η q025 που κόβει ο φύλακας απαντιέται) · ooc 5/5 ✓ · multi_hop +3..+10 ✗
        (+2) · κοντινές 32-39 ✗ (41 — ΚΑΜΙΑ επιπλέον διαρροή) · κόστος RAG 0.005-0.007 ✗ (0.0048) · πλήρες 0.012-0.018 ✓
        · 2-8× ✓ · TTFT RAG 1-3 ✗ (3.24) · πλήρες 4-12 ✗ (3.85). Έξι στις δέκα — ΥΠΕΡΕΚΤΙΜΗΣΑ το πλεονέκτημα του
        πλήρους context ΚΑΙ το κόστος του σε χρόνο.

    docker compose run --rm --no-deps -v eval_near_ooc_store:/tmp/eval_near_ooc_chroma backend \
        python -u evaluation/probe_longcontext.py --limit 2        # δοκιμή: 2 ανά σετ -> κόστος, cache
                                                                   # (σώζει runs/longcontext_smoke.csv)
    docker compose run --rm --no-deps -v eval_near_ooc_store:/tmp/eval_near_ooc_chroma backend \
        python -u evaluation/probe_longcontext.py                  # όλο: 121 ερωτήσεις
"""
import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
import probe_decomp_answers as PA
import scoreboard as S
import scoreboard_answers as SA

import gemini_rest

SETS = {"main": "golden_set_50.jsonl", "mh_new": "golden_multihop_v2.jsonl",
        "near_ooc": "golden_near_ooc.jsonl"}
RAG_ANS = {"main": "answers_baseline_main.json", "mh_new": "answers_perdoc_mh_new.json",
           "near_ooc": "answers_baseline_near_ooc.json"}
RAG_PAGES = {"main": "baseline.json", "mh_new": "perdoc.json", "near_ooc": "baseline.json"}
OUT_CSV = os.path.join(S.RUNS, "longcontext.csv")
IN_USD, OUT_USD, CACHED_SHARE = 0.30, 2.50, 0.25        # measure_cost.py · cache = 25% της εισόδου
TIMING_PER_SET = 2                                       # φρέσκες γεννήσεις RAG, μόνο για χρονομέτρηση


def usd(usage: dict) -> float:
    p = usage.get("promptTokenCount") or 0
    c = usage.get("cachedContentTokenCount") or 0
    o = (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)
    return ((p - c) * IN_USD + c * IN_USD * CACHED_SHARE + o * OUT_USD) / 1e6


def rag_usd(row: dict) -> float:
    return ((row.get("prompt_tokens") or 0) * IN_USD
            + ((row.get("out_tokens") or 0) + (row.get("thinking_tokens") or 0)) * OUT_USD) / 1e6


def kw_hits(answer: str, kws: list) -> int:
    a = answer.lower()
    return sum(k.lower() in a for k in kws)


def halves(t: dict, answer: str) -> int:
    a = answer.lower()
    return sum(any(k.lower() in a for k in kws) for kws in t["keywords_by_doc"].values())


async def answer_timed(ai_core, question: str, pages: list):
    """(κείμενο, TTFT, συνολικός χρόνος) από τον ΠΡΑΓΜΑΤΙΚΟ ask_ai με έτοιμες σελίδες."""
    t0 = time.perf_counter()
    ttft, out = None, ""
    async for chunk in ai_core.ask_ai(question, target_filenames=None, user_id=E.TEST_USER,
                                      precomputed=pages):
        if chunk.get("type") == "text":
            if ttft is None:
                ttft = time.perf_counter() - t0
            out += chunk.get("data", "")
    return out, ttft, time.perf_counter() - t0


async def run(args) -> int:
    real_once = gemini_rest.generate_once
    real_stream = gemini_rest.stream_generate
    tests = {sk: E.load_jsonl(os.path.join(E.HERE, fn)) for sk, fn in SETS.items()}
    if args.limit:
        tests = {sk: ts[:args.limit] for sk, ts in tests.items()}
    rag = {}
    for sk, fn in RAG_ANS.items():
        with open(os.path.join(S.OUT_DIR, fn), encoding="utf-8") as f:
            rag[sk] = {r["id"]: r for r in json.load(f)["rows"]}
    rag_pages = {}
    for sk, fn in RAG_PAGES.items():
        with open(os.path.join(S.OUT_DIR, fn), encoding="utf-8") as f:
            rag_pages[sk] = {r["id"]: r["pages"] for r in json.load(f)["rows"] if r["set"] == sk}

    ai_core, _near = E.open_store()
    if ai_core.collection.count() != S.EXPECTED_CHUNKS:
        print(f"!! Το store δεν έχει {S.EXPECTED_CHUNKS} chunks — σταματάω")
        return 1
    ai_core._translation_cache.clear()
    frozen = S.Frozen(S.FROZEN_PATH, fresh=False)    # το ask_ai δεν ψάχνει εδώ· δίχτυ ασφαλείας
    frozen.install()
    cache = SA.AnswerCache(SA.ANS_PATH)
    cache.install_stream(real_stream)
    frozen_stream = gemini_rest.stream_generate
    n0 = len(cache.data)

    metas = {}
    for m in ai_core._get_bm25_index()["metas"]:
        metas.setdefault((m["file_name"], int(m["page"])), m)
    corpus = [PA.page_of(ai_core, metas[k]) for k in sorted(metas)]
    meta_of = {f"{f}:{p}": m for (f, p), m in metas.items()}
    print(f"\nΣώμα: {len(corpus)} σελίδες, {sum(len(t) for t, _m in corpus):,} χαρακτήρες · "
          f"{sum(len(ts) for ts in tests.values())} ερωτήσεις · όριο κόστους {args.max_usd} $\n", flush=True)

    rows, spent = [], 0.0
    for sk, ts in tests.items():
        for t in ts:
            if spent >= args.max_usd:
                print(f"!! ΟΡΙΟ ΚΟΣΤΟΥΣ {args.max_usd} $ — σταματάω (ό,τι πληρώθηκε είναι σωσμένο)")
                break
            before = len(cache.data)
            n_err = len(cache.errors)
            answer, ttft, total = await answer_timed(ai_core, t["question"], corpus)
            if len(cache.errors) > n_err:
                print(f"!! {t['id']}: {cache.errors[-1]} — σταματάω")
                cache.save()
                return 1
            fresh = len(cache.data) > before
            usage = (cache.data.get(cache.last_key) or {}).get("usage") or {}
            cost = usd(usage)
            spent += cost if fresh else 0.0
            r_rag = rag[sk].get(t["id"], {})
            row = {"set": sk, "id": t["id"], "lang": S.lang_of(t, t["question"]),
                   "category": t.get("category", ""), "fresh": int(fresh),
                   "ttft": round(ttft, 2) if (fresh and ttft) else None,
                   "total_s": round(total, 2) if fresh else None,
                   "in_tokens": usage.get("promptTokenCount"), "cached_tokens": usage.get("cachedContentTokenCount") or 0,
                   "out_tokens": (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0),
                   "usd": round(cost, 5), "rag_usd": round(rag_usd(r_rag), 5) if r_rag else None,
                   "rag_in_tokens": r_rag.get("prompt_tokens") if r_rag else None,
                   "rag_error": int(bool(r_rag.get("error"))) if r_rag else 1,
                   "lc_refusal": int(bool(E.REFUSAL.search(answer))),
                   "rag_refusal": int(bool(r_rag.get("refusal"))) if r_rag else None}
            kws = t.get("keywords") or []
            row |= {"n_kw": len(kws), "lc_kw": kw_hits(answer, kws),
                    "rag_kw": kw_hits(r_rag.get("answer", ""), kws) if r_rag else None}
            if sk == "mh_new":
                row |= {"lc_halves": halves(t, answer),
                        "rag_halves": halves(t, r_rag.get("answer", "")) if r_rag else None}
            if sk == "near_ooc":
                row |= await SA.judge_near(cache, t, answer, real_once)
                row["lc_label"] = row.pop("label")
                row["rag_label"] = r_rag.get("label", "") if r_rag else ""
            row["answer"] = answer
            rows.append(row)
            cache.save()
            print(f"  {sk:<8} {t['id']:<5} {row['lang']} {'νέα ' if fresh else 'παγ.'} in {row['in_tokens']} "
                  f"(cache {row['cached_tokens']}) out {row['out_tokens']} · {row['usd']:.4f} $ · TTFT {row['ttft']}"
                  f" | λέξεις LC {row['lc_kw']}/{row['n_kw']} RAG {row['rag_kw']}"
                  + (f" | μισά LC {row['lc_halves']} RAG {row['rag_halves']}" if sk == "mh_new" else "")
                  + (f" | κριτής LC {row['lc_label']} RAG {row['rag_label']}" if sk == "near_ooc" else "")
                  + f" · σύνολο {spent:.3f} $", flush=True)
        else:
            continue
        break

    # φρέσκες γεννήσεις RAG ΜΟΝΟ για χρόνο (το κείμενο πετιέται)
    rag_ttft = []
    gemini_rest.stream_generate = real_stream
    for sk, ts in tests.items():
        for t in ts[:TIMING_PER_SET]:
            keys = [k for k in (rag_pages[sk].get(t["id"]) or "").split(";") if k]
            if keys:
                _a, ttft, _tot = await answer_timed(ai_core, t["question"],
                                                    [PA.page_of(ai_core, meta_of[k]) for k in keys])
                if ttft:
                    rag_ttft.append(ttft)
    gemini_rest.stream_generate = frozen_stream

    report(rows, rag_ttft, spent, len(cache.data) - n0)
    if rows:
        out = OUT_CSV.replace(".csv", "_smoke.csv") if args.limit else OUT_CSV
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
        with open(out.replace(".csv", "_rag_ttft.json"), "w", encoding="utf-8") as f:
            json.dump(rag_ttft, f)
        print(f"Σώθηκε: {out}")
    return 0


def report(rows, rag_ttft, spent, new_calls) -> None:
    def of(sk):
        return [r for r in rows if r["set"] == sk and not r["rag_error"]]

    print("\n" + "#" * 100)
    print(f"{'':<52}{'RAG (σήμερα)':>16}{'ΟΛΑ ΣΤΟ PROMPT':>18}")
    main = [r for r in of("main") if r["category"] != "out_of_corpus"]
    ooc = [r for r in of("main") if r["category"] == "out_of_corpus"]
    if main:
        n_kw = sum(r["n_kw"] for r in main)
        print(f"{f'κύριο ({len(main)}): λέξεις-κλειδιά στην απάντηση /{n_kw}':<52}"
              f"{sum(r['rag_kw'] for r in main):>16}{sum(r['lc_kw'] for r in main):>18}")
        print(f"{'κύριο: αρνήσεις σε ερωτήσεις ΜΕ απάντηση':<52}"
              f"{sum(r['rag_refusal'] for r in main):>16}{sum(r['lc_refusal'] for r in main):>18}")
    if ooc:
        print(f"{f'κύριο ooc ({len(ooc)}): αρνήσεις':<52}"
              f"{sum(r['rag_refusal'] for r in ooc):>16}{sum(r['lc_refusal'] for r in ooc):>18}")
    mh = of("mh_new")
    if mh:
        print(f"{f'multi_hop νέες ({len(mh)}): μισά με λέξη του paper /{2 * len(mh)}':<52}"
              f"{sum(r['rag_halves'] for r in mh):>16}{sum(r['lc_halves'] for r in mh):>18}")
        print(f"{'  «και τα δύο»':<52}{sum(r['rag_halves'] == 2 for r in mh):>16}"
              f"{sum(r['lc_halves'] == 2 for r in mh):>18}")
        better = [r["id"] for r in mh if r["lc_halves"] > r["rag_halves"]]
        worse = [r["id"] for r in mh if r["lc_halves"] < r["rag_halves"]]
        print(f"  όλα στο prompt καλύτερα σε {len(better)} {better} · χειρότερα σε {len(worse)} {worse}")
    near = of("near_ooc")
    if near:
        def good(lab):
            return lab in ("correct", "cut")
        for lab, name in (("correct", "σωστό (σιωπή ή «δεν υπάρχει»)"), ("soft_leak", "ήπια διαρροή (β)"),
                          ("leak", "διαρροή (γ)")):
            rag_n = sum(good(r["rag_label"]) if lab == "correct" else r["rag_label"] == lab for r in near)
            lc_n = sum(r["lc_label"] == lab for r in near)
            print(f"{f'κοντινές ooc ({len(near)}): {name}':<52}{rag_n:>16}{lc_n:>18}")
        flips = [f"{r['id']} {r['rag_label']}->{r['lc_label']}" for r in near
                 if good(r["rag_label"]) != (r["lc_label"] == "correct")]
        print(f"  άλλαξε συμπεριφορά: {flips}")

    fresh = [r for r in rows if r["fresh"]]
    print(f"\n{'ΚΟΣΤΟΣ / ΧΡΟΝΟΣ':<52}{'RAG (σήμερα)':>16}{'ΟΛΑ ΣΤΟ PROMPT':>18}")
    rag_in = [r["rag_usd"] for r in rows if r["rag_usd"]]
    rag_tok = [r["rag_in_tokens"] for r in rows if r.get("rag_in_tokens")]
    if fresh:
        med = statistics.median
        cached = sum(r["cached_tokens"] for r in fresh) / max(1, sum(r["in_tokens"] or 0 for r in fresh))
        print(f"{'tokens εισόδου ανά ερώτηση (διάμεσος)':<52}{(med(rag_tok) if rag_tok else 0):>16,.0f}"
              f"{med(r['in_tokens'] for r in fresh):>18,.0f}")
        print(f"{'  από αυτά από το cache του Gemini':<52}{'-':>16}{cached:>17.0%}")
        print(f"{'κόστος ανά ερώτηση (διάμεσος, $)':<52}{med(rag_in) if rag_in else 0:>16.4f}"
              f"{med(r['usd'] for r in fresh):>18.4f}")
        print(f"{'TTFT, χρόνος ως την 1η λέξη (διάμεσος, s)':<52}"
              f"{(f'{med(rag_ttft):.2f}' if rag_ttft else '-'):>16}{med(r['ttft'] for r in fresh if r['ttft']):>18.2f}")
        print(f"{'συνολικός χρόνος γέννησης (διάμεσος, s)':<52}{'':>16}{med(r['total_s'] for r in fresh):>18.2f}")
    print(f"\nGemini: {new_calls} νέες κλήσεις · κόστος τρεξίματος ~{spent:.3f} $ (+ κριτής κοντινών, λίγα σεντ)"
          f" · δείγμα χρόνου RAG: {len(rag_ttft)} φρέσκες γεννήσεις")
    print("(ανάκτηση RAG ΔΕΝ περιλαμβάνεται στο TTFT: ~0.8 s στο Ryzen — εδώ το μηχάνημα δεν μετράει)")
    print("#" * 100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="N ανά σετ (δοκιμή· σώζει στο longcontext_smoke.csv)")
    ap.add_argument("--max-usd", type=float, default=3.0, help="σταματάει πριν ξεπεράσει αυτό το κόστος")
    args = ap.parse_args()
    if not E.API_KEY:
        print("!! Λείπει GEMINI_API_KEY")
        return 1
    try:
        fd = os.open(S.LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"!! Τρέχει ήδη scoreboard ή probe στο ίδιο store (κλειδαριά {S.LOCK}) — σταματάω")
        return 1
    try:
        return asyncio.run(run(args))
    finally:
        os.close(fd)
        os.remove(S.LOCK)


if __name__ == "__main__":
    sys.exit(main())

"""Φάση 3.2α — ΣΥΓΚΡΙΣΗ ΜΕ ΑΛΛΑ RAG: ίδιες ερωτήσεις, ίδιο σώμα, ίδιο Gemini, ΙΔΙΟΙ κριτές.

ΤΟ ΕΡΩΤΗΜΑ: πόση αξία προσθέτει η δουλειά πάνω στην ανάκτηση (μετάφραση, BM25, reranker, φύλακας,
σελίδες, αναζήτηση ανά έγγραφο) σε σχέση με (α) το RAG του tutorial και (β) το πιο διαδεδομένο framework
με τις προεπιλογές του; Και ΠΟΥ χάνει — εκεί είναι ό,τι αξίζει να δανειστούμε.

ΣΥΣΤΗΜΑΤΑ:
    ours        το σημερινό σύστημα: οι ΗΔΗ κριμένες απαντήσεις του scoreboard (0 κλήσεις) —
                answers_baseline_{main,near_ooc,tables} + answers_perdoc_mh_new (Φάση 2)
    naive       «RAG του tutorial»: bge-m3, 4 πιο κοντινά κομμάτια (τα ΙΔΙΑ 418 κομμάτια του store μας),
                ΧΩΡΙΣ μετάφραση / BM25 / reranker / φύλακα / σελίδες / αναζήτηση ανά έγγραφο. Η απάντηση
                από τον ΙΔΙΟ ask_ai (ίδιο prompt, παραπομπές, μοντέλο, ρυθμίσεις) -> η διαφορά είναι
                ΜΟΝΟ η ανάκτηση. Το k=4 είναι η προεπιλογή του retriever του LangChain.
    llamaindex  LlamaIndex με τις προεπιλογές του (compare_llamaindex.py γράφει τις απαντήσεις)·
                εδώ ΜΟΝΟ ο κριτής. Ίδιο bge-m3, ίδιο Gemini — όλα τα άλλα δικά του.
ΚΡΙΤΕΣ — οι συναρτήσεις του scoreboard_answers ΑΥΤΟΥΣΙΕΣ, ίδιο πάγωμα (runs/scoreboard_answers.json):
    main 50      judge_main: 4 βαθμοί 1-5· η faithfulness κρίνεται ως προς ό,τι διάβασε ΤΟ ΚΑΘΕ σύστημα
    mh_new 29    judge_mh: ετικέτα ανά paper -> 58 μισά
    near_ooc 42  judge_near: σωστό / ήπια διαρροή / διαρροή· η σιωπή του φύλακα μετράει σωστό
    tables 16    χωρίς κριτή: eval_tables.check, 69 ακριβείς τιμές
ΣΥΓΚΡΙΣΗ ΑΝΑ ΕΡΩΤΗΣΗ (ζευγαρωτή): σε πόσες κερδίζει ο καθένας + ακριβές διωνυμικό p (McNemar)
    πάνω στις ερωτήσεις που διαφωνούν. Ένα p δεν αποφασίζει μόνο του — βλ. θόρυβο παρακάτω.

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (30/9/2026) — ours (από τα αρχεία, --table): τέλειες 42/45 ·
    μισά 44/56 (η m079 έμεινε χωρίς απάντηση στο 402 της Φάσης 2) · κοντινές 40/42 · πίνακες 16/16 ·
    0.47 σεντ/ερώτηση:
    naive       τέλειες 28-38/45 · μισά 20-32/56 · κοντινές 34-41/42 (ο φύλακας κόβει μόνο 2 από τις 42· η
                άρνηση έρχεται κυρίως από το prompt, που το naive ΕΧΕΙ) · πίνακες 9-14/16 · 0.2-0.3 σεντ
    llamaindex  τέλειες 22-35/45 · μισά 15-28/56 · κοντινές 25-36/42 (το πρότυπό του δεν λέει «πες ότι
                λείπει») · πίνακες 7-13/16 · 0.2-0.5 σεντ · ελληνικές ερωτήσεις με ελληνική απάντηση:
                λιγότερες από τις δικές μας
ΚΑΝΟΝΑΣ ΕΡΜΗΝΕΙΑΣ (γραμμένος πριν): διαφορά μετράει ΜΟΝΟ πάνω από τον θόρυβο — τέλειες ±3 (ο κριτής
    γυρίζει ~2/9 κατά μία μονάδα με ίδιο σύστημα), μισά ±4, κοντινές ±2, πίνακες ±1.
    ours μπροστά σε κύριο/μισά, ίσα στις κοντινές -> «η αξία είναι στο να βρεις τις σωστές σελίδες, όχι
        στην άρνηση» — γράφεται έτσι.
    ΟΠΟΙΑ μετρική χάνει το ours πάνω από τον θόρυβο -> γράφεται ΠΡΩΤΗ και γίνεται ο επόμενος υποψήφιος
        δανεισμός (π.χ. ανάλυση πινάκων).
    naive ≈ ours παντού -> «σε 7 papers η στοίβα ανάκτησης δεν φαίνεται στην ποιότητα» — γράφεται κι αυτό.

ΑΠΟΤΕΛΕΣΜΑ (30/9/2026, runs/compare/, table.csv · ~1.8 $ συνολικά):
                                                ours        naive       llamaindex
    κύριο: τέλειες (άρνηση σε ερώτηση με απάντηση = όχι) 42/45  34/45       30/45
    κύριο: «δεν το βρίσκω» ενώ υπάρχει απάντηση  1/45        2/45        5/45
    δύο papers: σωστά μισά                      44/56       20/58       20/56
    δύο papers: «και τα δύο σωστά»              17/28        3/29        2/28
    κοντινές ooc: σωστή άρνηση                  40/42       42/42       41/42
    πίνακες: πλήρως σωστές · τιμές              16 · 69/69  15 · 63/69  13 · 57/69
    ελληνική ερώτηση -> ελληνική απάντηση       18/18       18/18       16/18
    κόστος γέννησης (διάμεσος, σεντ)            0.47        0.19        0.17
    ζευγαρωτά (ours κερδίζει/χάνει): κύριο 10/2 (p=0.04) και 14/2 (p=0.004) · δύο papers 25/0 και 24/0 ·
    κοντινές 0/2 και 0/1 (θόρυβος) · πίνακες 1/0 και 3/0.
    ΚΑΝΟΝΑΣ -> «η αξία είναι στο να βρεις τις σωστές σελίδες, όχι στην άρνηση». Καμία μετρική όπου το
    ours χάνει πάνω από τον θόρυβο -> τίποτα προς δανεισμό από αυτά τα δύο.
    ΟΙ ΚΟΝΤΙΝΕΣ: η ασφάλεια ΔΕΝ έρχεται από τη στοίβα ανάκτησης (ο φύλακας κόβει 2/42) ούτε κυρίως από το
    prompt (το γενικό πρότυπο του LlamaIndex δίνει 41/42) — έρχεται από το ΜΟΝΤΕΛΟ. Οι 2 του ours (n024,
    n038) βγαίνουν από το ΠΕΡΙΣΣΟΤΕΡΟ σχετικό υλικό (8 σελίδες έναντι 2-4 κομματιών) — ίδιο με τη 3.1.
    ΕΥΡΗΜΑ ΓΙΑ ΤΗ ΜΕΤΡΗΣΗ: ο κριτής του κύριου σετ (eval_engine, από το v1) κρίνει ΩΣ ΠΡΟΣ ΤΟ CONTEXT και
    έδωσε 5/5/5/5 σε 7 «το κείμενο δεν περιέχει…» ερωτήσεων που ΕΧΟΥΝ απάντηση (llamaindex 5, naive 2):
    αν η ανάκτηση αποτύχει, η άρνηση βαθμολογείται τέλεια. Ωμές τέλειες 42/36/35 -> διορθωμένες 42/34/30.
    Δεν πιάνει την ΠΑΡΑΛΕΙΨΗ (llamaindex q046: μόνο το μισό, 5/5/5/5)· ο κριτής ανά paper των multi_hop
    (ως προς την ΑΝΑΦΟΡΑ) είναι η αξιόπιστη μέτρηση — και εκεί η διαφορά είναι η μεγαλύτερη.
    Πρόβλεψη: naive 2/5 (κοντινές, πίνακες, κόστος έξω) · llamaindex 4/6 (κοντινές πολύ καλύτερες από
    το 25-36 — το μοντέλο είναι πιο προσεκτικό απ' ό,τι υπέθεσα· κόστος λίγο κάτω).
    Όρια: 7 papers· ένας κριτής (Gemini κρίνει Gemini)· LlamaIndex με ΠΡΟΕΠΙΛΟΓΕΣ, όχι ρυθμισμένο — ένα
    ρυθμισμένο (hybrid, reranker, top-k 8) θα έκλεινε μέρος της διαφοράς· αυτό είναι ακριβώς η δουλειά
    που μετράει η σύγκριση.

ΚΟΣΤΟΣ: naive ~137 γεννήσεις + ~121 κριτές ≈ 0.9 $ · llamaindex ~121 κριτές ≈ 0.5 $ (+ ~0.5 $ οι
    γεννήσεις του στο compare_llamaindex). Φύλακας --max-usd (default 1.5).

ΤΡΕΞΙΜΟ (από τη ρίζα του repo, Git Bash, με σταματημένο backend):
    docker compose run --rm --no-deps -v eval_near_ooc_store:/tmp/eval_near_ooc_chroma backend \
      python -u evaluation/compare_systems.py --system naive --limit 2      # δοκιμή
    ... --system naive · ... --system llamaindex · ... --table              # ο πίνακας: 0 κλήσεις
"""
import argparse
import asyncio
import csv
import inspect
import json
import math
import os
import re
import statistics
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
import scoreboard as S
import scoreboard_answers as SA

import gemini_rest

CMP_DIR = os.path.join(S.RUNS, "compare")
SETS = ("main", "mh_new", "near_ooc", "tables")
OURS = {"main": "answers_baseline_main.json", "mh_new": "answers_perdoc_mh_new.json",
        "near_ooc": "answers_baseline_near_ooc.json", "tables": "answers_baseline_tables.json"}
SYSTEMS = ("ours", "naive", "llamaindex")
NAIVE_K = 4
IN_USD, OUT_USD = 0.30, 2.50                 # measure_cost.py
JUDGE_USD_EST = 0.004                         # ο κριτής δεν επιστρέφει usage· εκτίμηση ανά κλήση
NOISE = {"main": 3, "mh_new": 4, "near_ooc": 2, "tables": 1}


def usd(usage: dict | None) -> float | None:
    if not usage:
        return None
    p = usage.get("promptTokenCount") or 0
    o = (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)
    return (p * IN_USD + o * OUT_USD) / 1e6


def row_usd(r: dict) -> float | None:
    if r.get("prompt_tokens") is None:
        return None
    return ((r["prompt_tokens"] or 0) * IN_USD
            + ((r.get("out_tokens") or 0) + (r.get("thinking_tokens") or 0)) * OUT_USD) / 1e6


def answer_lang(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    greek = sum("Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῾" for c in letters)
    return "el" if letters and greek / len(letters) > 0.3 else "en"


def binom_p(b: int, c: int) -> float:
    """Ακριβές δίπλευρο διωνυμικό (McNemar) στις ερωτήσεις που διαφωνούν."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def naive_pages(ai_core, question: str) -> list:
    """Top-k κομμάτια με ΜΟΝΟ dense (bge-m3, ακριβές cosine), πάνω στην ΑΡΧΙΚΗ ερώτηση."""
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()
    ids = ai_core._dense_exact_ids(dm, question, idx["ids"], NAIVE_K)
    return [(idx["texts"][idx["pos"][i]], idx["metas"][idx["pos"][i]]) for i in ids]


def load_llamaindex() -> dict:
    path = os.path.join(CMP_DIR, "llamaindex_answers.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["rows"]


def base_row(sk: str, t: dict, answer: str, pages: list) -> dict:
    return {"set": sk, "id": t["id"], "lang": S.lang_of(t, t["question"]),
            "category": t.get("category", ""), "question": t["question"], "answer": answer,
            "answer_lang": answer_lang(answer), "n_pages": len(pages),
            "files": ";".join(f"{m.get('file_name')}:{m.get('page')}" for _x, m in pages),
            "refusal": int(bool(E.REFUSAL.search(answer))), "error": ""}


async def judge_row(sk, t, row, pages, answer, cache, ai_core, real_once, sdk_call, T) -> None:
    if sk == "main":
        row |= await SA.judge_main(cache, t, pages, answer, sdk_call)
    elif sk == "mh_new":
        row["both"] = S.both_docs(t, pages)
        for h, ep in enumerate(SA.mention_order(t), 1):
            sig = SA.half_signals(t, ep, pages)
            row |= {f"doc_{h}": sig["doc"], f"ev_{h}": sig["ev"], f"paper_{h}": sig["paper"],
                    f"kw_{h}": sig["kw"]}
        row |= await SA.judge_mh(cache, t, row, pages, answer, ai_core, real_once)
    elif sk == "near_ooc":
        row |= {"near_type": t.get("near_type", ""), "outcome": "no_gate"}
        row |= await SA.judge_near(cache, t, answer, real_once)
    else:
        found = T.check(answer, t["expected"])
        target = int(any(m.get("file_name") == t["doc"] and str(m.get("page")) == str(t["page"])
                         for _x, m in pages))
        row |= {"shape": t.get("shape", ""), "target": target, "values_ok": sum(found),
                "values_total": len(found),
                "missing": "|".join((v[0] if isinstance(v, list) else str(v))
                                    for v, f in zip(t["expected"], found) if not f),
                "verdict": SA.table_verdict(sum(found), len(found), target)}


def summarize(sk: str, rows: list[dict]) -> dict:
    return (SA.summarize_main(rows) if sk == "main" else SA.summarize(rows) if sk == "mh_new"
            else SA.summarize_near(rows) if sk == "near_ooc" else SA.summarize_tables(rows))


async def run_system(args) -> int:
    real_once = gemini_rest.generate_once
    real_stream = gemini_rest.stream_generate
    sets = [s for s in args.sets.split(",") if s]
    tests = {sk: E.load_jsonl(SA.SET_PATHS[sk])[:args.limit or None] for sk in sets}

    ai_core, _near = E.open_store()
    if ai_core.collection.count() != S.EXPECTED_CHUNKS:
        print(f"!! Το store δεν έχει {S.EXPECTED_CHUNKS} chunks — σταματάω")
        return 1
    import eval_engine
    import eval_tables as T
    missing = SA.check_main_judge(inspect.getsource(eval_engine.evaluate_answer))
    src = inspect.getsource(E.judge)
    if missing or not ("temperature=0.0, max_output_tokens=4096" in src and "thinking_budget=1024" in src):
        print(f"!! Κάποιος κριτής ΑΛΛΑΞΕ από το scoreboard — δεν συγκρίνεται: {missing}")
        return 1

    async def sdk_call(prompt):
        resp = await eval_engine.judge_model.generate_content_async(
            prompt, generation_config=eval_engine.genai.GenerationConfig(
                response_mime_type="application/json", temperature=0.0))
        return resp.text

    ai_core._translation_cache.clear()
    frozen = S.Frozen(S.FROZEN_PATH, fresh=False)   # ο ask_ai εδώ δεν ψάχνει· δίχτυ ασφαλείας
    frozen.install()
    cache = SA.AnswerCache(SA.ANS_PATH)
    cache.install_stream(real_stream)
    li = load_llamaindex() if args.system == "llamaindex" else {}

    spent, stopped = 0.0, False
    results: dict[str, list[dict]] = {}
    for sk in sets:
        rows = results.setdefault(sk, [])
        for t in tests[sk]:
            if spent >= args.max_usd:
                print(f"!! ΟΡΙΟ ΚΟΣΤΟΥΣ {args.max_usd} $ — σταματάω (ό,τι πληρώθηκε είναι σωσμένο· "
                      "το ξανατρέξιμο συνεχίζει με 0 κόστος για τα έτοιμα)")
                stopped = True
                break
            t0 = time.perf_counter()
            n_data, n_err = len(cache.data), len(cache.errors)
            gen_new = 0
            if args.system == "naive":
                pages = naive_pages(ai_core, t["question"])
                answer = await E.answer_of(ai_core, t["question"], pages)
                gen_new = len(cache.data) - n_data
                usage = (cache.data.get(cache.last_key) or {}).get("usage") or {}
                p, o, th = gemini_rest.usage_tokens(usage) if usage else (None, None, None)
            else:
                li_row = li.get(f"{sk}:{t['id']}")
                if li_row is None:
                    print(f"!! {sk}:{t['id']} λείπει από το llamaindex_answers.json — τρέξε πρώτα το "
                          "compare_llamaindex.py")
                    return 1
                pages = [(n["text"], {"file_name": n["file_name"], "page": n["page"]})
                         for n in li_row["pages"]]
                answer = li_row["answer"]
                u = li_row.get("usage") or {}
                p = u.get("promptTokenCount")
                o, th = u.get("candidatesTokenCount"), u.get("thoughtsTokenCount")
            row = base_row(sk, t, answer, pages)
            row |= {"prompt_tokens": p, "out_tokens": o, "thinking_tokens": th}
            if len(cache.errors) > n_err:
                row["error"] = cache.errors[-1]
            else:
                try:
                    await judge_row(sk, t, row, pages, answer, cache, ai_core, real_once, sdk_call, T)
                except Exception as e:
                    row["error"] = f"κριτής: {type(e).__name__}: {e}"[:200]
                    cache.errors.append(row["error"])
            new = len(cache.data) - n_data
            if gen_new:
                spent += (row_usd(row) or 0.0)
            spent += JUDGE_USD_EST * max(0, new - gen_new)
            rows.append(row)
            cache.save()
            verdict = (row.get("label") if sk == "near_ooc" else
                       f"τιμές {row.get('values_ok')}/{row.get('values_total')}" if sk == "tables" else
                       f"{row.get('label_1')}/{row.get('label_2')}" if sk == "mh_new" else
                       "".join(str(row.get(s)) for s in SA.SCORES))
            print(f"  {sk:<8} {t['id']:<6} {row['lang']}->{row['answer_lang']} "
                  + (f"ΣΦΑΛΜΑ {row['error']}" if row["error"] else f"{verdict}")
                  + f"  · νέες κλήσεις {new} · ~{spent:.3f} $ · {time.perf_counter() - t0:.1f}s", flush=True)
            if row["error"] and ("429" in row["error"] or "402" in row["error"]):
                print("!! Όριο/credits του Gemini — σταματάω")
                stopped = True
                break
        if stopped:
            break

    cache.save()
    print(f"\nGemini: {cache.misses} νέες / {cache.hits} παγωμένες · ~{spent:.3f} $")
    if cache.errors or frozen.errors:
        print(f"!! ΣΦΑΛΜΑΤΑ: {(cache.errors + frozen.errors)[:5]}")
    if stopped or args.limit:
        print("(δοκιμή ή διακοπή: δεν γράφεται αρχείο αποτελεσμάτων· οι πληρωμένες κλήσεις ΚΡΑΤΙΟΥΝΤΑΙ)")
        for sk, rows in results.items():
            if rows:
                print(f"  {sk}: {len(rows)} ερωτήσεις, σφάλματα {sum(bool(r['error']) for r in rows)}")
        return 1 if stopped else 0
    os.makedirs(CMP_DIR, exist_ok=True)
    for sk, rows in results.items():
        out = os.path.join(CMP_DIR, f"{args.system}_{sk}")
        with open(out + ".json", "w", encoding="utf-8") as f:
            json.dump({"system": args.system, "set": sk, "when": time.strftime("%Y-%m-%d %H:%M"),
                       "naive_k": NAIVE_K if args.system == "naive" else None,
                       "summary": summarize(sk, rows), "rows": rows}, f, ensure_ascii=False, indent=1)
        with open(out + ".csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
        print(f"Σώθηκε: {out}.json / .csv")
    return 1 if (cache.errors or frozen.errors) else 0


# --------------------------------------------------------------------------- ο πίνακας (0 κλήσεις)

def load_rows(system: str, sk: str) -> list[dict] | None:
    path = (os.path.join(S.OUT_DIR, OURS[sk]) if system == "ours"
            else os.path.join(CMP_DIR, f"{system}_{sk}.json"))
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        rows = [r for r in json.load(f)["rows"] if not r.get("error")]
    for r in rows:
        r.setdefault("answer_lang", answer_lang(r.get("answer", "")))
    return rows


# Άρνηση = η απάντηση ΞΕΚΙΝΑΕΙ με «δεν το βρίσκω / το κείμενο δεν περιέχει». Τα γενικά μοτίβα (E.REFUSAL)
# ακόμα και στην 1η πρόταση έπιασαν 4 ΨΕΥΔΩΣ: q035 «…ότι οι cloud functions δεν μπορούν…» (παράθεση
# ισχυρισμού) και q043 «…modules not present in the runtime» ×3 συστήματα· και ΕΧΑΣΑΝ το llamaindex q028
# («does not list»). Ελεγμένο με το μάτι σε ΚΑΘΕ απάντηση του κύριου σετ με φράση άρνησης (3 συστήματα).
_REFUSAL_OPENING = re.compile(
    r"^\s*(?:I\s+cannot\s+find|I\s+could\s+not\s+find|This\s+information\s+is\s+not\s+available"
    r"|The\s+(?:provided|given|selected)\s+[^.]{0,60}?\b(?:does|do)\s+not"
    r"|Δεν\s+(?:βρέθηκε|μπορώ\s+να\s+βρω|υπάρχουν|παρέχονται|αναφέρ)"
    r"|Τα\s+(?:παρεχόμενα\s+|επιλεγμένα\s+)?(?:έγγραφα|κείμενα)\s+δεν)", re.I)


def refused(r: dict) -> bool:
    return bool(_REFUSAL_OPENING.search(r.get("answer") or ""))


def perfect_main(r: dict) -> bool:
    """ΕΥΡΗΜΑ ΤΗΣ ΣΥΓΚΡΙΣΗΣ (30/9): ο κριτής του κύριου σετ (eval_engine, από το v1) κρίνει ΩΣ ΠΡΟΣ ΤΟ
    CONTEXT και δίνει 5/5/5/5 στο «το κείμενο δεν περιέχει…» όταν η ανάκτηση απέτυχε, σε ερώτηση που
    ΕΧΕΙ απάντηση (llamaindex q017/q039, naive q007/q028). Εδώ μια τέτοια άρνηση ΔΕΝ είναι τέλεια.
    Δεν πιάνει την ΠΑΡΑΛΕΙΨΗ (llamaindex q046: απάντησε μόνο για το PyWren, 5/5/5/5) — γι' αυτό η
    πιο αξιόπιστη μέτρηση είναι τα multi_hop, όπου ο κριτής κρίνει ΑΝΑ PAPER ως προς την αναφορά."""
    return all(r.get(s) == 5 for s in SA.SCORES) and not refused(r)


def unit_outcomes(sk: str, rows: list[dict]) -> dict:
    """Δυαδικό «πέτυχε;» ανά μονάδα σύγκρισης (ερώτηση, ή μισό στα multi_hop)."""
    if sk == "main":
        return {r["id"]: perfect_main(r) for r in rows if r["category"] != "out_of_corpus"}
    if sk == "mh_new":
        return {f"{r['id']}:{h}": r[f"label_{h}"] == "correct" for r in rows for h in (1, 2)}
    if sk == "near_ooc":
        return {r["id"]: r["label"] in ("correct", "cut") for r in rows}
    return {r["id"]: r["verdict"] == "σωστή" for r in rows}


def metric_lines(sk: str, rows: list[dict]) -> list[tuple[str, str]]:
    """(ετικέτα, τιμή «x/n») — ο παρονομαστής στην ΤΙΜΗ: ένα σφάλμα σε ένα σύστημα φαίνεται, δεν χάνεται."""
    if sk == "main":
        inn = [r for r in rows if r["category"] != "out_of_corpus"]
        ooc = [r for r in rows if r["category"] == "out_of_corpus"]
        m = SA._means(inn)
        el = [r for r in inn if r["lang"] == "el"]
        return [("κύριο: τέλειες 5/5/5/5 (όπως τις δίνει ο κριτής)", f"{m['perfect']}/{len(inn)}"),
                ("κύριο: τέλειες, άρνηση σε ερώτηση ΜΕ απάντηση = όχι",
                 f"{sum(perfect_main(r) for r in inn)}/{len(inn)}"),
                ("κύριο: ακρίβεια · πληρότητα · στήριξη (μέσοι)",
                 f"{m['accuracy']:.2f}·{m['completeness']:.2f}·{m['faithfulness']:.2f}"),
                ("κύριο: «δεν το βρίσκω» ενώ υπάρχει απάντηση", f"{sum(refused(r) for r in inn)}/{len(inn)}"),
                ("κύριο: εκτός σώματος, αρνήσεις", f"{sum(r['refusal'] for r in ooc)}/{len(ooc)}"),
                ("ελληνική ερώτηση -> ελληνική απάντηση",
                 f"{sum(r['answer_lang'] == 'el' for r in el)}/{len(el)}")]
    if sk == "mh_new":
        labs = [r[f"label_{h}"] for r in rows for h in (1, 2)]
        return [("δύο papers: σωστά μισά", f"{labs.count('correct')}/{len(labs)}"),
                ("δύο papers: «και τα δύο σωστά»",
                 f"{sum(r['label_1'] == r['label_2'] == 'correct' for r in rows)}/{len(rows)}"),
                ("δύο papers: μισά ΧΩΡΙΣ στήριξη (ψευδαίσθηση)", f"{labs.count('unsupported')}/{len(labs)}")]
    if sk == "near_ooc":
        return [("κοντινές ooc: σωστή άρνηση",
                 f"{sum(r['label'] in ('correct', 'cut') for r in rows)}/{len(rows)}"),
                ("κοντινές ooc: ήπια διαρροή · διαρροή",
                 f"{sum(r['label'] == 'soft_leak' for r in rows)} · {sum(r['label'] == 'leak' for r in rows)}")]
    return [("πίνακες: πλήρως σωστές", f"{sum(r['verdict'] == 'σωστή' for r in rows)}/{len(rows)}"),
            ("πίνακες: τιμές", f"{sum(int(r['values_ok']) for r in rows)}/"
                               f"{sum(int(r['values_total']) for r in rows)}")]


def table() -> int:
    data = {s: {sk: load_rows(s, sk) for sk in SETS} for s in SYSTEMS}
    systems = [s for s in SYSTEMS if any(data[s][sk] for sk in SETS)]
    w0, w = 50, 16
    print("\n" + "#" * (w0 + w * len(systems)))
    print(f"{'':<{w0}}" + "".join(f"{s:>{w}}" for s in systems))
    out_csv = []
    for sk in SETS:
        base = data["ours"][sk]
        if not base:
            continue
        labels = [lab for lab, _v in metric_lines(sk, base)]
        cols = {s: dict(metric_lines(sk, data[s][sk])) if data[s][sk] else {} for s in systems}
        for lab in labels:
            print(f"{lab:<{w0}}" + "".join(f"{cols[s].get(lab, '—'):>{w}}" for s in systems))
            out_csv.append({"metric": lab, **{s: cols[s].get(lab, "") for s in systems}})
    print(f"\n{'κόστος γέννησης ανά ερώτηση (διάμεσος, σεντ $)':<{w0}}", end="")
    for s in systems:
        vals = [row_usd(r) for sk in SETS for r in (data[s][sk] or []) if row_usd(r) is not None]
        print(f"{(f'{100 * statistics.median(vals):.2f}' if vals else '—'):>{w}}", end="")
    print("\n" + "#" * (w0 + w * len(systems)))

    print("\nΖΕΥΓΑΡΩΤΑ, ΑΝΑ ΕΡΩΤΗΣΗ (ours έναντι του καθενός): ours κερδίζει / χάνει · p · πάνω από θόρυβο;")
    for s in systems:
        if s == "ours":
            continue
        for sk in SETS:
            a, b = data["ours"][sk], data[s][sk]
            if not a or not b:
                continue
            ua, ub = unit_outcomes(sk, a), unit_outcomes(sk, b)
            common = ua.keys() & ub.keys()
            win = sorted(k for k in common if ua[k] and not ub[k])
            lose = sorted(k for k in common if ub[k] and not ua[k])
            diff = len(win) - len(lose)
            verdict = ("ours ΜΠΡΟΣΤΑ" if diff > NOISE[sk] else f"{s} ΜΠΡΟΣΤΑ" if -diff > NOISE[sk]
                       else "μέσα στον θόρυβο")
            print(f"  {s:<11} {sk:<9} {len(win):>3} / {len(lose):<3} · p={binom_p(len(win), len(lose)):.3f}"
                  f" · {verdict} (θόρυβος ±{NOISE[sk]})" + (f"   χάνει στα: {' '.join(lose)}" if lose else ""))
    os.makedirs(CMP_DIR, exist_ok=True)
    path = os.path.join(CMP_DIR, "table.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["metric", *systems])
        wr.writeheader()
        wr.writerows(out_csv)
    print(f"\nΣώθηκε: {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", choices=("naive", "llamaindex"))
    ap.add_argument("--table", action="store_true", help="μόνο ο πίνακας σύγκρισης (0 κλήσεις)")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--limit", type=int, default=0, help="N ανά σετ (δοκιμή· δεν γράφει αποτελέσματα)")
    ap.add_argument("--max-usd", type=float, default=1.5)
    args = ap.parse_args()
    if args.table:
        return table()
    if not args.system:
        ap.error("--system naive|llamaindex ή --table")
    if not E.API_KEY:
        print("!! Λείπει GEMINI_API_KEY")
        return 1
    try:
        fd = os.open(S.LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"!! Τρέχει ήδη scoreboard ή probe στο ίδιο store (κλειδαριά {S.LOCK}) — σταματάω")
        return 1
    try:
        return asyncio.run(run_system(args))
    finally:
        os.close(fd)
        os.remove(S.LOCK)


if __name__ == "__main__":
    raise SystemExit(main())

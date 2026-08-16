"""Probe: γενικεύεται η μετάφραση αν ο domain descriptor ΠΑΡΑΓΕΤΑΙ από το corpus
αντί να είναι ΚΑΡΦΩΜΕΝΟΣ «cloud/serverless»;

ΤΟ ΠΡΟΒΛΗΜΑ (εντοπίστηκε 15/8/2026):
Το optimize_query (ai_core.py:448-459) έχει ΚΑΡΦΩΜΕΝΟ το πεδίο μέσα στο prompt
μετάφρασης: «The corpus is ... cloud computing, serverless ...» + καρφωμένα
παραδείγματα (μηχανήματα->servers, έλεγχος->auditability). Αν αλλάξει το corpus
(π.χ. ιατρικά papers), το prompt είναι ΛΑΘΟΣ. Είναι σύζευξη corpus<->prompt.

Η ΥΠΟΘΕΣΗ ΠΡΟΣ ΜΕΤΡΗΣΗ:
Αν παράγουμε τον descriptor ΑΠΟ το corpus (μία κλήση, στο ingestion), το ΙΔΙΟ
prompt δουλεύει για ΟΠΟΙΟΔΗΠΟΤΕ πεδίο. Αλλά ΜΟΝΟ αν ο παραγόμενος descriptor
οδηγεί στις ΙΔΙΕΣ σωστές μεταφράσεις με τον καρφωμένο — αλλιώς θυσιάζουμε
ποιότητα για γενικότητα, και αυτό πρέπει να το ΞΕΡΟΥΜΕ, όχι να το υποθέσουμε.

ΤΙ ΜΕΤΡΑΕΙ (read-only· ΜΗΔΕΝ αλλαγή στην παραγωγή· ΔΕΝ αγγίζει το translation
cache — καλεί το gemini_rest.generate_once απευθείας με δικά του prompts):
Για κάθε ΕΛΛΗΝΙΚΗ golden ερώτηση, μεταφράζει με ΤΡΕΙΣ εκδοχές prompt:
  hardcoded : ακριβές αντίγραφο της παραγωγής (control)
  derived   : ίδια δομή, ο descriptor ΠΑΡΑΓΕΤΑΙ από το corpus
  none      : σκέτη μετάφραση, χωρίς πεδίο (baseline του «τι χάνεις»)
και για καθεμία τρέχει το retrieval (dense+BM25+RRF+rerank) -> best logit,
δηλαδή ΑΚΡΙΒΩΣ ό,τι συγκρίνει το gate.

ΤΟ ΚΡΙΤΗΡΙΟ:
  in-corpus:     best(derived) >= best(hardcoded)  (δεν χάνει ποιότητα)
  out_of_corpus: best(derived) κάτω από το gate     (δεν διαρρέει)
Αν το derived πετυχαίνει και τα δύο, το swap είναι ασφαλές ΚΑΙ γενικεύει δωρεάν.

ΚΟΣΤΟΣ API: (πλήθος ελληνικών ερωτήσεων) x 3 + 1 (descriptor). Χρησιμοποίησε
--limit για δοκιμή σε λίγες πρώτα. Ο descriptor αποθηκεύεται σε JSON -> δεν
ξανα-παράγεται (δώσε --refresh-descriptor για να τον ξαναφτιάξεις).

    docker compose exec backend python evaluation/probe_domain_translation.py --limit 4
    docker compose exec backend python evaluation/probe_domain_translation.py --csv evaluation/runs/domain_translation.csv
"""
import argparse
import asyncio
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")
import ai_core
import gemini_rest
from evaluation.eval_engine import GOLDEN_CORPUS

DEFAULT_SETS = [
    "/app/evaluation/golden_set_50.jsonl",
    "/app/evaluation/golden_multihop_new.jsonl",
]
DESCRIPTOR_FILE = "/app/evaluation/runs/domain_descriptor.json"


def load_sets(paths):
    tests = []
    for p in paths:
        if not os.path.exists(p):
            print(f"ΠΑΡΑΛΕΙΨΗ (δεν υπάρχει): {p}")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    t["_set"] = os.path.basename(p)
                    tests.append(t)
    return tests


async def derive_descriptor(allowed_ids, refresh=False):
    """Παράγει τον domain descriptor ΑΠΟ το corpus — μία κλήση Gemini, cached.
    Αυτό ακριβώς θα γινόταν ΜΙΑ ΦΟΡΑ στο ingestion στην πραγματική υλοποίηση."""
    if os.path.exists(DESCRIPTOR_FILE) and not refresh:
        with open(DESCRIPTOR_FILE, encoding="utf-8") as f:
            d = json.load(f)
        print(f"Descriptor από cache ({DESCRIPTOR_FILE}):")
        print(f"  DOMAIN: {d['domain']}")
        print(f"  TERMS:  {d['terms'][:200]}...\n")
        return d

    # Δείγμα από ΟΛΟ το corpus, ομοιόμορφα, κομμένο ώστε να χωράει σε ένα prompt.
    docs = ai_core.collection.get(where=ai_core._build_where(GOLDEN_CORPUS, None),
                                  include=["documents"])["documents"]
    step = max(1, len(docs) // 30)
    sample = "\n---\n".join(d[:500] for d in docs[::step][:30])

    prompt = (
        "You are analyzing a document corpus to build a translation glossary for "
        "an English-only search engine. Below are excerpts from the corpus.\n"
        "1) In ONE short line, name the domain/field of these documents.\n"
        "2) List up to 20 STANDARD ENGLISH TECHNICAL TERMS that a search over this "
        "corpus would rely on — the exact vocabulary the documents use.\n"
        "Output EXACTLY this format, nothing else:\n"
        "DOMAIN: <one line>\n"
        "TERMS: term1, term2, term3, ...\n\n"
        f"--- CORPUS EXCERPTS ---\n{sample}"
    )
    raw = (await gemini_rest.generate_once(
        prompt, model=ai_core.GEMINI_MODEL, api_key=ai_core.GEMINI_API_KEY)).strip()

    domain, terms = "", ""
    for line in raw.splitlines():
        if line.upper().startswith("DOMAIN:"):
            domain = line.split(":", 1)[1].strip()
        elif line.upper().startswith("TERMS:"):
            terms = line.split(":", 1)[1].strip()
    d = {"domain": domain, "terms": terms, "raw": raw}
    os.makedirs(os.path.dirname(DESCRIPTOR_FILE), exist_ok=True)
    with open(DESCRIPTOR_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print("Descriptor ΠΑΡΑΧΘΗΚΕ από το corpus (μία κλήση Gemini):")
    print(f"  DOMAIN: {domain}")
    print(f"  TERMS:  {terms}\n")
    return d


def prompt_hardcoded(query):
    """Ακριβές αντίγραφο της παραγωγής (ai_core.py:448-459) — το control."""
    return (
        "You are a translation assistant for an English-only academic search engine. "
        "The corpus is computer-science papers on cloud computing, serverless "
        "computing and distributed systems. Translate the user's question to English "
        "using the STANDARD TECHNICAL TERMINOLOGY of that field: map everyday words to "
        "the term a paper would actually use (e.g. Greek 'μηχανήματα' -> 'servers', "
        "NOT 'machinery'; 'έλεγχος' in a compliance context -> 'auditability'). "
        "Preserve every domain term already present in the question - never drop or "
        "generalize one. Output ONLY a concise English "
        "search query with the key terms. No quotes, no extra text.\n\n"
        f"User question: {query}"
    )


def prompt_derived(query, desc):
    """Ίδια δομή με το hardcoded, ΑΛΛΑ το πεδίο + το γλωσσάρι έρχονται από τον
    descriptor που παρήχθη από το corpus. ΚΑΝΕΝΑ καρφωμένο ελληνικό παράδειγμα:
    δοκιμάζει αν το αγγλικό γλωσσάρι ΜΟΝΟ του οδηγεί στους σωστούς όρους."""
    domain = desc["domain"] or "the documents in this corpus"
    return (
        "You are a translation assistant for an English-only academic search engine. "
        f"The corpus is about: {domain}. "
        "Translate the user's question to English using the STANDARD TECHNICAL "
        "TERMINOLOGY of that field: map everyday words to the term a document would "
        "actually use, NOT the literal everyday translation. "
        f"Use the vocabulary of this corpus glossary: {desc['terms']}. "
        "Preserve every domain term already present in the question - never drop or "
        "generalize one. Output ONLY a concise English search query with the key "
        "terms. No quotes, no extra text.\n\n"
        f"User question: {query}"
    )


def prompt_none(query):
    """Σκέτη μετάφραση — baseline του «τι χάνεις χωρίς πεδίο»."""
    return (
        "Translate the user's question to English. Output ONLY a concise English "
        "search query with the key terms. No quotes, no extra text.\n\n"
        f"User question: {query}"
    )


async def translate(prompt, model=None, thinking_budget=0, max_output_tokens=256):
    """model=None -> το μοντέλο της παραγωγής (Flash, thinking=0). Δώσε ρητό model
    για να μετρήσεις αν ισχυρότερο μοντέλο μεταφράζει καλύτερα. ΠΡΟΣΟΧΗ: τα Pro 3.x
    ΑΠΑΙΤΟΥΝ thinking>0 — δώσε thinking_budget θετικό (αυτό είναι ΚΑΙ κόστος: το Flash
    μεταφράζει με 0 thinking tokens, το Pro όχι)."""
    return (await gemini_rest.generate_once(
        prompt, model=model or ai_core.GEMINI_MODEL,
        api_key=ai_core.GEMINI_API_KEY, thinking_budget=thinking_budget,
        max_output_tokens=max_output_tokens)).strip(' "\'\n')


def best_logit(query, allowed_ids, idx, dm):
    """Αναπαράγει τα βήματα 2-5 του search_documents -> best rerank logit.
    ΙΔΙΕΣ συναρτήσεις με το measure_gate_margin.py."""
    d_ids = ai_core._dense_exact_ids(dm, query, allowed_ids,
                                     min(ai_core.DENSE_CANDIDATES, len(allowed_ids)))
    s_ids = ai_core._bm25_sparse_ids(idx, query, allowed_ids, ai_core.DENSE_CANDIDATES)
    rrf = ai_core._rrf_fuse(d_ids, s_ids, idx["ids"], idx["texts"], idx["metas"],
                            k=60, top_n=ai_core.RERANK_CANDIDATES, pos=idx["pos"])
    pairs = [[query, it[1]] for it in rrf]
    scores = [float(x) for x in
              ai_core.reranker.predict(pairs, batch_size=ai_core.RERANK_BATCH_SIZE)]
    return max(scores)


PROD_DESCRIPTOR = "/app/corpus_descriptor.json"


def load_prod_descriptor():
    """Ο ΖΩΝΤΑΝΟΣ descriptor της παραγωγής ({domain, terms}). Έτσι το derived
    prompt του probe είναι ΤΑΥΤΟΣΗΜΟ με ό,τι τρέχει στο optimize_query."""
    with open(PROD_DESCRIPTOR, encoding="utf-8") as f:
        return json.load(f)


async def run_model_comparison(greek, alt_model, thr, allowed_ids, idx, dm, csv_path):
    """Κρατά ΣΤΑΘΕΡΟ το production prompt (derived) και αλλάζει ΜΟΝΟ το μοντέλο
    μετάφρασης: Flash (παραγωγή) vs alt_model. Απαντά ΑΚΡΙΒΩΣ στο ερώτημα «αξίζει
    ισχυρότερο μοντέλο στη μετάφραση;» — 2 κλήσεις/ερώτηση, όχι 3.

    Κριτήριο: αν το alt δεν ανεβάζει το best-logit πάνω από τον θόρυβο (±0.04) ΚΑΙ
    δεν προσθέτει διαρροή out_of_corpus, το ισχυρότερο μοντέλο ΔΕΝ αξίζει."""
    desc = load_prod_descriptor()
    print(f"Descriptor παραγωγής ({PROD_DESCRIPTOR}):")
    print(f"  DOMAIN: {desc['domain']}")
    print(f"  TERMS:  {desc['terms'][:200]}...\n")
    print(f"ΜΟΝΤΕΛΟ ΜΕΤΑΦΡΑΣΗΣ: flash={ai_core.GEMINI_MODEL}  vs  alt={alt_model}")
    print(f"corpus: {len(allowed_ids)} chunks · gate = {thr}\n")
    print(f"{'id':<7} {'cat':<14} {'flash':>7} {'alt':>7} {'Δ':>7}")
    print("-" * 70)

    rows = []
    for t in greek:
        q = t["question"]
        p = prompt_derived(q, desc)
        tr_flash = await translate(p)                 # μοντέλο παραγωγής (thinking=0)
        # τα Pro 3.x απαιτούν thinking>0· max_output χωράει thinking + μετάφραση
        tr_alt = await translate(p, model=alt_model,
                                 thinking_budget=256, max_output_tokens=768)
        b_flash = best_logit(tr_flash, allowed_ids, idx, dm)
        b_alt = best_logit(tr_alt, allowed_ids, idx, dm)
        cat = t.get("category", "")
        rows.append(dict(id=t["id"], set=t["_set"], category=cat, question=q,
                         tr_flash=tr_flash, tr_alt=tr_alt,
                         best_flash=b_flash, best_alt=b_alt))
        print(f"{t['id']:<7} {cat:<14} {b_flash:>7.2f} {b_alt:>7.2f} "
              f"{b_alt - b_flash:>+7.2f}", flush=True)
        if tr_flash != tr_alt:
            print(f"    flash: {tr_flash}")
            print(f"    alt  : {tr_alt}")

    inc = [r for r in rows if r["category"] != "out_of_corpus"]
    ooc = [r for r in rows if r["category"] == "out_of_corpus"]

    print("\n" + "=" * 70)
    print("ΑΠΟΤΕΛΕΣΜΑ: αξίζει το ισχυρότερο μοντέλο στη μετάφραση;")
    print("=" * 70)
    if inc:
        deltas = [r["best_alt"] - r["best_flash"] for r in inc]
        better = [r for r in inc if r["best_alt"] > r["best_flash"] + 0.5]
        worse = [r for r in inc if r["best_alt"] < r["best_flash"] - 0.5]
        print(f"  in-corpus (n={len(inc)}): μέσο Δ {statistics.mean(deltas):>+.3f} logit "
              f"(θόρυβος ±0.04) · καλύτερο σε {len(better)} · χειρότερο σε {len(worse)}")
        print(f"    flash μέσος {statistics.mean([r['best_flash'] for r in inc]):>6.2f} · "
              f"alt μέσος {statistics.mean([r['best_alt'] for r in inc]):>6.2f}")
    if ooc:
        leak_f = [r for r in ooc if r["best_flash"] >= thr]
        leak_a = [r for r in ooc if r["best_alt"] >= thr]
        print(f"  out_of_corpus διαρροές (>= gate {thr}): flash {len(leak_f)} · "
              f"alt {len(leak_a)}"
              + (f" -> {', '.join(r['id'] for r in leak_a)}" if leak_a else ""))

    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {csv_path}")
    return 0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--sets", nargs="*", default=DEFAULT_SETS)
    ap.add_argument("--limit", type=int, default=None,
                    help="μόνο οι πρώτες N ελληνικές ερωτήσεις (για δοκιμή quota)")
    ap.add_argument("--ids", nargs="*", default=None,
                    help="μόνο συγκεκριμένα ids (π.χ. --ids q029 q042)")
    ap.add_argument("--refresh-descriptor", action="store_true")
    ap.add_argument("--model", default=None,
                    help="ισχυρότερο μοντέλο ΜΟΝΟ για τη μετάφραση· ενεργοποιεί τη "
                         "σύγκριση Flash vs <model> με ΤΟΝ ΙΔΙΟ (production) descriptor")
    args = ap.parse_args()

    tests = load_sets(args.sets)
    greek = [t for t in tests if ai_core._has_greek(t["question"])]
    if args.ids:
        greek = [t for t in greek if t["id"] in set(args.ids)]
    if args.limit:
        greek = greek[:args.limit]
    if not greek:
        print("Καμία ελληνική ερώτηση.")
        return 1

    per_q = 2 if args.model else 3
    n_calls = len(greek) * per_q + (0 if args.model or (os.path.exists(DESCRIPTOR_FILE)
                                    and not args.refresh_descriptor) else 1)
    print(f"Ελληνικές ερωτήσεις: {len(greek)} · κλήσεις Gemini: ~{n_calls}\n")

    allowed_ids = ai_core.collection.get(
        where=ai_core._build_where(GOLDEN_CORPUS, None), include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()
    thr = ai_core.MIN_RERANK_SCORE

    if args.model:
        return await run_model_comparison(greek, args.model, thr, allowed_ids,
                                          idx, dm, args.csv)

    desc = await derive_descriptor(allowed_ids, refresh=args.refresh_descriptor)

    print(f"corpus: {len(allowed_ids)} chunks · gate = {thr}\n")
    print(f"{'id':<7} {'cat':<14} {'hard':>7} {'deriv':>7} {'none':>7}   translations")
    print("-" * 100)

    rows = []
    for t in greek:
        q = t["question"]
        p_hard, p_deriv, p_none = prompt_hardcoded(q), prompt_derived(q, desc), prompt_none(q)
        tr_hard, tr_deriv, tr_none = (await translate(p_hard),
                                      await translate(p_deriv),
                                      await translate(p_none))
        b_hard = best_logit(tr_hard, allowed_ids, idx, dm)
        b_deriv = best_logit(tr_deriv, allowed_ids, idx, dm)
        b_none = best_logit(tr_none, allowed_ids, idx, dm)
        cat = t.get("category", "")
        rows.append(dict(id=t["id"], set=t["_set"], category=cat, question=q,
                         tr_hard=tr_hard, tr_deriv=tr_deriv, tr_none=tr_none,
                         best_hard=b_hard, best_deriv=b_deriv, best_none=b_none))
        print(f"{t['id']:<7} {cat:<14} {b_hard:>7.2f} {b_deriv:>7.2f} {b_none:>7.2f}",
              flush=True)
        print(f"    hard : {tr_hard}")
        print(f"    deriv: {tr_deriv}")
        print(f"    none : {tr_none}")

    inc = [r for r in rows if r["category"] != "out_of_corpus"]
    ooc = [r for r in rows if r["category"] == "out_of_corpus"]

    print("\n" + "=" * 100)
    print("ΣΥΓΚΡΙΣΗ best-logit (ό,τι κρίνει το gate)")
    print("=" * 100)
    for label, grp in (("in-corpus", inc), ("out_of_corpus", ooc)):
        if not grp:
            continue
        print(f"\n  {label} (n={len(grp)})")
        for key, name in (("best_hard", "hardcoded"), ("best_deriv", "derived"),
                          ("best_none", "none")):
            v = [r[key] for r in grp]
            print(f"    {name:<10} μέσος {statistics.mean(v):>7.2f} · "
                  f"min {min(v):>7.2f} · max {max(v):>7.2f}")

    # Το κρίσιμο κριτήριο, ανά ερώτηση.
    print("\n" + "=" * 100)
    print("ΚΡΙΤΗΡΙΟ: derived vs hardcoded")
    print("=" * 100)
    if inc:
        worse = [r for r in inc if r["best_deriv"] < r["best_hard"] - 0.5]
        better = [r for r in inc if r["best_deriv"] > r["best_hard"] + 0.5]
        print(f"  in-corpus: derived ΚΑΛΥΤΕΡΟ σε {len(better)}, "
              f"ΧΕΙΡΟΤΕΡΟ (>0.5 logit) σε {len(worse)}"
              + (f" -> {', '.join(r['id'] for r in worse)}" if worse else ""))
    if ooc:
        leak_h = [r for r in ooc if r["best_hard"] >= thr]
        leak_d = [r for r in ooc if r["best_deriv"] >= thr]
        print(f"  out_of_corpus διαρροές (>= gate {thr}): "
              f"hardcoded {len(leak_h)} · derived {len(leak_d)}"
              + (f" -> {', '.join(r['id'] for r in leak_d)}" if leak_d else ""))

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

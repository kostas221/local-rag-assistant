"""Φάση 0.2 — νέες ΔΥΣΚΟΛΕΣ ΠΑΡΑΦΡΑΣΕΙΣ: το hard set από 16 σε ~55.

ΤΟ ΕΡΩΤΗΜΑ ΜΕ ΑΠΛΑ ΛΟΓΙΑ:
    Οι χρήστες δεν ρωτάνε με τις λέξεις των papers. Το hard set παίρνει ερωτήσεις
    που ΞΕΡΟΥΜΕ ότι απαντιούνται και τις ξαναγράφει όπως θα τις έγραφε ένας
    άνθρωπος (χωρίς ορολογία, αόριστα, σε καθομιλουμένα ελληνικά...). Αν το σύστημα
    κόψει μια τέτοια ερώτηση, η κοπή είναι ΑΠΟΔΕΔΕΙΓΜΕΝΑ λάθος: η απάντηση υπάρχει.
    Είναι ο μόνος οδηγός βελτίωσης που έχει μείνει — αλλά με n=16 κάθε ερώτηση
    μετράει 6 ποσοστιαίες μονάδες. Εδώ το μεγαλώνουμε.

ΠΩΣ (ίδια συνταγή με το build_near_ooc):
    1. ΓΟΝΙΚΕΣ: οι 40 in-corpus ερωτήσεις (golden_set_50 + golden_multihop_new) που
       ΔΕΝ έχουν ήδη παράφραση. Κληρονομούνται keywords, reference_answer, κατηγορία.
    2. ΓΕΝΝΗΣΗ: 2 παραφράσεις ανά γονική, σε ΔΙΑΦΟΡΕΤΙΚΟΥΣ τύπους, κυκλικά ώστε οι
       5 τύποι να είναι ισορροπημένοι (το «short» όχι σε multi_hop: δύο έγγραφα
       δεν χωράνε σε 6 λέξεις).
    3. ΜΗΧΑΝΙΚΑ ΦΙΛΤΡΑ: η παράφραση ΔΕΝ περιέχει keyword της γονικής (αλλιώς δεν
       είναι δύσκολη — το BM25 τη βρίσκει)· short ≤ 7 λέξεις· greek = ελληνικά.
    4. ΕΠΑΛΗΘΕΥΣΗ με ΟΛΟ το σώμα στο context, τέσσερα ερωτήματα:
         same_need       — η reference_answer απαντά ΠΛΗΡΩΣ και ΣΩΣΤΑ την παράφραση;
         unique          — δείχνει στο ΙΔΙΟ σημείο των papers, όχι και σε άλλο;
         self_contained  — κατανοητή ως ΠΡΩΤΟ μήνυμα συνομιλίας;
         no_hint         — ΔΕΝ προδίδει κομμάτι της απάντησης; (προστέθηκε μετά το smoke:
                           η p001 ρωτούσε τον ορισμό του cloud «καλύπτοντας τα online
                           προγράμματα ΚΑΙ τα μηχανήματα, ειδικά όταν πληρώνεις όσο
                           χρησιμοποιείς» = η ίδια η απάντηση. Ο generator βλέπει την
                           απάντηση για να κρατήσει το νόημα, και τη «δανείζεται».)
       Ο ελεγκτής κρίνει την ΑΓΓΛΙΚΗ εκδοχή (question_en): οι έλεγχοι με LLM είναι
       γλωσσοεξαρτώμενοι (grounding verdict, 11/8: ελληνικές 6 ΝΑΙ/14 ΟΧΙ, αγγλικές
       34/2). Η δυσκολία της μετάφρασης μένει στην ίδια την ερώτηση — εκεί τη μετράμε.
    5. ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ (runs/hard_review.json): ο άνθρωπος μόνο ΠΕΤΑΕΙ, και μόνο
       για ΕΓΚΥΡΟΤΗΤΑ (άλλαξε το νόημα, διφορούμενη, διπλότυπη). ΤΟ ΣΥΣΤΗΜΑ ΔΕΝ
       ΤΡΕΧΕΙ ΠΡΙΝ ΠΑΓΩΣΕΙ ΤΟ ΣΕΤ — αλλιώς θα πετούσαμε ό,τι αποτυγχάνει, ακριβώς η
       survivorship bias που έκανε το gate «61/61 τέλειο».

ΔΙΑΦΟΡΑ ΑΠΟ ΤΙΣ 16 ΠΑΛΙΕΣ — ΣΚΟΠΙΜΗ:
    Οι h012/h016 («η συγκεκριμένη υλοποίηση», «εκείνο το παλιότερο σύστημα») δεν έχουν
    υποκείμενο· η σιωπή είναι η ΣΩΣΤΗ απάντηση, και στην παραγωγή τις λύνει το
    ιστορικό της συνομιλίας (_rewrite_query). Δεν μετράνε «λάθος κοπή». Οι νέες
    πρέπει να είναι ΑΥΤΟΤΕΛΕΙΣ: δύσκολες στη διατύπωση, όχι ελλιπείς στο νόημα.
    Οι 16 παλιές ΔΕΝ αγγίζονται (συγκρισιμότητα με το ιστορικό «13/16»)· οι νέες
    γράφονται σε ΞΕΧΩΡΙΣΤΟ αρχείο, golden_hard_new.jsonl, με σταθερά id p001 -> h101.

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (25/9/2026):
    η μηχανή κρατά 45-60 από τις 80 (56-75%)· οι περισσότερες απώλειες στο short και
    στο meta (μοναδικότητα στόχου)· μετά τον άνθρωπο ~40.

ΚΟΣΤΟΣ: 4 κλήσεις γέννησης (μικρές) + 7 κλήσεις επαλήθευσης × ~104k tokens (όλο το
σώμα, μετρημένο στο smoke) ≈ 0.75M tokens εισόδου. Χρόνος ~5 λεπτά (με αναμονή ανάμεσα
στις επαληθεύσεις).

SMOKE (4 γονικές, 8 υποψήφιες): η μηχανή κράτησε 7 — ΑΛΛΑ η p001 πρόδιδε την απάντηση
(-> κριτήριο no_hint) και το short «Three new aspects?» κόπηκε σωστά ως ακατανόητο
(-> το short πρέπει να λέει το θέμα). Το 88% ΔΕΝ είναι εκτίμηση: smoke n=8 δεν εκτιμά
απόδοση (build_near_ooc, 1ος γύρος: έπεσε έξω 2×).

    # δοκιμή σε 4 γονικές (8 υποψήφιες, 1 επαλήθευση):
    docker compose exec backend python evaluation/build_hard_paraphrase.py --parents 4
    # πλήρες:
    docker compose exec backend python evaluation/build_hard_paraphrase.py
    # μετά τον ανθρώπινο έλεγχο — ΜΗΔΕΝ κλήσεις:
    docker compose exec backend python evaluation/build_hard_paraphrase.py --apply-review
"""
import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_near_ooc as B

import gemini_rest

HERE = B.HERE
GOLDEN_50 = os.path.join(HERE, "golden_set_50.jsonl")
MULTIHOP = os.path.join(HERE, "golden_multihop_new.jsonl")
HARD_OLD = os.path.join(HERE, "golden_hard_paraphrase.jsonl")
REVIEW_PATH = os.path.join(HERE, "runs", "hard_review.json")
MODEL, API_KEY = B.MODEL, B.API_KEY

TYPES = ("nojargon", "vague", "meta", "short", "greek")
SHORT_MAX_WORDS = 7
# Κύρια ονόματα του σώματος (συστήματα, papers, εταιρείες, συνέδρια). Μια παράφραση που
# τα κρατά ΔΕΝ πετιέται — σημαδεύεται, γιατί το όνομα είναι ισχυρό λεξικό σήμα για BM25
# και reranker, άρα αναμένεται ευκολότερη. Τα αποτελέσματα αναφέρονται ΧΩΡΙΣΤΑ.
ENTITY_NAMES = {"excamera", "pywren", "mapreduce", "openwhisk", "berkeley", "cidr",
                "baldini", "hellerstein", "lambda", "s3", "aws", "amazon", "google",
                "ibm", "ec2", "mu", "hadoop", "knative", "numpywren", "cirrus", "sqlite",
                "windows", "linpack", "vpxenc", "firestore", "dynamodb"}
GEN_BATCH = 10          # γονικές ανά κλήση γέννησης
ID_OFFSET = 100         # p001 -> h101: ξεκάθαρα ξεχωριστά από τα h001-h016

GEN_PROMPT = """You are building a STRESS TEST for a question-answering system over this \
collection of research papers:

{overview}

Real users rarely phrase questions the way the papers do. For each ORIGINAL question below, \
write one paraphrase for EACH requested type. A paraphrase must ask for EXACTLY the same \
information: the given reference answer must remain a correct and complete answer to it.

Types:
- "nojargon": the same question in everyday words, without the technical terms and names \
the papers use. Example: "How fast is each of the different kinds of storage they compared?"
- "vague": the loose wording of someone who only half-remembers the topic - but still \
pointing to the same part of the papers. Example: "Why does the same thing end up costing \
different amounts in different places?"
- "meta": asks about the papers or their authors instead of the topic, describing the paper \
by its subject instead of its name. Example: "What do the authors of the paper about \
encoding video with thousands of tiny cloud functions say they contributed?"
- "short": at most 6 words, telegraphic, like typing into a search box - but it must still \
say what the topic is. Example: "Τι παράδειγμα δίνουν για κινητά;"
- "greek": natural, colloquial Greek, with everyday Greek words instead of the English \
technical terms. Example: "Γιατί συμφέρει να μην αγοράζεις μηχανήματα από πριν;"

Rules:
- The reference answer is given ONLY so that you keep the same meaning. NEVER put anything \
from it into the paraphrase that the original question does not already say: a real user \
does not know the answer, and a hint makes the question easier, not harder.
- Do NOT use the words listed under "avoid" for that question, in any form or inflection, \
and do not name systems, papers or products - describe them instead.
- The paraphrase must be understandable ON ITS OWN, as the first message of a conversation: \
no "that system", "the other paper", "it", "this one" pointing to something that the \
question itself does not name or describe. Referring to "the authors" or "the papers" is fine.
- It must still point to ONE specific thing in the papers: if a description could fit two \
different systems or passages, add a detail that tells them apart.
- Keep the language of the original question, except for type "greek" (always Greek).
- One sentence, or two very short ones.
- "question_en": the English version of your paraphrase (identical if it is already English).

ORIGINAL QUESTIONS:
{items}

Return ONLY a JSON array, one object per requested paraphrase, no prose:
[{{"parent": "q012", "type": "...", "question": "...", "question_en": "..."}}]
"""

VERIFY_PROMPT = """Below is the COMPLETE text of a collection of research papers, page by \
page. After it there is a list of test items. Each item has an ORIGINAL question, its \
REFERENCE ANSWER (known to be correct), and a PARAPHRASE written by someone else.

For EACH item judge the PARAPHRASE:
- "same_need": "yes" if the REFERENCE ANSWER is a correct and complete answer to the \
PARAPHRASE - a user who asked the paraphrase would be fully served by it. "no" if the \
paraphrase asks for more, for less, or for something else.
- "unique": "yes" if a careful reader of these papers, seeing ONLY the paraphrase, would \
look in the same place as the reference answer. "no" if the paraphrase fits another part of \
the papers just as well (name that other part in "reason").
- "self_contained": "yes" if it can be understood as the FIRST message of a conversation. \
"no" if it points to something ("that system", "the other paper", "it") that the question \
itself does not identify. Referring to "the authors" or "the papers" is fine.
- "no_hint": "yes" if the paraphrase gives away NOTHING of the reference answer beyond what \
the ORIGINAL question already says. "no" if it contains facts from the answer (for example \
listing the parts the answer is made of), which makes it easier than the original.
- "reason": one short sentence; required whenever any answer is "no".

=== PAPERS ===
{corpus}
=== END OF PAPERS ===

ITEMS:
{items}

Return ONLY a JSON array, one object per item, in the same order:
[{{"id": "...", "same_need": "yes|no", "unique": "yes|no", "self_contained": "yes|no", \
"no_hint": "yes|no", "reason": "..."}}]
"""


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_greek(text: str) -> bool:
    return bool(re.search(r"[α-ωΑ-Ωάέήίόύώ]", text))


def load_parents() -> list[dict]:
    """Οι in-corpus ερωτήσεις ΧΩΡΙΣ παράφραση στο παλιό hard set, με σταθερή σειρά."""
    used = {r["parent"] for r in load_jsonl(HARD_OLD)}
    pool = [r for r in load_jsonl(GOLDEN_50) if r.get("category") != "out_of_corpus"]
    pool += load_jsonl(MULTIHOP)
    return [r for r in pool if r["id"] not in used]


def assign_types(parents: list[dict]) -> list[tuple[dict, list[str]]]:
    """2 διαφορετικοί τύποι ανά γονική, κυκλικά -> ισορροπία τύπων.
    Το short παραλείπεται στα multi_hop (δύο έγγραφα δεν χωράνε σε 6 λέξεις)."""
    out, i = [], 0
    for p in parents:
        chosen: list[str] = []
        while len(chosen) < 2:
            t = TYPES[i % len(TYPES)]
            i += 1
            if t in chosen or (t == "short" and p["category"] == "multi_hop"):
                continue
            chosen.append(t)
        out.append((p, chosen))
    return out


def keyword_hit(question: str, keywords: list[str]) -> str:
    """Το πρώτο keyword της γονικής που εμφανίζεται στην παράφραση ('' αν κανένα).
    Ταίριασμα ανά token με πρόθεμα (ανεκτικό σε κλίσεις: straggler/stragglers).
    Πιάνει ΟΛΟ το keyword, ή ΕΝΑ μακρύ token του (≥7 χαρ.): το «undershoot» μόνο του
    αρκεί για να βρει το BM25 τη σελίδα του «undershoot-pct», το «start» του
    «cold start» όχι."""
    q_tokens = B.el_tokenize(question)

    def present(kt: str) -> bool:
        return any(qt.startswith(kt[:max(4, len(kt) - 2)]) for qt in q_tokens)

    for kw in keywords:
        kw_tokens = [t for t in B.el_tokenize(kw) if len(t) >= 3]
        if kw_tokens and (all(present(kt) for kt in kw_tokens)
                          or any(len(kt) >= 7 and present(kt) for kt in kw_tokens)):
            return kw
    return ""


def load_review() -> dict:
    """drop / notes ανά pid · keywords ανά ΓΟΝΙΚΗ (σφίξιμο: ίδια σελίδα-στόχος,
    σπανιότεροι όροι της απάντησης — ισχύει μόνο εδώ, το golden_set_50 μένει ως έχει)."""
    if not os.path.exists(REVIEW_PATH):
        return {"drop": {}, "notes": {}, "keywords": {}, "post_hoc": {}}
    with open(REVIEW_PATH, encoding="utf-8") as f:
        r = json.load(f)

    def clean(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}

    return {"drop": r.get("drop", {}), "notes": r.get("notes", {}),
            "keywords": clean(r.get("keywords", {})), "post_hoc": clean(r.get("post_hoc", {}))}


async def generate(assigned: list[tuple[dict, list[str]]], overview: str) -> list[dict]:
    cands = []
    for k in range(0, len(assigned), GEN_BATCH):
        chunk = assigned[k:k + GEN_BATCH]
        items = "\n".join(
            f"- parent: {p['id']} ({p['category']})\n"
            f"  question: {p['question']}\n"
            f"  reference answer: {p['reference_answer']}\n"
            f"  avoid: {', '.join(p['keywords'])}\n"
            f"  types: {', '.join(types)}"
            for p, types in chunk)
        raw = await gemini_rest.generate_once(
            GEN_PROMPT.format(overview=overview, items=items), model=MODEL, api_key=API_KEY,
            temperature=0.7, max_output_tokens=8192, thinking_budget=512)
        got = B.parse_json_array(raw)
        print(f"  γέννηση {k + len(chunk)}/{len(assigned)} γονικές: {len(got)} παραφράσεις",
              flush=True)
        cands.extend(got)
    return cands


async def verify(cands: list[dict], parents: dict, corpus: str, batch: int,
                 delay: float) -> dict:
    verdicts = {}
    for k in range(0, len(cands), batch):
        chunk = cands[k:k + batch]
        items = "\n".join(
            f"{c['pid']}:\n  ORIGINAL: {parents[c['parent']]['question']}\n"
            f"  REFERENCE ANSWER: {parents[c['parent']]['reference_answer']}\n"
            f"  PARAPHRASE: {c['question_en']}"
            for c in chunk)
        raw = await gemini_rest.generate_once(
            VERIFY_PROMPT.format(corpus=corpus, items=items), model=MODEL, api_key=API_KEY,
            temperature=0.0, max_output_tokens=16384, thinking_budget=1024)
        for v in B.parse_json_array(raw):
            verdicts[str(v.get("id", "")).strip()] = v
        print(f"  επαλήθευση {k + len(chunk)}/{len(cands)}", flush=True)
        if k + batch < len(cands):
            await asyncio.sleep(delay)   # περιθώριο για το per-minute όριο tokens
    return verdicts


def decide(c: dict, parent: dict, v: dict | None, seen: set) -> tuple[str, str]:
    """(απόφαση, λόγος). Στην αμφιβολία ΠΕΤΙΕΤΑΙ: μια παράφραση που άλλαξε νόημα θα
    μετρούσε ψεύτικη «λάθος κοπή» του συστήματος."""
    hit = keyword_hit(c["question"], parent["keywords"]) or \
        keyword_hit(c["question_en"], parent["keywords"])
    if hit:
        return "drop", f"περιέχει keyword «{hit}»"
    if c["type"] == "short" and len(c["question"].split()) > SHORT_MAX_WORDS:
        return "drop", f"short με {len(c['question'].split())} λέξεις"
    if c["type"] == "greek" and not is_greek(c["question"]):
        return "drop", "greek χωρίς ελληνικά"
    key = B.squash(c["question"])
    if key in seen:
        return "drop", "διπλότυπο"
    seen.add(key)
    if v is None:
        return "drop", "λείπει ετυμηγορία"
    failed = [f for f in ("same_need", "unique", "self_contained", "no_hint")
              if str(v.get(f, "")).strip().lower() != "yes"]
    if failed:
        return "drop", f"{'/'.join(failed)}: {v.get('reason', '')}"
    return "keep", ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", type=int, default=0, help="μόνο οι πρώτες N γονικές (δοκιμή)")
    ap.add_argument("--batch", type=int, default=12, help="υποψήφιες ανά κλήση επαλήθευσης")
    ap.add_argument("--delay", type=float, default=30.0, help="s ανάμεσα στις επαληθεύσεις")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--reuse-candidates", action="store_true",
                      help="ξανά επαλήθευση των σωσμένων υποψηφίων, χωρίς γέννηση")
    mode.add_argument("--apply-review", action="store_true",
                      help="ΜΗΔΕΝ κλήσεις: μόνο ο ανθρώπινος έλεγχος -> jsonl")
    args = ap.parse_args()
    if not API_KEY and not args.apply_review:
        print("!! Λείπει GEMINI_API_KEY στο περιβάλλον — σταματάω")
        return 1

    parents_list = load_parents()
    if args.parents:
        parents_list = parents_list[:args.parents]
    parents = {p["id"]: p for p in parents_list}
    suffix = "_smoke" if args.parents else ""
    cand_path = os.path.join(HERE, "runs", f"hard_candidates{suffix}.json")
    out_path = os.path.join(HERE, f"golden_hard_new{suffix}.jsonl")
    print(f"Γονικές χωρίς παράφραση: {len(parents_list)} "
          f"({dict(Counter(p['category'] for p in parents_list))})")

    papers = B.load_papers()
    if len(papers) != 7:
        print(f"!! Περίμενα 7 PDF στο {B.PAPERS_DIR}, βρήκα {len(papers)} — σταματάω")
        return 1
    corpus = "\n\n".join(f"[{p['file']} p.{i + 1}]\n{t}"
                         for p in papers for i, t in enumerate(p["pages"]))

    if args.reuse_candidates or args.apply_review:
        if not os.path.exists(cand_path):
            print(f"!! Δεν υπάρχει {cand_path} — τρέξε πρώτα χωρίς σημαία")
            return 1
        with open(cand_path, encoding="utf-8") as f:
            cands = json.load(f)
        print(f"Φορτώθηκαν {len(cands)} υποψήφιες από {cand_path}")
    else:
        overview = "\n".join(f"- {p['title']} ({p['file']})" for p in papers)
        assigned = assign_types(parents_list)
        print(f"Τύποι: {dict(Counter(t for _p, ts in assigned for t in ts))}")
        print(f"\nΓέννηση ({len(assigned)} γονικές × 2):")
        raw = asyncio.run(generate(assigned, overview))
        wanted = {(p["id"], t) for p, ts in assigned for t in ts}
        cands, got = [], set()
        for it in raw:
            key = (str(it.get("parent", "")).strip(), str(it.get("type", "")).strip())
            ok = (key in wanted and key not in got
                  and all(str(it.get(f) or "").strip() for f in ("question", "question_en")))
            if ok:
                got.add(key)
                cands.append({"pid": f"p{len(cands) + 1:03d}", "parent": key[0],
                              "type": key[1], "question": it["question"].strip(),
                              "question_en": it["question_en"].strip()})
        print(f"  έγκυρη μορφή: {len(cands)}/{len(raw)} (ζητήθηκαν {len(wanted)})")
        os.makedirs(os.path.dirname(cand_path), exist_ok=True)
        with open(cand_path, "w", encoding="utf-8") as f:
            json.dump(cands, f, ensure_ascii=False, indent=1)

    if not args.apply_review:
        print(f"\nΕπαλήθευση {len(cands)} με ΟΛΟΚΛΗΡΟ το σώμα "
              f"(≈{len(corpus) / 4.53:,.0f} tokens ανά κλήση, παρτίδες των {args.batch}):")
        verdicts = asyncio.run(verify(cands, parents, corpus, args.batch, args.delay))
        seen: set = set()
        for c in cands:
            c["decision"], c["reason"] = decide(c, parents[c["parent"]],
                                                verdicts.get(c["pid"]), seen)
        with open(cand_path, "w", encoding="utf-8") as f:
            json.dump(cands, f, ensure_ascii=False, indent=1)

    # --- Ανθρώπινος έλεγχος: μόνο ΠΕΤΑΕΙ, και μόνο για εγκυρότητα ---
    review = load_review()
    unknown = (set(review["drop"]) | set(review["notes"])
               | set(review["post_hoc"])) - {c["pid"] for c in cands}
    unknown |= set(review["keywords"]) - set(parents)
    if unknown:
        print(f"!! Ο έλεγχος αναφέρει άγνωστα pid/γονικές: {sorted(unknown)} — σταματάω")
        return 1
    rows = []
    for c in cands:
        if c.get("decision") != "keep" or c["pid"] in review["drop"]:
            continue
        p = parents[c["parent"]]
        row = {"id": f"h{ID_OFFSET + int(c['pid'][1:])}", "question": c["question"],
               "keywords": p["keywords"], "reference_answer": p["reference_answer"],
               "category": p["category"], "parent": p["id"], "hard_type": c["type"],
               "lang": "el" if is_greek(c["question"]) else "en",
               "question_en": c["question_en"], "round": 2,
               "names_entity": sorted(set(B.el_tokenize(c["question"])) & ENTITY_NAMES)}
        if c["pid"] in review["notes"]:
            row["review_note"] = review["notes"][c["pid"]]
        if c["pid"] in review["post_hoc"]:      # σφάλμα ΣΕΤ, βρέθηκε ΜΕΤΑ το τρέξιμο
            row["set_error"] = review["post_hoc"][c["pid"]]
        if p["id"] in review["keywords"]:
            row["keywords"] = review["keywords"][p["id"]]
            row["keywords_parent"] = p["keywords"]
        rows.append(row)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- Σύνοψη ---
    keep = [c for c in cands if c.get("decision") == "keep"]
    print("\n" + "#" * 78)
    print(f"Υποψήφιες {len(cands)} · κράτησε η μηχανή {len(keep)} "
          f"({len(keep) / max(1, len(cands)):.0%}) · μετά τον άνθρωπο {len(rows)}")
    print(f"{'τύπος':<10}{'γέννηση':>9}{'μηχανή':>8}{'άνθρωπος':>10}")
    for t in TYPES:
        print(f"{t:<10}{sum(c['type'] == t for c in cands):>9}"
              f"{sum(c['type'] == t for c in keep):>8}{sum(r['hard_type'] == t for r in rows):>10}")
    print(f"ανά κατηγορία: {dict(Counter(r['category'] for r in rows))}")
    print(f"ανά γλώσσα: {dict(Counter(r['lang'] for r in rows))}")
    print(f"γονικές με ≥1 παράφραση: {len({r['parent'] for r in rows})}/{len(parents_list)}")
    named = [r for r in rows if r["names_entity"]]
    print(f"ονομάζουν σύστημα/paper: {len(named)}/{len(rows)} "
          f"(ανά τύπο {dict(Counter(r['hard_type'] for r in named))})")
    reasons = Counter(c.get("reason", "").split(":")[0] for c in cands
                      if c.get("decision") == "drop")
    print(f"λόγοι απόρριψης (μηχανή): {dict(reasons.most_common())}")
    print("ΠΡΟΒΛΕΨΗ (γραμμένη πριν): μηχανή 45-60/80 · απώλειες κυρίως short/meta · "
          "μετά τον άνθρωπο ~40.")

    print("\n--- ΚΡΑΤΗΜΕΝΕΣ (για τον ανθρώπινο έλεγχο) ---")
    for c in keep:
        mark = "  [ΠΕΤΑΧΤΗΚΕ]" if c["pid"] in review["drop"] else ""
        print(f"{c['pid']} {c['parent']:<5} {c['type']:<9} {c['question']}{mark}")
    print("\n--- ΑΠΟΡΡΙΦΘΗΚΑΝ ---")
    for c in cands:
        if c.get("decision") == "drop":
            print(f"{c['pid']} {c['parent']:<5} {c['type']:<9} {c['question'][:80]}"
                  f"\n      -> {c['reason'][:150]}")
    print(f"\nΓράφτηκαν: {cand_path}\n           {out_path}")
    print("ΕΠΟΜΕΝΟ: ανθρώπινος έλεγχος στο runs/hard_review.json -> --apply-review.\n"
          "Μετά: πάτωμα τύχης των keywords (random_coverage_baseline) — ΠΡΙΝ τρέξει το σύστημα.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

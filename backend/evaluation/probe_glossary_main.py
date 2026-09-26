"""Ποιο κομμάτι της αλλαγής κόστισε; — glossary vs ΔΙΑΤΥΠΩΣΗ, στο ΚΥΡΙΟ σετ.

ΤΟ ΠΡΟΒΛΗΜΑ ΠΟΥ ΛΥΝΕΙ (η παγίδα του r_control.csv, ξανά):
Το baseline `runs/retrieval_l12.csv` είναι της 10/8, δηλαδή ΠΡΙΝ μπει το
`corpus_descriptor.json` (bf9aaf5, 15/8). Τότε το `_CORPUS_TERMS` ήταν ΚΕΝΟ και η
πρόταση glossary ΔΕΝ έμπαινε καν στο prompt. Άρα η σύγκριση «98.52% -> 97.04%»
μετράει ΤΡΕΙΣ αλλαγές μαζί: (α) 20 όρους glossary, (β) νέο domain string,
(γ) την περιοριστική διατύπωση. Δεν αποδίδεται σε καμία τους.

ΤΙ ΚΑΝΕΙ: τρέχει το golden_set_50 με ΜΟΝΗ μεταβλητή την πρόταση glossary, στο
ΠΑΡΑΓΩΓΙΚΟ collection, με τις μετρικές να έρχονται αυτούσιες από το eval_engine
(calculate_mrr/calculate_ndcg + ο ίδιος ορισμός coverage) ώστε να είναι απευθείας
συγκρίσιμο με τα υπάρχοντα artifacts.

  C_no_glossary  fallback domain + ΚΕΝΟΙ όροι   -> ΠΡΕΠΕΙ να αναπαράγει 98.52%
                                                   (CONTROL του ίδιου του probe)
  A_prod_instr   descriptor + «Prefer the vocabulary...»  <- η κατάσταση 15/8-χθες
  B_constrained  descriptor + περιοριστική οδηγία         <- η σημερινή

ΝΤΕΤΕΡΜΙΝΙΣΜΟΣ: ENABLE_CORRECTIVE=False. Ο agent γράφεται από το Gemini χωρίς
seed· στο κύριο σετ ενεργοποιείται μόνο στις 5 out_of_corpus, αλλά μια διαρροή
του θα χρεωνόταν λανθασμένα στο glossary.

ΚΟΣΤΟΣ: μόνο μεταφράσεις (20 ελληνικές ερωτήσεις x συνθήκες). ΜΗΔΕΝ γέννηση.
Το παραγωγικό translation cache ΔΕΝ γράφεται (_save_translation_cache -> no-op).
"""

import argparse
import asyncio
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_engine import GOLDEN_CORPUS, calculate_mrr, calculate_ndcg

import ai_core
import gemini_rest

# --- Οι τρεις διατυπώσεις. ΜΟΝΗ μεταβλητή του πειράματος. ---
G_PROD = "Prefer the vocabulary of this corpus glossary: {terms}. "

G_CONSTRAINED = (
    "Corpus glossary: {terms}. Use a glossary term ONLY as the translation of a "
    "word that is actually present in the question. NEVER add a glossary term "
    "that the question does not mention, and NEVER map a proper noun (a place, "
    "person, product or brand name) onto glossary vocabulary. "
)

# F: το B ΔΕΝ πιάνει τον μετρημένο μηχανισμό. Το q047 δεν ΠΡΟΣΘΕΣΕ όρο —
# ΑΝΤΙΚΑΤΕΣΤΗΣΕ λέξη που υπήρχε («επίπεδο» -> «serverless computing» αντί για
# «layer»). Το B απαγορεύει την προσθήκη, όχι τη ΓΕΝΙΚΕΥΣΗ.
G_STRICT = (
    "Corpus glossary (spelling reference ONLY): {terms}. Use a glossary term ONLY "
    "when it is the direct translation of a word actually present in the question. "
    "NEVER add a glossary term that the question does not mention, NEVER replace a "
    "word of the question with a broader or different glossary term, and NEVER map "
    "a proper noun (a place, person, product or brand name) onto glossary "
    "vocabulary. "
)

# --- Κατάσκοπος reranker: το best logit του 1ου pass, χωρίς να αγγίξει τα σκορ ---
_probe = {"best": None}


def _install_spy():
    orig = ai_core.reranker.predict

    def spy(pairs, **kw):
        scores = orig(pairs, **kw)
        if _probe["best"] is None and len(scores):
            _probe["best"] = float(max(scores))
        return scores

    ai_core.reranker.predict = spy


async def _optimize_variant(query: str, glossary_tmpl: str) -> str:
    """ΠΙΣΤΟ αντίγραφο του optimize_query με παραμετροποιημένη ΜΟΝΟ την πρόταση
    glossary. Ό,τι άλλο (σειρά προτάσεων, κεφαλαία, στίξη) μένει αυτούσιο —
    αλλιώς το πείραμα μετράει τη δική μου παραφθορά."""
    if not ai_core._has_greek(query):
        return query
    if query in ai_core._translation_cache:
        return ai_core._translation_cache[query]

    terms = ai_core._CORPUS_TERMS
    glossary = glossary_tmpl.format(terms=terms) if terms else ""
    prompt = (
        "You are a translation assistant for an English-only academic search engine. "
        f"The corpus is about: {ai_core._CORPUS_DOMAIN}. "
        "Translate the user's question to English using the STANDARD TECHNICAL "
        "TERMINOLOGY of that field: map everyday words to the term a document would "
        "actually use, NOT the literal everyday translation. "
        + glossary
        + "Preserve every domain term already present in the question - never drop or "
        "generalize one. Output ONLY a concise English search query with the key "
        "terms. No quotes, no extra text.\n\n"
        f"User question: {query}"
    )
    english = (
        await gemini_rest.generate_once(
            prompt, model=ai_core.GEMINI_MODEL, api_key=ai_core.GEMINI_API_KEY
        )
    ).strip(" \"'\n")
    ai_core._translation_cache[query] = english
    return english


async def run_condition(name, domain, terms, tmpl, tests, rows, corpus):
    ai_core._CORPUS_DOMAIN = domain
    ai_core._CORPUS_TERMS = terms
    ai_core._translation_cache.clear()   # κλειδί = η ΕΡΩΤΗΣΗ, όχι το prompt
    # **kw: το search_documents περνάει πια domain=/terms= από το assemble().
    # Τα ΑΓΝΟΟΥΜΕ επίτηδες — τη μοναδική μεταβλητή του πειράματος την ορίζει η
    # συνθήκη μέσω των globals, όχι τα metadata των chunks.
    ai_core.optimize_query = lambda q, **kw: _optimize_variant(q, tmpl)

    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}", flush=True)
    covs, leaks = [], []

    for t in tests:
        ooc = t.get("category") == "out_of_corpus"
        tr = await ai_core.optimize_query(t["question"])   # δωρεάν: μπαίνει στο cache
        _probe["best"] = None
        retrieved = await ai_core.search_documents(
            t["question"], target_filenames=corpus
        )
        texts = [text for text, _meta in retrieved]

        kws = t.get("keywords") or []
        mrrs = [calculate_mrr(k, texts) for k in kws] if texts else []
        mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0
        ndcgs = [calculate_ndcg(k, texts, k=3) for k in kws] if texts else []
        ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        cov = (sum(1 for s in mrrs if s > 0) / len(kws) * 100) if kws else 0.0

        best = _probe["best"]
        if ooc:
            if texts:
                leaks.append(t["id"])
        else:
            covs.append(cov)

        rows.append({
            "condition": name, "id": t["id"], "category": t.get("category", ""),
            "mrr": f"{mrr:.4f}", "ndcg": f"{ndcg:.4f}",
            "keyword_coverage": f"{cov:.1f}",
            "pages": len(texts),
            "best_logit": f"{best:+.2f}" if best is not None else "",
            "silent": "" if not ooc else ("ΔΙΑΡΡΟΗ" if texts else "σιωπή"),
            "translation": tr,
        })
        mark = "ΔΙΑΡΡΟΗ" if (ooc and texts) else (f"{cov:5.1f}%" if not ooc else "σιωπή")
        bs = f"{best:+7.2f}" if best is not None else "      -"
        print(f"  {t['id']:<5} {mark:>8}  {bs}  {tr[:64]}", flush=True)

    mean_cov = sum(covs) / len(covs) if covs else 0.0
    n_ooc = sum(1 for t in tests if t.get("category") == "out_of_corpus")
    print(f"\n  >>> coverage in-corpus {mean_cov:.2f}%  ·  διαρροές {len(leaks)}/{n_ooc} "
          f"{leaks if leaks else ''}", flush=True)
    return mean_cov, leaks, n_ooc


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="evaluation/golden_set_50.jsonl")
    ap.add_argument("--csv", default="evaluation/runs/glossary_main.csv")
    ap.add_argument("--only", default="", help="π.χ. A,B — παράλειψη συνθηκών")
    ap.add_argument("--corpus", default="",
                    help="αρχεία χωρισμένα με κόμμα· κενό = GOLDEN_CORPUS (τα 7 cloud papers)")
    ap.add_argument("--terms-file", default="evaluation/runs/doc_terms.json",
                    help="JSON {αρχείο: όροι} από το probe_doc_terms.py — τροφοδοτεί το G")
    args = ap.parse_args()

    with open(args.golden, encoding="utf-8") as f:
        tests = [json.loads(ln) for ln in f if ln.strip()]

    corpus = [s.strip() for s in args.corpus.split(",") if s.strip()] or GOLDEN_CORPUS
    print(f"corpus: {corpus}")

    ai_core.ENABLE_CORRECTIVE = False        # ντετερμινισμός
    ai_core._save_translation_cache = lambda: None   # μη μολύνεις την παραγωγή
    _install_spy()

    prod_domain, prod_terms = ai_core._CORPUS_DOMAIN, ai_core._CORPUS_TERMS
    print(f"descriptor σε χρήση: domain='{prod_domain}' · όροι={len(prod_terms.split(','))}")

    # G: όροι ΑΝΑ ΑΡΧΕΙΟ, ενωμένοι ΜΟΝΟ για όσα αρχεία είναι στο scope. Αυτό είναι
    # ακριβώς το σχήμα του βήματος 3 (metadata ανά αρχείο -> glossary στο query time).
    doc_terms, doc_domain, rich_domain = "", "", ""
    if os.path.exists(args.terms_file):
        with open(args.terms_file, encoding="utf-8") as f:
            per_file = json.load(f)
        seen, merged, domains = set(), [], []
        for fn in corpus:
            entry = per_file.get(fn) or ""
            # Δύο σχήματα: {αρχείο: "όροι"} (παλιό) και {αρχείο: {domain, terms}}.
            dom = entry.get("domain", "") if isinstance(entry, dict) else ""
            trm = entry.get("terms", "") if isinstance(entry, dict) else entry
            if dom and dom not in domains:
                domains.append(dom)
            for t in trm.split(","):
                t = t.strip()
                if t and t.lower() not in seen:
                    seen.add(t.lower())
                    merged.append(t)
        doc_terms = ", ".join(merged)
        # Ένωση με «;»: κάθε αρχείο κρατάει τη δική του περιγραφή, ΧΩΡΙΣ να
        # συγχωνεύεται σε μια γενικότερη — αυτό ακριβώς θα κάνει το query time.
        doc_domain = "; ".join(domains)
        # Το _FALLBACK_DOMAIN ΔΕΝ είναι γενικό: «computer-science papers on cloud
        # computing, serverless computing and distributed systems» — ονομάζει ΕΙΔΟΣ
        # εγγράφου + υποπεδία, σε μία φράση. Τα per-file domains είναι σκέτες
        # ΕΤΙΚΕΤΕΣ («Distributed Computing»). Το rich_domain ξαναχτίζει το ΣΧΗΜΑ του
        # fallback από τα ίδια metadata, με ΓΕΝΙΚΟ είδος ώστε να μη χρειάζεται να
        # ξέρουμε το πεδίο εκ των προτέρων -> γενικεύεται σε οποιοδήποτε corpus.
        if domains:
            rich_domain = "scientific papers on " + ", ".join(d.lower() for d in domains)
        print(f"όροι ανά αρχείο: {len(merged)} από {len(corpus)} αρχεία")
        if doc_domain:
            print(f"domains ανά αρχείο: {len(domains)} -> {doc_domain[:160]}")
            print(f"rich domain: {rich_domain[:200]}")

    conds = [
        ("C_no_glossary", ai_core._FALLBACK_DOMAIN, "", G_PROD),
        ("A_prod_instr", prod_domain, prod_terms, G_PROD),
        ("B_constrained", prod_domain, prod_terms, G_CONSTRAINED),
        # D: ο descriptor άλλαξε ΔΥΟ πράγματα ταυτόχρονα (domain string + 20 όρους).
        # Το D κρατάει το νέο domain με ΚΕΝΟΥΣ όρους -> αν βγει 98.52% φταίνε οι όροι,
        # αν βγει 97.04% φταίει η συρρίκνωση του domain string.
        ("D_domain_only", prod_domain, "", G_PROD),
        # E: το αντίστροφο — παλιό περιγραφικό domain + οι 20 όροι.
        ("E_terms_only", ai_core._FALLBACK_DOMAIN, prod_terms, G_CONSTRAINED),
        # F: το B + ρητή απαγόρευση ΑΝΤΙΚΑΤΑΣΤΑΣΗΣ (όχι μόνο προσθήκης).
        # ΚΡΙΤΗΡΙΑ ΓΡΑΜΜΕΝΑ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ: κύριο >= 97.78% ΚΑΙ test_domains
        # >= 95.0% με ΟΛΕΣ τις out_of_corpus σιωπηλές. Αλλιώς κρατάμε το B.
        ("F_strict", prod_domain, prod_terms, G_STRICT),
        # G: ΙΔΙΑ οδηγία με το B — μόνη μεταβλητή το ΠΕΡΙΕΧΟΜΕΝΟ των όρων.
        # A/B/F έδωσαν ΤΑΥΤΟΣΗΜΟ 97.04% με 3 διαφορετικές οδηγίες, άρα η διατύπωση
        # δεν είναι μοχλός. Αν το G επιστρέψει στο 98.52%, φταίει το ΕΙΔΟΣ των όρων
        # (γενικές ετικέτες πεδίου) και το βήμα 3 λύνει το πρόβλημα από μόνο του.
        ("G_doc_terms", prod_domain, doc_terms, G_CONSTRAINED),
        # H: Η ΠΑΡΑΓΩΓΙΚΗ ΔΙΑΜΟΡΦΩΣΗ ΤΟΥ ΒΗΜΑΤΟΣ 3, ολόκληρη.
        # Το βήμα 3 ΚΑΤΑΡΓΕΙ το corpus_descriptor.json (χωρίς CLI δεν υπάρχει ποιος
        # θα το χτίσει), άρα το domain ΔΕΝ μπορεί να είναι το cloud-specific string
        # του G. Γίνεται ΓΕΝΙΚΟ — μετρημένο στα νέα πεδία (CG: 95.0%, ooc 5/5),
        # ΑΜΕΤΡΗΤΟ στο κύριο. Και το D έδειξε ότι το domain string ΜΟΝΟ ΤΟΥ κοστίζει
        # 0.74pp, άρα δεν είναι ουδέτερη μεταβλητή.
        # ΚΡΙΤΗΡΙΟ ΓΡΑΜΜΕΝΟ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ: >= 98.52% (ταυτόσημο με το G) ΚΑΙ
        # 5/5 σιωπηλές. Αν πέσει, το domain πρέπει να παραχθεί κι αυτό ανά αρχείο
        # στο ingest — όχι να καρφωθεί γενικό.
        ("H_generic_domain", "scientific research papers", doc_terms, G_CONSTRAINED),
        # I: ΤΟ ΠΡΑΓΜΑΤΙΚΟ ΣΧΗΜΑ ΤΗΣ ΠΑΡΑΓΩΓΗΣ. Ούτε CLI string (G), ούτε καρφωμένο
        # γενικό (H): ένα domain ΑΝΑ ΑΡΧΕΙΟ, γραμμένο στο ingest, ενωμένο στο query
        # time για ΟΣΑ αρχεία είναι στο scope. Το H απέτυχε (97.78%, το q047 έχασε
        # το «layer») -> το domain string δεν είναι ουδέτερη μεταβλητή.
        # ΚΡΙΤΗΡΙΟ ΓΡΑΜΜΕΝΟ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ: >= 98.52% ΚΑΙ 5/5 σιωπηλές. Αλλιώς
        # η σχεδίαση «χωρίς CLI» δεν στέκει και επιστρέφουμε στο τραπέζι.
        ("I_doc_domain", doc_domain, doc_terms, G_CONSTRAINED),
        # J: ΤΟ ΚΕΝΟ ΚΕΛΙ ΤΟΥ ΠΙΝΑΚΑ. Κάθε συνθήκη ΜΕ glossary κουβαλούσε terms·
        # κάθε συνθήκη ΧΩΡΙΣ terms κουβαλούσε ΚΑΘΟΛΙΚΟ domain. Το «σωστό domain ΑΝΑ
        # ΑΡΧΕΙΟ + ΜΗΔΕΝ terms» δεν μετρήθηκε ποτέ, σε κανένα από τα δύο σετ.
        # ΓΙΑΤΙ ΕΧΕΙ ΣΗΜΑΣΙΑ: τα terms είναι η ΜΟΝΑΔΙΚΗ επιφάνεια διαρροής (ο
        # Βεζούβιος/o4 μπήκε επειδή υπήρχε λεξιλόγιο να ενεθεί). Αν το domain μόνο
        # του κρατάει το coverage, τα terms φεύγουν και η διαρροή γίνεται αδύνατη
        # εξ ορισμού — όχι απλώς περιορισμένη από διατύπωση.
        # ΚΡΙΤΗΡΙΑ ΓΡΑΜΜΕΝΑ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ: κύριο >= 97.78% (όσο το I, που ΕΧΕΙ
        # terms) ΚΑΙ 5/5 σιωπηλές· cross-domain >= 95.0% (όσο η CG/CH) ΚΑΙ 5/5.
        # Αν πέσει στο cross-domain, τα terms κάνουν τη δουλειά και το backfill
        # προχωράει με terms όπως σχεδιάστηκε.
        ("J_doc_domain_only", doc_domain, "", G_PROD),
        # K: ΙΔΙΑ metadata με το J, ΑΛΛΟ ΣΧΗΜΑ — η φράση του fallback αντί για
        # σκέτες ετικέτες. Απομονώνει τη ΔΕΥΤΕΡΗ μεταβλητή που έκρυβε η σύγκριση
        # C vs D: «ανά αρχείο vs καθολικό» ΚΑΙ «περιγραφή vs ετικέτα» άλλαζαν μαζί.
        # Αν K > J, μοχλός είναι η διατύπωση· αν K == J, μόνο η προέλευση μετράει.
        ("K_rich_domain", rich_domain, "", G_PROD),
    ]
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        conds = [c for c in conds if c[0].split("_")[0] in keep]

    rows, summary = [], []
    for name, dom, terms, tmpl in conds:
        cov, leaks, n_ooc = await run_condition(
            name, dom, terms, tmpl, tests, rows, corpus)
        summary.append((name, cov, leaks, n_ooc))

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "condition", "id", "category", "mrr", "ndcg", "keyword_coverage",
            "pages", "best_logit", "silent", "translation"])
        w.writeheader()
        w.writerows(rows)

    n_in = sum(1 for t in tests if t.get("category") != "out_of_corpus")
    print(f"\n{'=' * 62}\nΣΥΓΚΕΝΤΡΩΤΙΚΑ (in-corpus n={n_in})\n{'=' * 62}")
    for name, cov, leaks, n_ooc in summary:
        flag = (f"ΔΙΑΡΡΟΕΣ: {','.join(leaks)}" if leaks
                else f"ooc {n_ooc}/{n_ooc} σιωπηλά")
        print(f"  {name:<16} coverage {cov:6.2f}%   {flag}")
    print("\n  αναφορά κύριου σετ 13/8 (πριν τον descriptor): 98.52%")
    print(f"\nCSV -> {args.csv}")


if __name__ == "__main__":
    asyncio.run(main())

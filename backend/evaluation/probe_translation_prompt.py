"""ΜΙΑ εκδοχή του prompt μετάφρασης, A/B σε ΟΛΕΣ τις ελληνικές ερωτήσεις (6 σετ, 2 store).

ΓΙΑΤΙ ΥΠΑΡΧΕΙ (28/9/2026):
    Από τις 16-18/8 (γλωσσάρι ανά αρχείο + περιοριστική οδηγία, χωρίς τα γραμμένα με
    το χέρι παραδείγματα) δύο ερωτήσεις με απάντηση μένουν ΠΑΝΤΑ αναπάντητες — 5/5
    ίδιες μεταφράσεις, και η δεύτερη ευκαιρία αποτυγχάνει:
      q025 «τιμές ηλεκτρικού ρεύματος» -> «cost per kilowatt-hour»: ο όρος «cost» του
           γλωσσαρίου ΚΑΤΑΠΙΝΕΙ το «electricity» (-2.93, ήταν +0.35 στις 10/8)
      q059 «εξάρτηση από το ecosystem ενός παρόχου» -> «provider ecosystem dependency»
           αντί για «vendor lock-in» (-4.43, ήταν +1.32· το σωστό chunk είναι 1ο)
    Ο φύλακας ΔΕΝ διορθώνει τίποτα: -4.43 είναι κάτω και από την καλύτερη άσχετη
    (-3.12). Η λύση είναι στη μετάφραση (runs/gate_margin_now.csv).

Η ΕΚΔΟΧΗ (NEW) — τρεις ΓΕΝΙΚΟΙ κανόνες, παραδείγματα από ΑΣΧΕΤΑ πεδία (όχι από τα
papers ή τα σετ μας — το λάθος του Αυγούστου ήταν ότι το prompt είχε μέσα τις
απαντήσεις των h004/h007):
    (1) κράτα ΚΑΘΕ έννοια της ερώτησης, μην πετάς / μη γενικεύεις     <- q025
    (2) καθημερινές λέξεις -> καθιερωμένος τεχνικός όρος, ΚΑΙ εκτός γλωσσαρίου  <- q059
    (3) ΠΟΤΕ πρόσθετη έννοια, ΠΟΤΕ κύριο όνομα σε άλλο λεξιλόγιο       <- άμυνα Βεζούβιου
    Το γλωσσάρι μένει, ως «ορθογραφική αναφορά».

ΜΕΘΟΔΟΣ: ίδιος κώδικας παραγωγής (search_documents, γλωσσάρι από τα in-scope αρχεία),
μόνη μεταβλητή το prompt. ΚΑΙ ΟΙ ΔΥΟ συνθήκες μεταφράζουν ΑΠΟ ΤΗΝ ΑΡΧΗ (κανένα cache)
-> συμμετρικό. Μόνο 1ο πέρασμα (ENABLE_CORRECTIVE=0): ντετερμινιστικό δεδομένης της
μετάφρασης. Δύο απομονωμένα store: cloud (418 chunks) και άλλα πεδία (καρδιολογία +
ηφαιστειολογία, ingest με τον κώδικα παραγωγής — ό,τι θα γινόταν αν τα ανέβαζε χρήστης).

ΚΡΙΤΗΡΙΟ, ΓΡΑΜΜΕΝΟ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (28/9/2026) — ΠΕΡΝΑΕΙ μόνο αν ισχύουν ΚΑΙ ΤΑ ΤΡΙΑ:
    (α) σχεδιασμός: q025 ΚΑΙ q059 περνάνε τον φύλακα με ≥1 λέξη-κλειδί
    (β) διαρροές: ΚΑΜΙΑ ερώτηση χωρίς απάντηση (κύριο σετ + άλλα πεδία) που κόβεται
        με CUR δεν περνάει με NEW
    (γ) επικύρωση (όλες οι in-corpus ΕΚΤΟΣ q025/q059): «με υλικό» NEW ≥ CUR − 1
    Οι κοντινές ooc (near_ooc) περνάνε τον φύλακα 40/42 ΟΥΤΩΣ Η ΑΛΛΩΣ — η άμυνά τους
    είναι το system prompt της απάντησης, που δεν αλλάζει. Εδώ μόνο περιγραφικά.
    ΜΙΑ προσπάθεια: αν αποτύχει, ΔΕΝ δοκιμάζεται δεύτερη εκδοχή — καταγράφεται.

ΠΡΟΒΛΕΨΗ (γραμμένη πριν): q025 ✓ · q059 ✓ · 0 νέες διαρροές · επικύρωση +0 έως +3.
    Ρίσκο: η o5 (στατίνες) — ο κανόνας (2) μπορεί να τη μεταφράσει «HMG-CoA reductase
    inhibitors», που διέρρευσε στο probe_corrective_domains (-1.87).

ΑΠΟΤΕΛΕΣΜΑ (28/9/2026) — ΔΕΝ ΠΕΡΝΑΕΙ. Κόπηκε στο (α), άρα το υπόλοιπο δεν αλλάζει την
ετυμηγορία. Το cloud τρέξιμο ολοκληρώθηκε· τα άλλα πεδία έσκασαν (βλ. κάτω) και ΔΕΝ
ξανατρέχουν — μία προσπάθεια, όπως ορίστηκε.
    (α) q059 ✓  CUR «Baldini et al. BaaS services provider ecosystem dependency» -4.43 κομμένη
                NEW «Baldini et al. warnings about vendor BaaS ecosystem dependency» -2.31, 2/2
        q025 ✗  NEW «What kilowatt-hour electricity prices does the paper report and why do
                they differ» -3.55 ΚΟΜΜΕΝΗ — ΧΑΜΗΛΟΤΕΡΑ από το σημερινό (-2.93).
                Η ΔΙΑΓΝΩΣΗ ΜΟΥ ΗΤΑΝ ΛΑΘΟΣ: το «electricity» ΞΑΝΑΜΠΗΚΕ και δεν βοήθησε. Δεν
                φταίει η χαμένη λέξη· ο reranker βαθμολογεί χαμηλά αυτή τη ΜΟΡΦΗ ερώτησης.
                Ειρωνεία: η «σπασμένη» μετάφραση «cost per» περνάει με +0.73 και 3/3.
    (β) κύριο σετ: q020 / q049 κομμένες και στις δύο -> 0 διαρροές. Άλλα πεδία: ΔΕΝ μετρήθηκαν.
    (γ) cloud, n=50: με υλικό CUR 44 -> NEW 43 (εντός ορίου −1)
        χάθηκαν h145 (-2.29 3/3 -> -7.87 κομμένη) · h126 (1/3 -> 0/3)
        κέρδος  h132 (-3.17 κομμένη -> -0.34 3/3)
        λιγότερες λέξεις με υλικό: h121 3->1 · h122 2->1 · q045 3->1
    near_ooc (περιγραφικά): περνάνε τον φύλακα 19/21 -> 20/21 (n078 -4.66 -> -1.96)
    ΠΑΡΑΤΗΡΗΣΗ ΥΠΕΡ ΤΟΥ NEW, χωρίς βάρος στην ετυμηγορία: h007 CUR «Why serverless is
        cost-effective» — ο σημερινός μεταφραστής ΠΡΟΣΘΕΤΕΙ έννοια που δεν ρωτήθηκε και περνάει
        +6.79 με 0/3 (σίγουρο λάθος υλικό)· NEW κρατάει την ερώτηση και κόβεται (-4.82).
    Πρόβλεψη: q025 ✓ ✗ · q059 ✓ ✓ · 0 διαρροές ✓ (μόνο κύριο) · επικύρωση +0..+3 ✗ (−1).

ΠΑΡΑΠΛΕΥΡΑ ΕΥΡΗΜΑΤΑ ΤΟΥ ΤΡΕΞΙΜΑΤΟΣ:
    • CUR q025 = «cost per» ΞΑΝΑ, ενώ το generate_once ελέγχει πλέον finishReason. Ήρθε με
      STOP -> ΔΕΝ είναι κομμένη απάντηση· είναι σπάνια, ολοκληρωμένη, κακή μετάφραση. Η
      διάγνωση της 25/9 («μισή απάντηση») ήταν λάθος. Ο έλεγχος finishReason μένει (σωστή
      προστασία για MAX_TOKENS/SAFETY/κομμένη ροή), αλλά ΔΕΝ καλύπτει αυτή την περίπτωση.
      Το «cost per» ΔΕΝ είναι όρος του γλωσσαρίου (υπάρχει μόνο «cost», EECS-2009-28).
    • Το store άλλων πεδίων βρέθηκε ΔΙΠΛΟ (200 chunks = κάθε PDF δύο φορές) και το
      _dense_exact_ids έσκασε (IndexError: πίνακας 163 γραμμών, δείκτης 163). ΑΙΤΙΑ: το
      script έτρεξε ΔΥΟ ΦΟΡΕΣ ΤΑΥΤΟΧΡΟΝΑ (δύο τερματικά, 09:28) και τα δύο έχτισαν το ίδιο
      store — ΟΧΙ «απομεινάρι παλιότερης απόπειρας», όπως γράφτηκε πρώτα. Το cloud store
      μόνο διαβαζόταν, άρα τα νούμερα του cloud ισχύουν. Το «χτίζεται πάντα από την αρχή»
      μένει, αλλά δεν προστατεύει από ταυτόχρονα τρεξίματα.
    • Το δεύτερο τρέξιμο = αθέλητη ΕΠΑΝΑΛΗΨΗ: q025 NEW -3.55 και q059 NEW -2.31 ΤΑΥΤΟΣΗΜΑ
      -> η ετυμηγορία στέκει. Το (γ) ΓΥΡΙΣΕ (43 -> 44· το CUR μετέφρασε αλλιώς το h145) και
      στις near_ooc το n078 NEW κόπηκε (-3.34) -> το ±1 της επικύρωσης είναι θόρυβος.

ΚΟΣΤΟΣ: 100 ελληνικές × 2 = ~200 σύντομες κλήσεις Gemini. Χρόνος ~8-10 λεπτά
(+ ~2 λεπτά για το store των άλλων πεδίων).

    docker compose exec backend python evaluation/probe_translation_prompt.py
    docker compose exec backend python evaluation/probe_translation_prompt.py --limit 1   # δοκιμή
"""
import argparse
import asyncio
import csv
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_near_ooc as E

import gemini_rest

CSV_PATH = os.path.join(E.HERE, "runs", "translation_prompt_ab.csv")
TD_DB = "/tmp/td_prod_chroma"  # noqa: S108  εφήμερο, μέσα στο container
TD_PAPERS = [os.path.join(E.HERE, "test_papers", f)
             for f in ("cureus-0015-00000046486.pdf", "s41598-017-03833-3.pdf")]
DESIGN = {"q025", "q059"}
CLOUD_SETS = [("κύριο", "golden_set_50.jsonl"), ("multihop", "golden_multihop_new.jsonl"),
              ("hard_new", "golden_hard_new.jsonl"), ("hard_old", "golden_hard_paraphrase.jsonl"),
              ("near_ooc", "golden_near_ooc.jsonl")]
TD_SETS = [("άλλα_πεδία", "golden_test_domains.jsonl")]
FIELDS = ["set", "id", "category", "cond", "translation", "best1", "passed", "kw", "n_kw"]

# --- Η ΕΚΔΟΧΗ. Αν περάσει, ΑΥΤΟ το κείμενο μπαίνει αυτούσιο στο ai_core.optimize_query.
NEW_HEAD = (
    "You are a translation assistant for an English-only academic search engine. "
    "The corpus is about: {domain}. Translate the user's question to English following "
    "these rules. "
    "(1) Keep EVERY concept of the question - the thing asked about and each of its "
    "qualifiers; never drop or generalize one (e.g. never reduce 'apartment rental "
    "prices' to 'costs'). "
    "(2) Where the question uses everyday words for a concept that the field has an "
    "established technical term for, use that term (e.g. 'a general rise in prices' -> "
    "'inflation'), even if the term is not in the glossary. "
    "(3) NEVER add a concept that the question does not express, and NEVER map a proper "
    "noun (a place, person, product or brand name) onto other vocabulary. "
)
NEW_GLOSSARY = "Corpus glossary (spelling reference for terms this corpus uses): {terms}. "
NEW_TAIL = "Output ONLY a concise English search query. No quotes, no extra text.\n\nUser question: "


def is_greek(q: str) -> bool:
    return bool(re.search("[α-ωΑ-Ω]", q))


def covered(pages, kws) -> int:
    blob = "\n".join(t for t, _m in pages).lower()
    return sum(1 for k in kws if k.lower() in blob)


def load(sets, limit):
    out = []
    for name, fn in sets:
        rows = [r for r in E.load_jsonl(os.path.join(E.HERE, fn)) if is_greek(r["question"])]
        out += [(name, r) for r in rows[:limit or None]]
    return out


def make_new_optimize(ai_core):
    async def new_optimize(query, domain=None, terms=None):
        if not ai_core._has_greek(query):
            return query
        if query in ai_core._translation_cache:
            return ai_core._translation_cache[query]
        domain = domain or ai_core._CORPUS_DOMAIN
        terms = terms or ai_core._CORPUS_TERMS
        prompt = (NEW_HEAD.replace("{domain}", domain)
                  + (NEW_GLOSSARY.replace("{terms}", terms) if terms else "")
                  + NEW_TAIL + query)
        try:
            out = (await gemini_rest.generate_once(
                prompt, model=ai_core.GEMINI_MODEL, api_key=ai_core.GEMINI_API_KEY)).strip(" \"'\n")
        except Exception as e:  # ίδια συμπεριφορά με την παραγωγή: αρχική ερώτηση, χωρίς cache
            print(f"   !! μετάφραση απέτυχε: {e}")
            return query
        ai_core._translation_cache[query] = out
        return out
    return new_optimize


def open_td(ai_core):
    # ΠΑΝΤΑ από την αρχή: στο 1ο τρέξιμο βρέθηκε ΔΙΠΛΟ (κάθε PDF δύο φορές) από
    # παλιότερη απόπειρα, και η αναζήτηση έσκασε. ~2 λεπτά, κανένα κόστος API.
    import chromadb
    if os.path.exists(TD_DB):
        shutil.rmtree(TD_DB)
    col = chromadb.PersistentClient(path=TD_DB).get_or_create_collection(
        name="td_prod", embedding_function=ai_core.sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"})
    ai_core.collection = col
    print("Χτίσιμο store άλλων πεδίων (~2 λεπτά)...", flush=True)
    for i, p in enumerate(TD_PAPERS):
        ai_core.ingest_pdf(p, os.path.basename(p), user_id=E.TEST_USER, is_public=False,
                           doc_id=900 + i)
    ai_core._bump_corpus_version()
    print(f"Store άλλων πεδίων: {col.count()} chunks", flush=True)


async def run(ai_core, cond, fn, tests, rows):
    ai_core.optimize_query = fn
    for s, t in tests:
        q = t["question"]
        ai_core._translation_cache.pop(q, None)          # ΚΑΘΕ συνθήκη μεταφράζει από την αρχή
        pages, b1, _b2, _rw, out = await E.retrieve(ai_core, q)
        tr = ai_core._translation_cache.get(q, "")
        row = {"set": s, "id": t["id"], "category": t.get("category", ""), "cond": cond,
               "translation": tr, "best1": E.fmt(b1), "passed": int(out != "cut"),
               "kw": covered(pages, t["keywords"]), "n_kw": len(t["keywords"])}
        rows.append(row)
        print(f"  {cond} {s:<10} {t['id']:<5} {row['best1']:>6} "
              f"{'περνάει' if row['passed'] else 'ΚΟΒΕΤΑΙ':<8} λέξεις {row['kw']}/{row['n_kw']} · {tr[:70]}",
              flush=True)


def report(rows):
    by = {(r["cond"], r["id"]): r for r in rows}
    ids = [(r["set"], r["id"], r["category"]) for r in rows if r["cond"] == "CUR"]
    ooc = lambda c: c == "out_of_corpus"  # noqa: E731
    mat = lambda r: r["passed"] and r["kw"] > 0  # noqa: E731

    print("\n" + "#" * 80 + "\nΑΝΑ ΣΕΤ (CUR = σημερινό prompt · NEW = εκδοχή)")
    for s in dict.fromkeys(x[0] for x in ids):
        inc = [i for ss, i, c in ids if ss == s and not ooc(c)]
        oo = [i for ss, i, c in ids if ss == s and ooc(c)]
        line = f"  {s:<11} "
        if inc:
            line += (f"με απάντηση n={len(inc):<3} με υλικό CUR {sum(mat(by['CUR', i]) for i in inc):>2}"
                     f" -> NEW {sum(mat(by['NEW', i]) for i in inc):>2}   ")
        if oo:
            line += (f"χωρίς απάντηση n={len(oo):<3} περνάνε τον φύλακα CUR "
                     f"{sum(by['CUR', i]['passed'] for i in oo)} -> NEW {sum(by['NEW', i]['passed'] for i in oo)}")
        print(line)

    print("\n--- ό,τι άλλαξε (φύλακας ή υλικό) ---")
    for s, i, c in ids:
        a, b = by["CUR", i], by["NEW", i]
        if (a["passed"], mat(a)) != (b["passed"], mat(b)):
            tag = "ΣΧΕΔΙΑΣΜΟΣ" if i in DESIGN else ("χωρίς απάντηση" if ooc(c) else "")
            print(f"  {s:<10} {i:<5} {tag}\n     CUR {a['best1']:>6} λέξεις {a['kw']}/{a['n_kw']} · {a['translation']}"
                  f"\n     NEW {b['best1']:>6} λέξεις {b['kw']}/{b['n_kw']} · {b['translation']}")

    # --- ΚΡΙΤΗΡΙΟ (γραμμένο πριν) ---
    design_ok = all(i in [x[1] for x in ids] and mat(by["NEW", i]) for i in DESIGN)
    leaks = [i for s, i, c in ids if ooc(c) and s != "near_ooc"
             and not by["CUR", i]["passed"] and by["NEW", i]["passed"]]
    val = [i for s, i, c in ids if not ooc(c) and i not in DESIGN]
    cur_m, new_m = sum(mat(by["CUR", i]) for i in val), sum(mat(by["NEW", i]) for i in val)
    print("\n" + "#" * 80 + "\nΚΡΙΤΗΡΙΟ (γραμμένο πριν το τρέξιμο)")
    print(f"  (α) q025 και q059 με υλικό με NEW:         {'ΝΑΙ' if design_ok else 'ΟΧΙ'}")
    print(f"  (β) νέες διαρροές (κύριο + άλλα πεδία):    {leaks or 'καμία'}")
    print(f"  (γ) επικύρωση n={len(val)}: με υλικό CUR {cur_m} -> NEW {new_m}   "
          f"({'OK' if new_m >= cur_m - 1 else 'ΧΕΙΡΟΤΕΡΟ'})")
    ok = design_ok and not leaks and new_m >= cur_m - 1
    print(f"\n  ΑΠΟΤΕΛΕΣΜΑ: {'ΠΕΡΝΑΕΙ' if ok else 'ΔΕΝ ΠΕΡΝΑΕΙ'}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="μόνο οι πρώτες N ελληνικές ανά σετ (δοκιμή)")
    ap.add_argument("--csv", default=CSV_PATH)
    args = ap.parse_args()
    if not E.API_KEY:
        print("!! Λείπει GEMINI_API_KEY")
        return 1

    ai_core, _ = E.open_store()
    ai_core.ENABLE_CORRECTIVE = False
    cur, new = ai_core.optimize_query, make_new_optimize(ai_core)
    cloud, td = load(CLOUD_SETS, args.limit), load(TD_SETS, args.limit)
    if args.limit:                                   # η δοκιμή περιλαμβάνει ΠΑΝΤΑ τον σχεδιασμό
        mh = [(n, r) for n, fn in CLOUD_SETS[:2] for r in E.load_jsonl(os.path.join(E.HERE, fn))
              if r["id"] in DESIGN]
        cloud = [x for x in cloud if x[1]["id"] not in DESIGN] + mh
    print(f"Ελληνικές: cloud {len(cloud)} · άλλα πεδία {len(td)} -> ~{2 * (len(cloud) + len(td))} κλήσεις\n")

    rows: list[dict] = []
    await run(ai_core, "CUR", cur, cloud, rows)
    await run(ai_core, "NEW", new, cloud, rows)
    open_td(ai_core)
    await run(ai_core, "CUR", cur, td, rows)
    await run(ai_core, "NEW", new, td, rows)

    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    report(rows)
    print(f"\nΓράφτηκε: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

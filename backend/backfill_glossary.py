"""Backfill του per-file glossary στα ΗΔΗ ανεβασμένα έγγραφα.

ΤΙ ΠΡΟΒΛΗΜΑ ΛΥΝΕΙ:
    Το ingest γράφει πλέον `terms`/`domain` σε ΚΑΘΕ chunk (ai_core.ingest_pdf), κι
    από εκεί τα διαβάζει το query path (corpus_glossary.assemble -> optimize_query).
    Τα έγγραφα που μπήκαν ΠΡΙΝ υπάρξει ο corpus_glossary δεν έχουν αυτά τα κλειδιά,
    οπότε το assemble γυρίζει κενό και πέφτουν στα ΚΑΘΟΛΙΚΑ _CORPUS_DOMAIN/_TERMS
    (corpus_descriptor.json). Δηλαδή το σύστημα τρέχει ΔΥΟ μηχανισμούς ταυτόχρονα:
    per-file για ό,τι ανέβηκε πρόσφατα, καθολικό CLI descriptor για τα παλιά.

    Αυτό το script εξαλείφει τον δεύτερο, ώστε να μπορεί να ΣΒΗΣΤΕΙ ο descriptor.

ΓΙΑΤΙ ΟΧΙ ΑΠΛΟ RE-INGEST:
    Το re-ingest ξαναϋπολογίζει embeddings (λεπτά CPU) και αλλάζει τα chunk ids, που
    σπάει ό,τι τα κρατάει. Εδώ αλλάζουν ΜΟΝΟ δύο κλειδιά metadata — τα documents,
    τα embeddings και τα ids μένουν άθικτα.

ΤΙ ΓΡΑΦΕΙ ΚΑΙ ΤΙ ΟΧΙ — ΤΟ ΚΡΙΣΙΜΟ ΣΗΜΕΙΟ:
    Γράφει `terms`. ΔΕΝ γράφει `domain` (το θέτει κενό), και αυτό ΔΕΝ είναι παράλειψη.
    Μετρήθηκε στο test-domains σετ (probe_domain_glossary.py, runs/domain_glossary_JK.csv):

        CD_constrained  γενικό domain + όροι ανά αρχείο   95.0%   ooc 5/5 σιωπηλά
        CG_specific     γενικό domain + ειδικοί όροι      95.0%   ooc 5/5 σιωπηλά
        CH_doc_domain   domain ΑΝΑ ΑΡΧΕΙΟ + όροι          95.0%   ooc 4/5  ΔΙΑΡΡΟΗ o4
        CJ_doc_domain_only  domain ΑΝΑ ΑΡΧΕΙΟ, ΜΗΔΕΝ όροι 95.0%   ooc 4/5  ΔΙΑΡΡΟΗ o4
        CK_rich_domain      το ίδιο, σε πλούσια φράση     95.0%   ooc 4/5  ΔΙΑΡΡΟΗ o4

    Το per-file domain δίνει ΤΟ ΙΔΙΟ coverage με το γενικό (95.0%, οι ίδιες 19/20) και
    πληρώνει επιπλέον μία διαρροή. Ο μηχανισμός φαίνεται στο o4 («Πότε εξερράγη ο
    Βεζούβιος;»): με γενικό domain μεταφράζεται "When did Vesuvius last erupt?" (-3.28,
    σιωπή)· με domain ανά αρχείο γίνεται "Vesuvius last eruption date" (-1.56, ΔΙΑΡΡΟΗ).
    ΚΑΜΙΑ λέξη του corpus δεν προστέθηκε — άλλαξε το ΣΧΗΜΑ, από ερώτηση σε ονοματική
    φράση-keyword, και μόνο αυτό σηκώνει το logit κατά +1.72 πάνω από την πύλη (-2.6).
    Η περιοριστική οδηγία του optimize_query ΔΕΝ το πιάνει: διέπει το glossary, όχι το
    domain (το CH το έχει και διαρρέει· το CJ διαρρέει με ΜΗΔΕΝ όρους). Άρα το domain
    ανά αρχείο είναι μοχλός διαρροής χωρίς αντιστάθμισμα -> δεν μπαίνει.

ΚΟΚΚΙΩΣΗ — CHUNKS, ΟΧΙ ΣΕΛΙΔΕΣ:
    Η εξαγωγή τρέφεται με τα CHUNKS του αρχείου. Μετρήθηκε
    (evaluation/runs/granularity_probe.csv, test_domains n=20 in + 5 ooc):

        C0_no_glossary   85.0%   ooc 5/5 σιωπηλά
        CD_chunks        95.0%   ooc 5/5 σιωπηλά
        CM_pages         95.0%   ooc 4/5  ΔΙΑΡΡΟΗ o4

    ΙΔΙΟ coverage, αλλά η κοκκίωση σελίδας ΔΙΑΡΡΕΕΙ. Ο μηχανισμός: το φίλτρο
    boilerplate κόβει ό,τι είναι σε >=90% των μονάδων. Με σελίδες οι μονάδες είναι
    λίγες και μεγάλες, οπότε το κατώφλι πιάνει το ΘΕΜΑ: από το ηφαιστειολογικό paper
    έφυγαν τα «effusive, magma, eruptions». Χωρίς το «eruptions» ο μεταφραστής
    συνέθεσε το ερώτημα από τους όρους που ΕΜΕΙΝΑΝ -> «Vesuvius explosive activity»
    (-1.60, ΠΑΝΩ από την πύλη -2.6) αντί για το αθώο «Vesuvius eruptions» (-3.46).
    Στα 7 cloud papers η ίδια κοκκίωση σελίδας έκοβε mapreduce/serverless/cloud/faas.

ΠΟΥ ΠΡΟΣΓΕΙΩΝΕΤΑΙ Η ΠΑΡΑΓΩΓΗ ΜΕΤΑ:
    γενικό καθολικό domain + όροι ανά αρχείο + περιοριστική οδηγία, δηλαδή ΑΚΡΙΒΩΣ
    το H_generic_domain του κύριου σετ (97.78%, ooc 5/5) και το CD_constrained του
    cross-domain (95.0%, ooc 5/5). Και τα δύο ΜΕΤΡΗΜΕΝΑ. Χωρίς backfill, το σβήσιμο
    του descriptor αφήνει τα παλιά αρχεία σε «γενικό domain + ΜΗΔΕΝ όρους» — κελί που
    δεν έχει μετρηθεί σε κανένα από τα δύο σετ.

    ΠΡΟΫΠΟΘΕΣΗ: μετά το backfill πρέπει (α) να σβηστεί το corpus_descriptor.json και
    (β) το _FALLBACK_DOMAIN να γίνει γενικό. Όσο ο descriptor υπάρχει, το
    _load_corpus_descriptor τον διαβάζει πρώτο και το domain μένει cloud-specific.

ΧΡΗΣΗ (dry-run ΕΞ ΟΡΙΣΜΟΥ — δεν γράφει τίποτα χωρίς --apply):

    docker compose exec backend python backfill_glossary.py
    docker compose exec backend python backfill_glossary.py --apply
"""

import argparse
import re
import sys

import ai_core
import corpus_glossary

# Το chunk id είναι "{filename}_p{page-1}_c{chunk_idx}_{uuid}". Χρειαζόμαστε το
# chunk_idx για να ξαναφτιάξουμε τη ΣΕΙΡΑ μέσα στη σελίδα — το Chroma δεν εγγυάται
# σειρά επιστροφής και το κείμενο πρέπει να ενωθεί όπως ήταν.
_CHUNK_IDX = re.compile(r"_c(\d+)_[0-9a-f]+$")

BATCH = 500          # το Chroma κόβει τα πολύ μεγάλα update batches


def _chunk_idx(chunk_id: str) -> int:
    m = _CHUNK_IDX.search(chunk_id)
    return int(m.group(1)) if m else 0


def collect_by_file(col):
    """{file_name: {"ids": [...], "metas": [...], "pages": {page: [(idx, text)]}}}"""
    got = col.get(include=["documents", "metadatas"])
    files: dict[str, dict] = {}
    for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
        fn = meta.get("file_name")
        if not fn:
            continue
        e = files.setdefault(fn, {"ids": [], "metas": [], "pages": {}})
        e["ids"].append(cid)
        e["metas"].append(meta)
        e["pages"].setdefault(meta.get("page", 0), []).append((_chunk_idx(cid), doc))
    return files


def chunk_texts_of(entry: dict) -> list[str]:
    """Τα chunks του αρχείου, με τη σειρά τους. ΑΥΤΟ τροφοδοτεί την εξαγωγή όρων.

    ΓΙΑΤΙ chunks και ΟΧΙ σελίδες — ΜΕΤΡΗΜΕΝΟ, δες το docstring του module:
    το κατώφλι boilerplate (df >= 0.9*n) κόβει ό,τι εμφανίζεται σχεδόν παντού. Με
    σελίδες οι μονάδες είναι λίγες και μεγάλες, οπότε το κατώφλι πιάνει το ΘΕΜΑ και
    όχι μόνο τα υποσέλιδα. Με chunks (πολλά, μικρά) πιάνει μόνο τα υποσέλιδα.
    """
    out = []
    for _page, chunks in sorted(entry["pages"].items()):
        chunks.sort(key=lambda x: x[0])
        out.extend(text for _idx, text in chunks)
    return out


def page_texts_of(entry: dict) -> list[str]:
    """ΚΕΙΜΕΝΟ ΑΝΑ ΣΕΛΙΔΑ. ΔΕΝ χρησιμοποιείται από το backfill — μετρήθηκε ότι
    διαρρέει (CM_pages: 95.0% αλλά ooc 4/5). Μένει γιατί το probe_granularity.py
    το χρειάζεται για να ΑΝΑΠΑΡΑΓΕΙ τη σύγκριση· μη το βάλεις στη διαδρομή γραφής.
    """
    out = []
    for _page, chunks in sorted(entry["pages"].items()):
        chunks.sort(key=lambda x: x[0])
        out.append("\n".join(text for _idx, text in chunks))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="ΓΡΑΨΕ στα metadata· χωρίς αυτό είναι dry-run")
    ap.add_argument("--files", default="",
                    help="μόνο αυτά τα file_name (χωρισμένα με κόμμα)")
    ap.add_argument("--force", action="store_true",
                    help="ξαναγράψε ΚΑΙ όσα έχουν ήδη terms")
    ap.add_argument("--use-llm", action="store_true",
                    help="opt-in εξαγωγή με Gemini αντί για στατιστική. ΜΕΤΡΗΜΕΝΗ "
                         "ΙΣΟΠΑΛΙΑ (95.0%% και οι δύο) -> δεν συνιστάται· υπάρχει "
                         "για πεδίο όπου η στατιστική δώσει γενικούς όρους")
    args = ap.parse_args()

    col = ai_core.collection
    files = collect_by_file(col)
    if not files:
        print("Η συλλογή είναι κενή — τίποτα να γίνει.")
        return 0

    wanted = {s.strip() for s in args.files.split(",") if s.strip()}
    if wanted:
        missing = wanted - set(files)
        if missing:
            print(f"ΔΕΝ βρέθηκαν στη συλλογή: {sorted(missing)}")
            return 1
        files = {k: v for k, v in files.items() if k in wanted}

    print(f"{'ΕΦΑΡΜΟΓΗ' if args.apply else 'DRY-RUN (καμία εγγραφή)'} · "
          f"{len(files)} αρχεία στη συλλογή\n")

    planned = []
    for fn, entry in sorted(files.items()):
        n_chunks, n_pages = len(entry["ids"]), len(entry["pages"])
        had_terms = (entry["metas"][0].get("terms") or "").strip()
        had_domain = (entry["metas"][0].get("domain") or "").strip()

        if had_terms and not args.force:
            print(f"  ΠΑΡΑΛΕΙΨΗ {fn}  ({n_chunks} chunks) — έχει ήδη terms "
                  f"({len(had_terms.split(','))} όροι)· --force για επανεγγραφή")
            continue

        gloss = corpus_glossary.extract_glossary(
            chunk_texts_of(entry),
            use_llm=args.use_llm,
            model=ai_core.GEMINI_MODEL if args.use_llm else None,
            api_key=ai_core.GEMINI_API_KEY if args.use_llm else None,
        )
        terms = gloss["terms"]
        if not terms:
            print(f"  ΠΡΟΣΟΧΗ {fn}: η εξαγωγή δεν έδωσε όρους — παραλείπεται")
            continue

        planned.append((fn, entry, terms))
        print(f"  {fn}  ({n_chunks} chunks, {n_pages} σελίδες)")
        print(f"     terms  <- {terms}")
        # Το domain μηδενίζεται ΠΑΝΤΑ: μετρημένος μοχλός διαρροής (δες docstring).
        if had_domain:
            print(f"     domain <- (κενό)   [ΑΦΑΙΡΕΙΤΑΙ: '{had_domain}']")
        else:
            print("     domain <- (κενό)   [μένει κενό -> καθολικό fallback]")

    if not planned:
        print("\nΤίποτα προς εγγραφή.")
        return 0

    total = sum(len(e["ids"]) for _fn, e, _t in planned)
    if not args.apply:
        print(f"\nDRY-RUN: θα ενημερώνονταν {total} chunks σε {len(planned)} αρχεία.")
        print("Τρέξε ξανά με --apply για να γραφτούν.")
        return 0

    written = 0
    for fn, entry, terms in planned:
        ids, metas = entry["ids"], entry["metas"]
        # ΣΥΓΧΩΝΕΥΣΗ, όχι αντικατάσταση: το update γράφει ΟΛΟΚΛΗΡΟ το metadata dict
        # ανά id, οπότε τα υπάρχοντα κλειδιά (file_name/page/user_id/is_public/doc_id)
        # πρέπει να ξαναγραφτούν αυτούσια — αλλιώς χάνεται το authz και το scoping.
        new_metas = [{**m, "terms": terms, "domain": ""} for m in metas]
        for i in range(0, len(ids), BATCH):
            col.update(ids=ids[i:i + BATCH], metadatas=new_metas[i:i + BATCH])
        written += len(ids)
        print(f"  γράφτηκε {fn}: {len(ids)} chunks")

    # Τα bm25/dense caches κρατούν τα παλιά metadata -> χωρίς αυτό το assemble
    # συνεχίζει να βλέπει κενό glossary μέχρι το επόμενο restart.
    ai_core._bump_corpus_version()
    print(f"\nΣΥΝΟΛΟ: {written} chunks σε {len(planned)} αρχεία. Caches ακυρώθηκαν.")
    print("ΕΠΟΜΕΝΑ (χειροκίνητα, δεν τα κάνει αυτό το script):")
    print("  1. σβήσε το backend/corpus_descriptor.json")
    print("  2. _FALLBACK_DOMAIN -> 'scientific research papers' (ai_core.py)")
    print("  3. ξανατρέξε evaluation/thesis_numbers.py και ενημέρωσε τα νούμερα")
    return 0


if __name__ == "__main__":
    sys.exit(main())

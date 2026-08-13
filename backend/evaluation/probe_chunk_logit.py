"""Τι βαθμολογεί ο reranker για ΕΝΑ συγκεκριμένο ζεύγος (ερώτημα, chunk);

ΓΙΑΤΙ ΥΠΑΡΧΕΙ:
Όλα τα υπόλοιπα εργαλεία μετρούν το top-1 ή τις τελικές σελίδες. Όταν ένα
chunk κόβεται ΠΡΙΝ τον reranker (h002: RRF θέση 25), καμία μέτρηση δεν αγγίζει
ποτέ τη βαθμολογία του -- κι έτσι μια υπόθεση σαν «φταίει ο θόρυβος των
παραπομπών μέσα στο chunk» μένει αναπόδεικτη προς τις δύο κατευθύνσεις.

Εδώ δίνουμε το ζεύγος ΧΕΙΡΟΚΙΝΗΤΑ και βλέπουμε το ωμό logit, προαιρετικά με
κομμένο πρόθεμα, ώστε να απομονωθεί η επίδραση του ΚΕΙΜΕΝΟΥ του chunk από την
επίδραση της ΔΙΑΤΥΠΩΣΗΣ του ερωτήματος.

    docker compose exec backend python evaluation/probe_chunk_logit.py \
        --file excamera-nsdi17.pdf --page 15 --cut-before "Appendix" \
        --q "What exact settings did they use when they ran the encoder?" \
        --q "What exact settings did they use when they compressed the videos?"
"""
import argparse
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import ai_core


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--q", action="append", required=True,
                    help="ερώτημα (επαναλαμβανόμενο)")
    ap.add_argument("--cut-before", default=None,
                    help="πέτα ό,τι προηγείται αυτού του κειμένου μέσα στο chunk")
    args = ap.parse_args()

    where = {"$and": [{"file_name": args.file}, {"page": args.page}]}
    got = ai_core.collection.get(where=where)
    if not got["ids"]:
        print("Καμία εγγραφή για %s:%d" % (args.file, args.page))
        return 1
    pairs = sorted(zip(got["ids"], got["documents"]))

    variants = []
    for cid, doc in pairs:
        variants.append((cid, "ΠΛΗΡΕΣ", doc))
        if args.cut_before:
            i = doc.find(args.cut_before)
            if i > 0:
                variants.append((cid, "ΚΟΜΜΕΝΟ", doc[i:]))
                print("[%s] κόπηκαν %d από %d χαρακτήρες (%.0f%%) πριν το '%s'"
                      % (cid, i, len(doc), 100.0 * i / len(doc), args.cut_before))

    print("\ngate = %.1f (ωμά logits)\n" % ai_core.MIN_RERANK_SCORE)
    print("%-58s %-9s %8s" % ("ερώτημα", "chunk", "logit"))
    print("-" * 78)
    for q in args.q:
        scores = [float(x) for x in ai_core.reranker.predict(
            [[q, txt] for _c, _l, txt in variants],
            batch_size=ai_core.RERANK_BATCH_SIZE)]
        for (_cid, label, _txt), s in zip(variants, scores):
            flag = "" if s >= ai_core.MIN_RERANK_SCORE else "  ΚΑΤΩ ΑΠΟ ΤΟ GATE"
            print("%-58s %-9s %8.2f%s" % (q[:58], label, s, flag))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

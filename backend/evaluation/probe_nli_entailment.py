"""NLI entailment αντί για βαθμολογία κατατάκτη: διαφορετικό σήμα ή το ίδιο;

Η ΙΔΕΑ ΠΟΥ ΔΟΚΙΜΑΖΕΤΑΙ
----------------------
Ο cross-encoder μετράει  `query <-> passage similarity`.
Το NLI μετράει          `passage => answer`.
Το δεύτερο είναι διαφορετικό objective, εκπαιδευμένο σε entailment/contradiction
και όχι σε ranking. Άρα ΔΕΝ αποκλείεται από όσα ξέρουμε για τον κατατάκτη — και
δεν επιτρέπεται να απορριφθεί χωρίς μέτρηση.

ΓΙΑΤΙ ΜΗΔΕΝ ΚΟΣΤΟΣ API
----------------------
Οι υποψήφιες απαντήσεις και οι σελίδες παράγονται ΗΔΗ από το `probe_no_gate.py`
και διαβάζονται από το `no_gate_full.json`. Εδώ τρέχει ΜΟΝΟ το NLI μοντέλο σε
CPU. Άρα το ίδιο ζευγάρι γεννήσεων απαντά σε δύο ερωτήματα.

ΤΟ ΚΡΙΤΗΡΙΟ — claim-level, όχι document-level
---------------------------------------------
Μία βαθμολογία για ολόκληρη την απάντηση θα ήταν πάλι scalar, δηλαδή ακριβώς το
πρόβλημα που πάμε να λύσουμε. Αντ' αυτού:

    για ΚΑΘΕ πρόταση της απάντησης:
        max P(entailment) πάνω σε ΟΛΑ τα παράθυρα των σελίδων
    coverage = ποσοστό προτάσεων που περνούν το κατώφλι
    ACCEPT μόνο αν coverage == 100%

Αυτό είναι το «proof obligation»: αν έστω μία πρόταση δεν στηρίζεται, απόρριψη.

Η ΠΡΟΒΛΕΨΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (γράφεται ΠΡΙΝ, για να μετράει)
---------------------------------------------------------
Ταυτόσημο 3/1/0 με τον κριτή και το quote-first. Λόγος: η υποψήφια απάντηση
ΓΡΑΦΤΗΚΕ από τις σελίδες, άρα το `page => answer` τείνει στο ταυτολογικό. Ο
κρίκος που σπάει είναι `question -> answer`, τον οποίο το NLI δεν βλέπει.
Τρεις λάθος προβλέψεις έχουν ήδη καταγραφεί σε probes αυτής της σειράς, οπότε
η πρόβλεψη δεν αντικαθιστά τη μέτρηση.

ΠΡΟΣΟΧΗ ΣΤΗ ΓΛΩΣΣΑ
------------------
Μετρήθηκε ότι ένας κριτής LLM καταρρέει σε ελληνική ερώτηση πάνω σε αγγλικό
κείμενο (6 ΝΑΙ/14 ΟΧΙ έναντι 34/2). Εδώ 2 από τις 11 απαντήσεις είναι ελληνικές
πάνω σε αγγλικές σελίδες. Το `-xnli` μοντέλο είναι πολύγλωσσο ΚΑΙ περιλαμβάνει
ελληνικά, αλλά το cross-lingual entailment παραμένει δυσκολότερο. Το CSV έχει
στήλη `lang` ώστε να ελεγχθεί αν το φαινόμενο επανεμφανίζεται.

ΤΟ ΑΠΑΓΟΡΕΥΤΙΚΟ: έστω ΜΙΑ out_of_corpus που περνάει ρίχνει την ιδέα.

    docker compose exec backend python evaluation/probe_nli_entailment.py \\
        --csv evaluation/runs/nli.csv
"""
import argparse
import csv
import json
import os
import re
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

DEFAULT_SOURCE = "/app/evaluation/runs/no_gate_full.json"
# Το μοντέλο της πρότασης ήταν το `DeBERTa-v3-base-mnli-xnli`. ΕΓΙΝΕ GATED
# (401 στις 12/8/2026) — χρησιμοποιούμε τον πολύγλωσσο αδελφό του, ίδιος
# συγγραφέας, mDeBERTa-v3-base εκπαιδευμένο σε 2.7M ζεύγη XNLI/MNLI.
# ΓΙΑΤΙ ΥΠΟΧΡΕΩΤΙΚΑ ΠΟΛΥΓΛΩΣΣΟ: οι απαντήσεις είναι στη γλώσσα της ερώτησης
# (ελληνικές) και οι σελίδες αγγλικές, άρα ο έλεγχος είναι cross-lingual.
# Αγγλικό μοντέλο θα ξανάπεφτε στην παγίδα του `probe_grounding_verdict.py`.
DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

GREEK = re.compile(r"[α-ωΑ-Ωά-ώΆ-Ώ]")


def split_sentences(text: str):
    """Απλός τεμαχισμός. Πετάει bullets/αριθμήσεις και ό,τι είναι πολύ κοντό."""
    t = re.sub(r"\s+", " ", text or "").strip()
    parts = re.split(r"(?<=[.;!?·])\s+(?=[A-ZΑ-Ω0-9])", t)
    out = []
    for p in parts:
        p = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", p).strip()
        # <40 χαρ = τίτλος ή θραύσμα, δεν είναι ισχυρισμός προς επαλήθευση.
        if len(p) >= 40:
            out.append(p[:400])
    return out


def windows(pages, size: int, stride: int):
    """Παράθυρα χαρακτήρων. Το NLI δέχεται 512 tokens — ολόκληρη σελίδα δεν χωράει."""
    out = []
    for pg in pages:
        t = re.sub(r"\s+", " ", pg or "").strip()
        if not t:
            continue
        if len(t) <= size:
            out.append(t)
            continue
        for i in range(0, len(t) - stride, stride):
            w = t[i:i + size]
            if len(w) >= 120:
                out.append(w)
    return out


def entail_index(model) -> int:
    """Η θέση του `entailment` ΔΙΑΦΕΡΕΙ ανά checkpoint — λάθος index = λάθος probe."""
    id2label = getattr(model.config, "id2label", {}) or {}
    for i, lab in id2label.items():
        if str(lab).lower().startswith("entail"):
            return int(i)
    raise SystemExit(f"Δεν βρέθηκε ετικέτα entailment στο {id2label}")


def main(args) -> int:
    if not os.path.exists(args.source):
        print(f"Λείπει το {args.source} — τρέξε πρώτα probe_no_gate.py --csv ...")
        return 1
    with open(args.source, encoding="utf-8") as fh:
        data = json.load(fh)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(int(os.getenv("TORCH_THREADS", "8")))
    print(f"φόρτωση {args.model} ...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()
    ent = entail_index(model)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"φορτώθηκε σε {time.perf_counter() - t0:.1f}s · {n_par / 1e6:.0f}M "
          f"παράμετροι · entailment index {ent} · "
          f"ετικέτες {model.config.id2label}\n")

    field = "answer_" + args.scenario
    pfield = "pages_" + args.scenario
    print(f"σενάριο {args.scenario} · top-{args.top_pages} σελίδες · "
          f"κατώφλι entail {args.threshold:.2f}\n")

    rows, total_pairs, t_start = [], 0, time.perf_counter()
    for d in data:
        ans = (d.get(field) or "").strip()
        claims = split_sentences(ans)
        wins = windows((d.get(pfield) or [])[:args.top_pages],
                       args.window, args.stride)
        lang = "EL" if GREEK.search(ans) else "EN"
        if not claims or not wins:
            rows.append({"id": d["id"], "category": d["category"], "lang": lang,
                         "claims": len(claims), "windows": len(wins),
                         "supported": 0, "claim_coverage": 0.0,
                         "min_entail": 0.0, "mean_entail": 0.0, "accept": False})
            print(f"{d['id']:<6} [{d['category']:<14}] χωρίς ισχυρισμούς ή σελίδες")
            continue

        best_per_claim = []
        for c in claims:
            best = 0.0
            for i in range(0, len(wins), args.batch):
                chunk = wins[i:i + args.batch]
                enc = tok(chunk, [c] * len(chunk), truncation=True,
                          max_length=args.max_len, padding=True,
                          return_tensors="pt")
                with torch.no_grad():
                    logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, ent]
                best = max(best, float(probs.max()))
                total_pairs += len(chunk)
            best_per_claim.append(best)

        ok = [b >= args.threshold for b in best_per_claim]
        cov = 100.0 * sum(ok) / len(ok)
        accept = all(ok)
        rows.append({"id": d["id"], "category": d["category"], "lang": lang,
                     "claims": len(claims), "windows": len(wins),
                     "supported": sum(ok), "claim_coverage": round(cov, 1),
                     "min_entail": round(min(best_per_claim), 4),
                     "mean_entail": round(sum(best_per_claim) / len(best_per_claim), 4),
                     "accept": accept})
        print(f"{d['id']:<6} [{d['category']:<14}] {lang}  "
              f"ισχυρισμοί {sum(ok):>2}/{len(ok):<2}  κάλυψη {cov:5.1f}%  "
              f"min {min(best_per_claim):.3f}  {'ACCEPT' if accept else 'reject'}",
              flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\n{total_pairs} ζεύγη NLI σε {elapsed:.1f}s "
          f"({1000 * elapsed / max(total_pairs, 1):.1f} ms/ζεύγος)")

    # ------------------------------------------------ σάρωση κατωφλίων
    print("\n" + "=" * 78)
    print(f"{'κατώφλι':<10}{'σωσμένες':>10}{'ΧΩΡΙΣ ΥΛΙΚΟ':>14}{'ΔΙΑΡΡΟΕΣ ooc':>15}")
    print("-" * 78)
    kw_cov = {d["id"]: d for d in data}
    for thr in (0.30, 0.50, 0.70, 0.90, 0.95):
        saved = nomat = leaks = 0
        for r in rows:
            # Ξανακρίνουμε με το ΙΔΙΟ κριτήριο, άλλο κατώφλι.
            passes = r["claims"] > 0 and r["min_entail"] >= thr
            if not passes:
                continue
            if r["category"] == "out_of_corpus":
                leaks += 1
                continue
            d = kw_cov[r["id"]]
            pages = " ".join(d.get("pages_" + args.scenario) or []).lower()
            has = any(str(k).lower() in pages for k in (d.get("keywords") or []))
            saved += bool(has)
            nomat += not has
        flag = "  <== ΑΠΑΓΟΡΕΥΤΙΚΟ" if leaks else ""
        print(f"{thr:<10.2f}{saved:>10}{nomat:>14}{leaks:>15}{flag}")
    print("-" * 78)

    # ------------------------------------- διαχωρίζει το min_entail;
    with_mat, without, oocs = [], [], []
    for r in rows:
        d = kw_cov[r["id"]]
        pages = " ".join(d.get("pages_" + args.scenario) or []).lower()
        has = any(str(k).lower() in pages for k in (d.get("keywords") or []))
        if r["category"] == "out_of_corpus":
            oocs.append(r["min_entail"])
        elif has:
            with_mat.append(r["min_entail"])
        else:
            without.append(r["min_entail"])
    print("ΔΙΑΧΩΡΙΖΕΙ ΤΟ min_entail;")
    for label, vals in (("ΜΕ υλικό     ", sorted(with_mat)),
                        ("ΧΩΡΙΣ υλικό  ", sorted(without)),
                        ("out_of_corpus", sorted(oocs))):
        if vals:
            print(f"  {label} n={len(vals):<3} " + " ".join(f"{v:.3f}" for v in vals))
    if with_mat and (without or oocs):
        bad = (without or []) + (oocs or [])
        verdict = ("ΚΑΘΑΡΟΣ ΔΙΑΧΩΡΙΣΜΟΣ — υπάρχει κατώφλι"
                   if min(with_mat) > max(bad)
                   else "ΕΠΙΚΑΛΥΨΗ — κανένα κατώφλι δεν τις χωρίζει")
        print(f"  {verdict}")
    print("=" * 78)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"→ γράφτηκε {args.csv}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="NLI entailment ως σήμα επάρκειας τεκμηρίων.")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--scenario", choices=["A", "B"], default="B",
                    help="B = σελίδες 2ου περάσματος (η πραγματική επιλογή)")
    ap.add_argument("--top-pages", type=int, default=3,
                    help="πόσες σελίδες ελέγχονται (κόστος CPU)")
    ap.add_argument("--window", type=int, default=1200)
    ap.add_argument("--stride", type=int, default=900)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--csv", default=None)
    raise SystemExit(main(ap.parse_args()))

"""Φάση 1.1, βήμα 1: πόσο ΑΡΓΕΙ ένας άλλος reranker στη ΔΙΚΗ μας CPU; — ΜΟΝΟ ταχύτητα.

ΓΙΑΤΙ ΠΡΩΤΑ Η ΤΑΧΥΤΗΤΑ: ο προϋπολογισμός της Φάσης 1 (γραμμένος 25/9) είναι ανάκτηση ≤1.2 s.
Το gte-reranker-modernbert-base έχει ~149M παραμέτρους έναντι 33M του MiniLM-L-12 (~4.5×). Αν
βγει 4-5 s, ΟΛΗ η υπόλοιπη δουλειά (βαθμονόμηση φύλακα, corrective, scoreboard, κριτής) είναι
χαμένη. Αυτό το script κοστίζει ΜΗΔΕΝ κλήσεις Gemini.

ΤΙ ΚΑΝΕΙ:
  1. Τρέχει τον ΠΡΑΓΜΑΤΙΚΟ search_documents στο απομονωμένο store (418 chunks, όπως το scoreboard)
     για τις ΑΓΓΛΙΚΕΣ ερωτήσεις του golden_set_50 — αυτές δεν μεταφράζονται, άρα 0 Gemini.
     Ο corrective είναι κλειστός και το generate_once σκάει αν κληθεί: εγγύηση, όχι ελπίδα.
  2. «Πιάνει» τα 15 ζευγάρια (ερώτημα, chunk) που φτάνουν στον reranker — ΑΚΡΙΒΩΣ της παραγωγής.
  3. Χρονομετρεί ΚΑΙ τα δύο μοντέλα πάνω στα ΙΔΙΑ ζευγάρια, εναλλάξ (ίδιος θόρυβος μηχανήματος),
     2 επαναλήψεις ανά ερώτηση, διάμεσος. Με batch 4 (παραγωγή) και 16 (όλα μαζί).
     15 ερωτήσεις αρκούν: ψάχνουμε αν χωράει στο 1.2 s, όχι το τρίτο δεκαδικό (~5 λεπτά max).
  4. Εκτίμηση ανάκτησης με το νέο = (μετρημένη ανάκτηση σήμερα) − (rerank MiniLM) + (rerank νέου).

ΕΛΕΓΧΟΣ ΚΛΙΜΑΚΑΣ: ο φύλακας δουλεύει με ΩΜΑ logits (−2.6). Αν το νέο μοντέλο βγάζει sigmoid [0,1],
το −2.6 περνάει ΤΑ ΠΑΝΤΑ. Το script ζητάει activation Identity και ΕΛΕΓΧΕΙ ότι βγήκαν αρνητικά σκορ.

═══ ΠΡΟΒΛΕΨΗ — γραμμένη ΠΡΙΝ το τρέξιμο (29/9/2026) ═══
  rerank MiniLM-L-12 (batch 4)   0.5-0.8 s ανά ερώτηση
  rerank gte-modernbert           3-5 s (4-7× πιο αργό)
  εκτίμηση ανάκτησης με το νέο    ~3.5-5 s  ->  ΑΠΟΤΥΓΧΑΝΕΙ τον προϋπολογισμό
  batch 16 vs 4                   διαφορά <15% και στα δύο

═══ ΑΠΟΤΕΛΕΣΜΑ (29/9/2026, runs/reranker_latency_gte-reranker-modernbert-base.txt) ═══
  rerank, 15 ερωτήσεις × 15 ζευγάρια, διάμεσος (p90):
    batch 4   MiniLM 0.590 (0.800) · gte-modernbert 3.264 (3.963)  -> 5.5× πιο αργό
    batch 16  MiniLM 0.794 (1.189) · gte-modernbert 4.069 (6.311)  -> 5.1×
  ανάκτηση σήμερα 0.735 s (rerank 0.714 = 97%) -> ΕΚΤΙΜΗΣΗ με το νέο 3.285 s έναντι 1.2 s
  -> ΑΠΟΤΥΓΧΑΝΕΙ τον προϋπολογισμό, 2.7× πάνω από το όριο.
  Πρόβλεψη: MiniLM 0.5-0.8 ✓ · νέο 3-5 s ✓ · εκτίμηση 3.5-5 s ✗ (3.29 — το εκτός-rerank κομμάτι
  είναι ~0.02 s, όχι ~0.5 που υπέθεσα) · batch 16 ≈ 4 ✗ (το 16 είναι 25-35% ΠΙΟ ΑΡΓΟ και στα δύο:
  το batch 4 της παραγωγής ισχύει και για το νέο μοντέλο). Δύο στις τέσσερις· η ετυμηγορία σωστή.
  Πληροφορία, ΟΧΙ κρίση ποιότητας: κλίμακα νέου −1.09…3.85 (MiniLM −4.44…9.27) -> ο φύλακας θα
  ήθελε πλήρη επαναβαθμονόμηση· ίδιο top-1 7/15.

═══ ΚΑΝΟΝΑΣ ΑΠΟΦΑΣΗΣ — γραμμένος ΠΡΙΝ ═══
  εκτίμηση ≤ 1.2 s   -> συνεχίζουμε στην ποιότητα (φύλακας -> corrective -> scoreboard -> κριτής)
  εκτίμηση > 1.2 s   -> ΑΠΟΤΥΓΧΑΝΕΙ τον προϋπολογισμό. ΔΕΝ χαλαρώνουμε το όριο εκ των υστέρων·
                        σταματάμε και αποφασίζουμε μαζί (απόρριψη ή μικρότερο μοντέλο)

    docker compose exec backend python evaluation/bench_reranker_latency.py
    docker compose exec backend python evaluation/bench_reranker_latency.py --model <όνομα HF>

Την ΠΡΩΤΗ φορά κατεβάζει το μοντέλο (~600 MB). ΜΗΝ τρέχει ταυτόχρονα με scoreboard ή άλλο eval
(κοινό store + ανταγωνισμός για CPU που θα χάλαγε και τη μέτρηση).
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_near_ooc as E

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = "/tmp/scoreboard.lock"  # noqa: S108  ΙΔΙΑ κλειδαριά με το scoreboard (κοινό store)
BUDGET_S = 1.2


class _Tee:
    """Ό,τι τυπώνεται πάει ΚΑΙ σε αρχείο στο runs/ — τα αποτελέσματα δεν ζουν μόνο σε ένα τερματικό."""

    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")  # noqa: SIM115  ζει όσο το script
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


def timed(model, pairs, batch: int) -> tuple[float, list[float]]:
    t0 = time.perf_counter()
    s = model.predict(pairs, batch_size=batch)
    return time.perf_counter() - t0, [float(x) for x in s]


def load_candidate(name: str, device: str):
    import torch
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(name, device=device, activation_fn=torch.nn.Identity())
    except TypeError:                      # παλαιότερο όνομα παραμέτρου
        return CrossEncoder(name, device=device, default_activation_function=torch.nn.Identity())


async def run(args) -> int:
    ai_core, _saved = E.open_store()
    if ai_core.collection.count() != 418:
        print(f"!! Το store έχει {ai_core.collection.count()} chunks, όχι 418 — σταματάω.")
        return 1

    # ΜΗΔΕΝ Gemini: corrective κλειστός + ό,τι πάει να καλέσει γέννηση σκάει
    ai_core.ENABLE_CORRECTIVE = False

    async def _no_gemini(*_a, **_k):
        raise RuntimeError("bench_reranker_latency: απαγορεύεται κλήση Gemini")

    E.gemini_rest.generate_once = _no_gemini

    with open(E.GOLDEN_50, encoding="utf-8") as f:
        golden = [json.loads(x) for x in f if x.strip()]
    qs = [t for t in golden if not ai_core._has_greek(t["question"])][: args.n or None]

    # ---- 1-2: πραγματική ανάκτηση, πιάνουμε τα ζευγάρια του reranker ----
    captured: list[tuple[list, float]] = []
    inner = ai_core.reranker.predict           # ήδη τυλιγμένο από τον spy του open_store

    def capture(pairs, **kw):
        t0 = time.perf_counter()
        out = inner(pairs, **kw)
        captured.append(([list(p) for p in pairs], time.perf_counter() - t0))
        return out

    ai_core.reranker.predict = capture
    print(f"\nΑνάκτηση (MiniLM, όπως η παραγωγή) σε {len(qs)} αγγλικές ερωτήσεις...", flush=True)
    await ai_core.search_documents(qs[0]["question"], None, user_id=E.TEST_USER)   # ζέσταμα
    captured.clear()
    rows = []
    for t in qs:
        n0 = len(captured)
        t0 = time.perf_counter()
        await ai_core.search_documents(t["question"], None, user_id=E.TEST_USER)
        total = time.perf_counter() - t0
        if len(captured) == n0:                       # δεν έφτασε στον reranker (π.χ. αναφορικό)
            print(f"  {t['id']}: χωρίς rerank — παραλείπεται")
            continue
        pairs, rr = captured[n0]
        rows.append({"id": t["id"], "pairs": pairs, "total": total, "rerank_prod": rr})
    ai_core.reranker.predict = inner

    # ---- 3: τα δύο μοντέλα στα ΙΔΙΑ ζευγάρια ----
    print(f"\nΦόρτωση υποψήφιου: {args.model} (την 1η φορά κατεβαίνει) ...", flush=True)
    t0 = time.perf_counter()
    cand = load_candidate(args.model, ai_core.DEVICE)
    print(f"  φορτώθηκε σε {time.perf_counter() - t0:.1f} s")
    p_new = sum(p.numel() for p in cand.model.parameters()) / 1e6
    p_old = sum(p.numel() for p in ai_core.reranker.model.parameters()) / 1e6
    print(f"  παράμετροι: νέο {p_new:.0f}M · MiniLM {p_old:.0f}M ({p_new / p_old:.1f}×) · "
          f"max_length νέου {getattr(cand, 'max_length', '?')} · torch threads "
          f"{__import__('torch').get_num_threads()}")

    models = {"minilm": ai_core.reranker, "new": cand}
    batches = [int(b) for b in args.batches.split(",")]
    t0 = time.perf_counter()
    for _ in range(2):                                   # ζέσταμα (1η κλήση = compile/allocs)
        timed(cand, rows[0]["pairs"], batches[0])
    print(f"  ζέσταμα νέου: {time.perf_counter() - t0:.1f} s")

    times = {(m, b): [] for m in models for b in batches}
    scores_new: list[float] = []
    top1_same = 0
    print(f"\n{'id':<7}" + "".join(f"{m + '@' + str(b):>13}" for b in batches for m in models)
          + f"{'best MiniLM':>13}{'best νέο':>11}")
    for r in rows:
        line = f"{r['id']:<7}"
        best = {}
        for b in batches:
            for m, model in models.items():
                reps = []
                for _ in range(args.reps):
                    dt, s = timed(model, r["pairs"], b)
                    reps.append(dt)
                med = statistics.median(reps)
                times[(m, b)].append(med)
                line += f"{med:>13.3f}"
                best[m] = s
        scores_new += best["new"]
        top1_same += max(range(len(best["new"])), key=best["new"].__getitem__) == \
            max(range(len(best["minilm"])), key=best["minilm"].__getitem__)
        print(line + f"{max(best['minilm']):>13.2f}{max(best['new']):>11.2f}", flush=True)

    # ---- 4: αναφορά ----
    print("\n" + "=" * 78)
    print(f"RERANK, {len(rows)} ερωτήσεις × {len(rows[0]['pairs'])} ζευγάρια, διάμεσος (p90), δευτερόλεπτα")
    print("=" * 78)
    for b in batches:
        mo, mn = statistics.median(times[("minilm", b)]), statistics.median(times[("new", b)])
        print(f"  batch {b:<3} MiniLM {mo:.3f} ({pct(times[('minilm', b)], .9):.3f}) · "
              f"νέο {mn:.3f} ({pct(times[('new', b)], .9):.3f})  -> {mn / mo:.1f}× πιο αργό")

    total = statistics.median(r["total"] for r in rows)
    rr_prod = statistics.median(r["rerank_prod"] for r in rows)
    best_b = min(batches, key=lambda b: statistics.median(times[("new", b)]))
    new_rr = statistics.median(times[("new", best_b)])
    est = total - rr_prod + new_rr
    print(f"\n  ανάκτηση σήμερα (διάμεσος, όλο το search_documents): {total:.3f} s "
          f"— εκ των οποίων rerank {rr_prod:.3f} s")
    print(f"  ΕΚΤΙΜΗΣΗ ανάκτησης με το νέο (καλύτερο batch {best_b}): {est:.3f} s  "
          f"(προϋπολογισμός {BUDGET_S} s)")

    neg = sum(1 for s in scores_new if s < 0)
    in01 = all(0.0 <= s <= 1.0 for s in scores_new)
    print(f"\n  κλίμακα νέου: min {min(scores_new):.2f} · max {max(scores_new):.2f} · "
          f"αρνητικά {neg}/{len(scores_new)}"
          + ("   !! ΟΛΑ ΣΤΟ [0,1] — μοιάζει sigmoid, ο φύλακας −2.6 ΘΑ ΠΕΡΝΑΓΕ ΤΑ ΠΑΝΤΑ" if in01
             else "   (ωμά logits ✓)"))
    print(f"  ίδιο top-1 με το MiniLM: {top1_same}/{len(rows)}   (πληροφορία μόνο — ΔΕΝ κρίνει ποιότητα)")

    verdict = "ΣΥΝΕΧΙΖΟΥΜΕ στην ποιότητα" if est <= BUDGET_S else \
        "ΑΠΟΤΥΓΧΑΝΕΙ τον προϋπολογισμό — σταματάμε και αποφασίζουμε μαζί"
    print(f"\n  ΕΤΥΜΗΓΟΡΙΑ (κανόνας γραμμένος πριν): {verdict}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Alibaba-NLP/gte-reranker-modernbert-base")
    ap.add_argument("--n", type=int, default=15, help="πλήθος αγγλικών ερωτήσεων (0 = όλες)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--batches", default="4,16")
    args = ap.parse_args()
    out = os.path.join(HERE, "runs", f"reranker_latency_{args.model.split('/')[-1]}.txt")
    sys.stdout = _Tee(out)
    print(f"(η έξοδος σώζεται και στο {os.path.relpath(out, HERE)})")
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"!! Τρέχει ήδη scoreboard/eval στο κοινό store (ή έμεινε κλειδαριά: σβήσε το {LOCK}).")
        return 1
    try:
        return asyncio.run(run(args))
    finally:
        os.close(fd)
        os.remove(LOCK)


if __name__ == "__main__":
    sys.exit(main())

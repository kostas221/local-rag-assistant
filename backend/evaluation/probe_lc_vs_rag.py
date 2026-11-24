"""RAG ή ΟΛΟ το κείμενο των ΕΠΙΛΕΓΜΕΝΩΝ άρθρων; Μέρος 1 από 2: οι απαντήσεις (στο Docker).
Προκαταχώριση: docs/LONGCONTEXT_VS_RAG_PREREG_2026-11-23.md. Μέρος 2 (κρίση GPT-4.1 + πίνακας + κανόνας):
judge_lc_vs_rag.py, στον υπολογιστή.

ΤΙ ΚΑΝΕΙ: για κάθε ερώτηση, με επιλεγμένα ΜΟΝΟ τα άρθρα της, δύο απαντήσεις από τον ΠΡΑΓΜΑΤΙΚΟ ask_ai
(ίδιο prompt, ίδιες ετικέτες [S#], ίδιο μοντέλο):
    rag   search_documents(q, target_doc_ids=επιλεγμένα) -> ask_ai(precomputed=οι σελίδες της αναζήτησης)
    full  ask_ai(precomputed=ΟΛΕΣ οι σελίδες των επιλεγμένων, με τη σειρά) — χωρίς αναζήτηση, χωρίς φύλακα
Οι σελίδες του «full» είναι το ίδιο κείμενο με τις σελίδες της αναζήτησης (probe_decomp_answers.page_of).

ΣΚΕΛΗ (μέγεθος = πόσα άρθρα επιλέγονται):
    one      36 κύριου (main_set_papers.json) + 16 πινάκων (golden_tables, πεδίο doc)       1 άρθρο
    two      29 (golden_multihop_v2)                                                      2 άρθρα
    three    12 (golden_three_papers)                                                     3 άρθρα
    safety   42 κοντινές + 5 άσχετες του κύριου, ΚΑΘΕ ΜΙΑ σε 1, 2 και 3 άρθρα (κυκλικά)   1-3 άρθρα
    distract (χωριστό διαγνωστικό, ΔΕΝ μπαίνει στον προκαταχωρισμένο κανόνα): 14 γνωστές
             μονοαρθρικές ερωτήσεις (2/άρθρο) με +1 ή +2 επιπλέον PDF (κυκλικά, 1 κλήρωση) 2-3 άρθρα
Κληρώσεις: 2 για one/two/three · 1 για safety και distract. Η αναζήτηση του RAG γίνεται ΜΙΑ φορά ανά
ερώτηση· οι κληρώσεις αφορούν τη γέννηση. Μέσα σε κάθε ομάδα ίδιων PDF τρέχει ΠΡΩΤΑ όλη η κλήρωση 1
(πραγματικές διαδοχικές διαφορετικές ερωτήσεις) και ΜΕΤΑ η κλήρωση 2 (επανάληψη για το πείραμα), ώστε
κάθε γραμμή να ξεχωρίζει `cache_role` = first / next / repeat / cut.

ΠΑΓΩΜΑ & ΠΛΑΦΟΝ:
    runs/longctx/generation_calls.json   ΜΟΝΟ γεννήσεις απαντήσεων (ποτέ κλήσεις ανάκτησης)
    runs/longctx/retrieval_calls.json    νέες κλήσεις ανάκτησης (+ usage)
    runs/longctx/live_timing.json        ζωντανή χρονομέτρηση από έναρξη ερώτησης ως 1ο ορατό κείμενο
    runs/longctx/budget.json             σωρευτική δαπάνη (γέννηση + ανάκτηση + ζωντανή μέτρηση) που
                                         διατηρείται στο resume· ελέγχεται με περιθώριο ΠΡΙΝ από κάθε
                                         πληρωμένη κλήση. Σε διακοπή σώζει πρόοδο και επιστρέφει exit != 0.

    # δοκιμή: 2 ερωτήσεις ανά σκέλος -> runs/longctx/answers_smoke.json
    docker compose -f docker-compose.yml -f docker-compose.upgrade.yml run --rm --no-deps \\
        -e DATABASE_URL=sqlite:// -e GEMINI_MODEL=gemini-3.8-flash -v eval_tmp:/tmp backend \\
        python -u evaluation/probe_lc_vs_rag.py --limit 2
    # Φάση Α (ποιότητα, κόστος, χρόνος) -> runs/longctx/answers_phaseA.json
    docker compose -f docker-compose.yml -f docker-compose.upgrade.yml run --rm --no-deps \\
        -e DATABASE_URL=sqlite:// -e GEMINI_MODEL=gemini-3.8-flash -v eval_tmp:/tmp backend \\
        python -u evaluation/probe_lc_vs_rag.py --phase A --no-distract --live-timing 2 --max-usd 6
    # Φάση Β (ασφάλεια) ΜΟΝΟ αν τη ζητήσει ο κριτής μετά την Α, στα μεγέθη που λέει (π.χ. 2,3)
    #   ... python -u evaluation/probe_lc_vs_rag.py --phase B --safety-sizes 2,3 --max-usd <νέο πλαφόν>

ΔΥΟ ΦΑΣΕΙΣ (τροποποίηση 24/11/2026, πριν από το πλήρες τρέξιμο — βλ. προκαταχώριση): Α = σκέλη one/two/three,
Β = ασφάλεια μόνο στα μεγέθη που πέρασαν την Α. Ο κανόνας είναι ίδιος· αλλάζει μόνο η σειρά.
"""
import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from itertools import groupby

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
import lc_vs_rag_spec as SPEC
import probe_decomp_answers as PA
import scoreboard as S

import gemini_rest

HERE = os.path.dirname(os.path.abspath(__file__))
LC_DIR = SPEC.LC_DIR
GEN_PATH = os.path.join(LC_DIR, "generation_calls.json")
RET_PATH = os.path.join(LC_DIR, "retrieval_calls.json")
LIVE_PATH = os.path.join(LC_DIR, "live_timing.json")
BUDGET_PATH = SPEC.BUDGET_PATH
MODEL = "gemini-3.8-flash"
SEED = SPEC.SEED
DRAWS = SPEC.DRAWS
ARMS = SPEC.ALL_ARMS
PRIMARY_ARMS = SPEC.PRIMARY_ARMS
OUT_NAME = {"A": "answers_phaseA.json", "B": "answers_phaseB.json", "all": "answers.json"}


def build_items(files: list[str], *, include_distract: bool = True, arms: tuple[str, ...] = PRIMARY_ARMS,
                safety_sizes: tuple[int, ...] = SPEC.SAFETY_SIZES) -> list[dict]:
    return SPEC.build_items(files, include_distract=include_distract, arms=arms, safety_sizes=safety_sizes)


class DrawStream:
    """Πάγωμα ΜΟΝΟ της γέννησης απαντήσεων (όταν self.active=True) με κλειδί κλήρωση + prompt.

    Όταν self.active=False (δηλαδή κλήσεις του generate_once κατά την ανάκτηση), ΔΕΝ αγγίζει το
    generation_calls.json ούτε τους μετρητές γέννησης — απλώς ελέγχει το σωρευτικό πλαφόν ΠΡΙΝ
    την κλήση και καταγράφει τη δαπάνη ανάκτησης στο BudgetLedger.
    """

    def __init__(self, real, budget: SPEC.BudgetLedger, gen_path: str = GEN_PATH):
        self.real = real
        self.budget = budget
        self.gen_path = gen_path
        self.data: dict = {}
        if os.path.exists(gen_path):
            with open(gen_path, encoding="utf-8") as f:
                self.data = json.load(f)
        self.active = False      # True ΜΟΝΟ μέσα στην answer() -> δεν πιάνει ποτέ το generate_once
        self.live_mode = False   # True στη ζωντανή χρονομέτρηση -> δεν διαβάζει/γράφει generation_calls
        self.draw = 1
        self.new = self.hits = 0
        self.errors: list[str] = []
        self.last: dict | None = None
        self.last_fresh = False
        self.last_ret_usage: dict = {}
        self.last_ret_s: float = 0.0
        self.last_ret_usd: float = 0.0

    async def __call__(self, prompt, **kw):
        if not self.active:
            # Κλήση από gemini_rest.generate_once (μετάφραση / αναδιατύπωση / κριτής επάρκειας).
            # Φτάνει εδώ ΜΟΝΟ σε cache miss του RetrievalCalls (ή σε live bypass) -> πληρωμένη κλήση.
            max_out = kw.get("max_output_tokens", 1024)
            kind = "live" if self.live_mode else "ret"
            self.budget.check_before_call(prompt, max_out, label="ανάκτηση" if kind == "ret" else "ζωντανή ανάκτηση")
            t0 = time.perf_counter()
            usage: dict = {}
            try:
                async for ev_kind, d in self.real(prompt, **kw):
                    if ev_kind == "usage":
                        usage = d
                    yield ev_kind, d
            finally:
                elapsed = time.perf_counter() - t0
                cost = self.budget.record(kind, usage) if usage else 0.0
                self.last_ret_usage = usage
                self.last_ret_s = elapsed
                self.last_ret_usd = cost
            return

        if self.live_mode:
            # Ζωντανή χρονομέτρηση γέννησης: πάντα νέα κλήση, χωρίς να πειράξει το generation_calls.json.
            self.budget.check_before_call(prompt, kw.get("max_output_tokens", 4096), label="ζωντανή γέννηση")
            t0 = time.perf_counter()
            text, usage, finish, ttft = "", {}, None, None
            try:
                async for kind, d in self.real(prompt, **kw):
                    if kind == "text":
                        if ttft is None and d:
                            ttft = time.perf_counter() - t0
                        text += d
                    elif kind == "finish":
                        finish = d
                    else:
                        usage = d
                    yield kind, d
            except Exception as e:
                self.errors.append(f"ζωντανή γέννηση: {type(e).__name__}: {e}"[:200])
                raise
            finally:
                cost = self.budget.record("live", usage) if usage else 0.0
            self.last = {
                "out": text, "finish": finish, "usage": usage, "usd": cost,
                "ttft": round(ttft, 3) if ttft is not None else None,
                "total_s": round(time.perf_counter() - t0, 3),
            }
            self.last_fresh = True
            return

        key = hashlib.sha256(f"{self.draw}\x00{prompt}".encode()).hexdigest()
        if key in self.data:
            entry = self.data[key]
            self.hits += 1
            self.last, self.last_fresh = entry, False
            yield "text", entry["out"]
            if kw.get("with_finish") and entry.get("finish"):
                yield "finish", entry["finish"]
            yield "usage", entry.get("usage") or {}
            return

        self.budget.check_before_call(prompt, kw.get("max_output_tokens", 4096), label="γέννηση")
        t0 = time.perf_counter()
        text, usage, finish, ttft = "", {}, None, None
        try:
            async for kind, d in self.real(prompt, **kw):
                if kind == "text":
                    if ttft is None and d:
                        ttft = time.perf_counter() - t0
                    text += d
                elif kind == "finish":
                    finish = d
                else:
                    usage = d
                yield kind, d
        except Exception as e:
            # Το ask_ai ΚΑΤΑΠΙΝΕΙ το σφάλμα και γράφει «Προσωρινό πρόβλημα…» -> χωρίς αυτό θα κρίναμε
            # το μήνυμα σφάλματος ως απάντηση.
            if not self.budget.exceeded:
                self.errors.append(f"γέννηση: {type(e).__name__}: {e}"[:200])
            raise
        finally:
            if usage:
                self.budget.record("gen", usage)
        if not text.strip():
            self.errors.append("γέννηση: κενή απάντηση")
            return
        self.new += 1
        entry = {
            "draw": self.draw, "tail": S.safe_tail(prompt), "out": text, "finish": finish,
            "usage": usage, "ttft": round(ttft, 3) if ttft is not None else None,
            "total_s": round(time.perf_counter() - t0, 3),
        }
        self.data[key] = entry
        self.last, self.last_fresh = entry, True

    def save(self) -> None:
        SPEC._atomic_write_json(self.gen_path, self.data)


class RetrievalCalls(S.Frozen):
    """Το πάγωμα του scoreboard (μόνο ανάγνωση) + δικό μας αρχείο για όσες κλήσεις είναι νέες.
    Καταγράφει ανά ερώτηση πόσες κλήσεις ήταν ζωντανές vs παγωμένες, καθώς και το usage/κόστος τους."""

    def __init__(self, gen: DrawStream, ret_path: str = RET_PATH):
        super().__init__(ret_path, fresh=False)
        self.gen = gen
        self.base = {}
        if os.path.exists(S.FROZEN_PATH):
            with open(S.FROZEN_PATH, encoding="utf-8") as f:
                self.base = json.load(f)
        self.data = {**self.base, **self.data}
        self.bypass = False
        self.q_live = 0
        self.q_frozen = 0
        self.q_live_s = 0.0
        self.q_ret_usd = 0.0

    def reset_question_stats(self) -> None:
        self.q_live = 0
        self.q_frozen = 0
        self.q_live_s = 0.0
        self.q_ret_usd = 0.0

    def install(self) -> None:
        inner = gemini_rest.generate_once

        async def frozen(prompt, **kw):
            m = S._DOMAIN_RE.search(prompt)
            if m:
                self.last_domain = m.group(1)
            key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if not self.bypass and key in self.data:
                self.hits += 1
                self.q_frozen += 1
                text = self.data[key]["out"]
                E._prompts.append((prompt[:60], text.strip(" \"'\n")))
                return text
            self.gen.last_ret_usage = {}
            self.gen.last_ret_s = 0.0
            self.gen.last_ret_usd = 0.0
            try:
                text = await inner(prompt, **kw)
            except Exception as e:
                if not self.gen.budget.exceeded:
                    self.errors.append(f"{type(e).__name__}: {e}"[:200])
                raise
            self.q_live += 1
            self.q_live_s += self.gen.last_ret_s
            self.q_ret_usd += self.gen.last_ret_usd
            if not self.bypass:
                self.misses += 1
                self.data[key] = {
                    "tail": S.safe_tail(prompt),
                    "out": text,
                    "usage": self.gen.last_ret_usage,
                    "total_s": round(self.gen.last_ret_s, 3),
                }
            return text

        gemini_rest.generate_once = frozen

    def save(self) -> None:
        own = {k: v for k, v in self.data.items() if k not in self.base}
        SPEC._atomic_write_json(self.path, own)


def usd(usage: dict | None, args) -> float:
    return SPEC.usage_usd(usage, args.in_usd, args.out_usd, args.cached_share)


async def answer(ai_core, gen: DrawStream, question: str, pages: list) -> tuple[str, float | None]:
    """Επιστρέφει (κείμενο απάντησης, χρόνος ως το 1ο ορατό κείμενο από την αρχή του ask_ai)."""
    out = ""
    t0 = time.perf_counter()
    first_visible_s: float | None = None
    gen.active = True
    try:
        async for ev in ai_core.ask_ai(question, None, user_id=E.TEST_USER, precomputed=pages):
            if ev.get("type") == "text":
                chunk = ev.get("data", "")
                if first_visible_s is None and chunk:
                    first_visible_s = time.perf_counter() - t0
                out += chunk
    finally:
        gen.active = False
    return out, first_visible_s


def load_live_cache(path: str = LIVE_PATH) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


async def run_live_timing(
    ai_core,
    gen: DrawStream,
    ret: RetrievalCalls,
    budget: SPEC.BudgetLedger,
    live_items: list[dict],
    doc_id: dict[str, int],
    load_full_fresh,
    live_cache: dict,
    live_path: str = LIVE_PATH,
) -> tuple[list[dict], bool]:
    """Ζωντανή μέτρηση end-to-end χρόνου αναμονής χρήστη (έναρξη ερώτησης -> 1ο ορατό κείμενο)
    χωρίς πάγωμα μετάφρασης/ανάκτησης, αποθηκευμένη για resume."""
    results: list[dict] = []
    if not live_items:
        return results, True
    print(f"\n===== ΖΩΝΤΑΝΗ ΧΡΟΝΟΜΕΤΡΗΣΗ ({len(live_items)} ερωτήσεις, el+en ανά μέγεθος) =====", flush=True)
    for idx, it in enumerate(live_items, 1):
        t, q = it["t"], it["t"]["question"]
        docs_s = ";".join(it["docs"])
        lkey = f"{it['arm']}|{t['id']}|{docs_s}"
        if lkey in live_cache:
            rec = live_cache[lkey]
            results.append(rec)
            print(f"  [live {idx}/{len(live_items)}] {it['size']}α {t['id']:<5} ({rec['lang']}) "
                  f"RAG {rec['rag_e2e_ttft']}s · FULL {rec['full_e2e_ttft']}s  [παγ.]", flush=True)
            continue

        # 1) RAG ζωντανά από την αρχή της ερώτησης (καθαρό translation cache + bypass retrieval freeze)
        ai_core._translation_cache.clear()
        ret.reset_question_stats()
        ret.bypass = True
        gen.live_mode = True
        gen.last = None
        n_gen_err = len(gen.errors)
        n_ret_err = len(ret.errors)
        try:
            t_start = time.perf_counter()
            rag_pages = await ai_core.search_documents(
                q, None, user_id=E.TEST_USER, target_doc_ids=[doc_id[d] for d in it["docs"]]
            )
            t_search = time.perf_counter() - t_start
            if budget.exceeded or len(ret.errors) > n_ret_err:
                return results, False
            _rag_ans, vis_after_search = await answer(ai_core, gen, q, rag_pages)
            rag_e2e = t_search + (vis_after_search or 0.0)
            if budget.exceeded or len(gen.errors) > n_gen_err:
                return results, False
            rag_gen = gen.last or {}
            rag_cut = int(not rag_pages)
            rag_ret_calls = ret.q_live
            rag_ret_api_s = round(ret.q_live_s, 3)
            rag_usd = round(ret.q_ret_usd + float(rag_gen.get("usd") or 0.0), 6)

            # 2) Full ζωντανά από την αρχή της ερώτησης (άμεση φόρτωση σελίδων χωρίς full_cache)
            gen.last = None
            t_start_f = time.perf_counter()
            full_pages_live = load_full_fresh(it["docs"])
            t_prep = time.perf_counter() - t_start_f
            _full_ans, vis_after_prep = await answer(ai_core, gen, q, full_pages_live)
            full_e2e = t_prep + (vis_after_prep or 0.0)
            if budget.exceeded or len(gen.errors) > n_gen_err:
                return results, False
            full_gen = gen.last or {}
            full_usd = round(float(full_gen.get("usd") or 0.0), 6)
        finally:
            ret.bypass = False
            gen.live_mode = False

        rec = {
            "arm": it["arm"], "set": it["set"], "id": t["id"], "size": it["size"],
            "lang": S.lang_of(t, q), "docs": docs_s,
            "rag_e2e_ttft": round(rag_e2e, 3), "rag_search_s": round(t_search, 3),
            "rag_gen_ttft": rag_gen.get("ttft"), "rag_cut": rag_cut,
            "rag_ret_calls": rag_ret_calls, "rag_ret_api_s": rag_ret_api_s, "rag_usd": rag_usd,
            "full_e2e_ttft": round(full_e2e, 3), "full_prep_s": round(t_prep, 3),
            "full_gen_ttft": full_gen.get("ttft"), "full_usd": full_usd,
        }
        live_cache[lkey] = rec
        SPEC._atomic_write_json(live_path, live_cache)
        results.append(rec)
        print(f"  [live {idx}/{len(live_items)}] {it['size']}α {t['id']:<5} ({rec['lang']}) "
              f"RAG {rec['rag_e2e_ttft']:.2f}s (search {rec['rag_search_s']:.2f}s, {rag_ret_calls} API) · "
              f"FULL {rec['full_e2e_ttft']:.2f}s · σωρ. {budget.total_usd:.3f} $", flush=True)
    return results, True


async def run(args) -> int:
    os.makedirs(LC_DIR, exist_ok=True)
    ai_core, _saved = E.open_store()
    if ai_core.GEMINI_MODEL != MODEL or E.MODEL != MODEL:
        print(f"!! Μοντέλο {ai_core.GEMINI_MODEL} / {E.MODEL}, όχι {MODEL} — βάλε -e GEMINI_MODEL={MODEL}")
        return 1
    if ai_core.collection.count() != S.EXPECTED_CHUNKS:
        print(f"!! Το store δεν έχει {S.EXPECTED_CHUNKS} chunks — σταματάω")
        return 1
    ai_core._translation_cache.clear()

    budget = SPEC.BudgetLedger(
        BUDGET_PATH,
        max_usd=args.max_usd,
        in_usd=args.in_usd,
        out_usd=args.out_usd,
        cached_share=args.cached_share,
        seed_files=(("gen", GEN_PATH), ("ret", RET_PATH)),
    )
    gen = DrawStream(gemini_rest.stream_generate, budget)
    ret = RetrievalCalls(gen)
    ret.install()
    gemini_rest.stream_generate = gen

    metas = {}
    for m in ai_core._get_bm25_index()["metas"]:
        metas.setdefault((m["file_name"], int(m["page"])), m)
    files = sorted({f for f, _p in metas})
    doc_id = {m["file_name"]: m["doc_id"] for m in metas.values()}
    full_cache: dict = {}
    full_prep_s: dict = {}

    def load_full_fresh(docs: list[str]) -> list:
        return [
            PA.page_of(ai_core, metas[(f, p)])
            for f in docs
            for (ff, p) in sorted(metas)
            if ff == f
        ]

    def full_pages(docs: list[str]) -> tuple[list, float]:
        key = tuple(docs)
        if key not in full_cache:
            t0 = time.perf_counter()
            full_cache[key] = load_full_fresh(docs)
            full_prep_s[key] = time.perf_counter() - t0
        return full_cache[key], full_prep_s[key]

    arms = SPEC.PHASE_ARMS[args.phase]
    safety_sizes = tuple(int(x) for x in args.safety_sizes.split(",") if x.strip())
    if not set(safety_sizes) <= set(SPEC.SAFETY_SIZES):
        print(f"!! --safety-sizes {args.safety_sizes}: επιτρέπονται μόνο 1, 2, 3")
        return 1
    include_distract = not args.no_distract and args.phase != "B"
    items = build_items(files, include_distract=include_distract, arms=arms, safety_sizes=safety_sizes)
    if args.limit:
        # οι πρώτες N ερωτήσεις κάθε σκέλους (σε safety/distract: σε όλα τα μεγέθη τους)
        keep: dict = {}
        for it in items:
            ids = keep.setdefault(it["arm"], [])
            if it["t"]["id"] not in ids and len(ids) < args.limit:
                ids.append(it["t"]["id"])
        items = [it for it in items if it["t"]["id"] in keep[it["arm"]]]

    live_per_size = min(args.live_timing, args.limit) if args.limit else args.live_timing
    live_items = SPEC.select_live_timing_items(items, per_size=live_per_size)
    expected_rows = sum(DRAWS[it["arm"]] * 2 for it in items)
    expected_primary_rows = sum(DRAWS[it["arm"]] * 2 for it in items if it["arm"] in PRIMARY_ARMS)

    print(
        f"\n{len(items)} ερωτήσεις-επιλογές · {expected_rows} γραμμές "
        f"(προκαταχωρισμένες: {expected_primary_rows}"
        + (f", distractor: {expected_rows - expected_primary_rows}" if include_distract else "")
        + f") · ζωντανό δείγμα: {len(live_items)} · άρθρα: {len(files)} · "
        f"σωρευτικό όριο {args.max_usd:.2f} $ (ήδη {budget.total_usd:.4f} $)\n",
        flush=True,
    )

    out_path = os.path.join(LC_DIR, "answers_smoke.json" if args.limit else OUT_NAME[args.phase])
    args.arms, args.safety_list = arms, safety_sizes        # για το meta (validate_answers)
    rows: list[dict] = []
    live_rows: list[dict] = []
    seen_doc_sys: set[tuple[str, str, str]] = set()
    aborted_reason: str | None = None
    step = 0

    # Ομαδοποίηση κατά (arm, size, docs): μέσα σε κάθε ομάδα τρέχει ΠΡΩΤΑ όλη η κλήρωση 1
    # (διαφορετικές διαδοχικές ερωτήσεις στα ίδια PDF -> ρεαλιστικό cache) και ΜΕΤΑ η κλήρωση 2 (repeat).
    groups = [list(g) for _k, g in groupby(items, key=lambda it: (it["arm"], it["size"], tuple(it["docs"])))]

    for grp in groups:
        if aborted_reason:
            break
        arm = grp[0]["arm"]
        rag_by_idx: dict[int, tuple[list, float, dict]] = {}
        for draw in range(1, DRAWS[arm] + 1):
            if aborted_reason:
                break
            gen.draw = draw
            for idx_in_grp, it in enumerate(grp):
                t, q = it["t"], it["t"]["question"]
                docs_s = ";".join(it["docs"])
                if idx_in_grp not in rag_by_idx:
                    ai_core._translation_cache.clear()
                    ret.reset_question_stats()
                    n_ret_err = len(ret.errors)
                    t0 = time.perf_counter()
                    rag_pages = await ai_core.search_documents(
                        q, None, user_id=E.TEST_USER, target_doc_ids=[doc_id[d] for d in it["docs"]]
                    )
                    search_s = time.perf_counter() - t0
                    if budget.exceeded:
                        aborted_reason = budget.exceeded
                        break
                    if len(ret.errors) > n_ret_err:
                        aborted_reason = f"{t['id']}: {ret.errors[-1]}"
                        break
                    rag_by_idx[idx_in_grp] = (
                        rag_pages,
                        search_s,
                        {
                            "live": ret.q_live,
                            "frozen": ret.q_frozen,
                            "live_s": round(ret.q_live_s, 3),
                            "ret_usd": round(ret.q_ret_usd, 6),
                        },
                    )

                rag_pages, search_s, ret_info = rag_by_idx[idx_in_grp]
                f_pages, f_prep_s = full_pages(it["docs"])
                pages = {"rag": rag_pages, "full": f_pages}

                for system in ("full", "rag"):
                    n_err = len(gen.errors)
                    gen.last = None
                    gen.last_fresh = False
                    ans, _vis_s = await answer(ai_core, gen, q, pages[system])
                    if budget.exceeded:
                        aborted_reason = budget.exceeded
                        break
                    if len(gen.errors) > n_err:
                        aborted_reason = f"{t['id']} {system}: {gen.errors[-1]}"
                        break
                    cut = not pages[system]
                    entry = gen.last or {}
                    usage = entry.get("usage") or {}
                    p, c_out, th = SPEC.usage_tokens(usage) if usage else (None, None, None)
                    cost = usd(usage, args) if usage else 0.0
                    if cut:
                        cache_role = "cut"
                    elif draw > 1:
                        cache_role = "repeat"
                    elif (arm, docs_s, system) not in seen_doc_sys:
                        cache_role = "first"
                        seen_doc_sys.add((arm, docs_s, system))
                    else:
                        cache_role = "next"

                    ttft = entry.get("ttft")
                    if system == "rag":
                        e2e_ttft = round(search_s + (ttft or 0.0), 3) if (cut or ttft is not None) else None
                        e2e_live = int(ret_info["frozen"] == 0 and (cut or bool(gen.last_fresh)))
                    else:
                        e2e_ttft = round(f_prep_s + ttft, 3) if ttft is not None else None
                        e2e_live = int(bool(gen.last_fresh))

                    step += 1
                    row = {
                        "arm": arm, "set": it["set"], "id": t["id"], "size": it["size"],
                        "lang": S.lang_of(t, q), "docs": docs_s, "draw": draw, "system": system,
                        "cache_role": cache_role,
                        "cut": int(cut),
                        "second_search": int(any(m.get("second_search") for _x, m in pages[system])),
                        "n_pages": len(pages[system]),
                        "pages": ";".join(f"{m.get('file_name')}:{m.get('page')}" for _x, m in pages[system])
                                 if system == "rag" else "",
                        "in_tokens": p, "cached_tokens": int(usage.get("cachedContentTokenCount") or 0),
                        "out_tokens": (c_out or 0) + (th or 0) if usage else None,
                        "usd": round(cost, 6), "fresh": int(gen.last_fresh and not cut),
                        "finish": entry.get("finish"),
                        "ttft": ttft, "total_s": entry.get("total_s"),
                        "search_s": round(search_s, 3) if system == "rag" else None,
                        "prep_s": round(f_prep_s, 4) if system == "full" else None,
                        "e2e_ttft": e2e_ttft, "e2e_live": e2e_live,
                        "ret_live_calls": ret_info["live"] if system == "rag" else 0,
                        "ret_frozen_calls": ret_info["frozen"] if system == "rag" else 0,
                        "ret_live_s": ret_info["live_s"] if system == "rag" else 0.0,
                        "ret_usd": ret_info["ret_usd"] if system == "rag" else 0.0,
                        "refusal_re": int(bool(E.REFUSAL.search(ans))),
                        "answer": ans,
                    }
                    if it.get("target_doc"):
                        row["target_doc"] = it["target_doc"]
                    rows.append(row)
                    print(
                        f"  [{step}/{expected_rows}] {arm:<8} {t['id']:<5} {it['size']}α κλ.{draw} "
                        f"{system:<4} {cache_role:<6} "
                        f"{'ΣΙΩΠΗ ' if cut else ''}in {row['in_tokens']} (cache {row['cached_tokens']}) "
                        f"out {row['out_tokens']} · {row['usd']:.4f} $ · TTFT {row['ttft']} "
                        f"{'νέα' if row['fresh'] else 'παγ.'} · σωρ. {budget.total_usd:.3f} $",
                        flush=True,
                    )
                if aborted_reason:
                    break
                gen.save()
                ret.save()
                save_rows(
                    out_path, rows, live_rows, args, budget, gen, ret,
                    complete=False, expected_rows=expected_rows,
                    expected_primary_rows=expected_primary_rows,
                    include_distract=include_distract,
                    live_target=len(live_items),
                )

    live_ok = True
    if not aborted_reason and live_items:
        live_cache = load_live_cache()
        live_rows, live_ok = await run_live_timing(
            ai_core, gen, ret, budget, live_items, doc_id, load_full_fresh, live_cache
        )
        if not live_ok:
            aborted_reason = budget.exceeded or (gen.errors[-1] if gen.errors else None) or (
                ret.errors[-1] if ret.errors else "αποτυχία ζωντανής χρονομέτρησης"
            )

    primary_done = sum(1 for r in rows if r["arm"] in PRIMARY_ARMS)
    complete = (
        aborted_reason is None
        and len(rows) == expected_rows
        and primary_done == expected_primary_rows
        and live_ok
        and len(live_rows) == len(live_items)
    )
    gen.save()
    ret.save()
    budget.save()
    save_rows(
        out_path, rows, live_rows, args, budget, gen, ret,
        complete=complete, expected_rows=expected_rows,
        expected_primary_rows=expected_primary_rows,
        include_distract=include_distract,
        live_target=len(live_items),
        aborted_reason=aborted_reason,
    )
    summary(rows, live_rows, budget, gen, ret)

    if not complete:
        print(f"!! ΗΜΙΤΕΛΕΣ ΤΡΕΞΙΜΟ ({len(rows)}/{expected_rows} γραμμές): {aborted_reason}")
        print(f"   Ό,τι πληρώθηκε σώθηκε ({out_path}, budget.json: {budget.total_usd:.4f} $). "
              f"Ξανατρέξε (με μεγαλύτερο --max-usd αν χρειάζεται) για να συνεχίσει χωρίς διπλοχρέωση.")
        return 1

    print(
        f"→ {out_path}\nΕπόμενο: python backend/evaluation/judge_lc_vs_rag.py"
        + (" --smoke" if args.limit else "")
        + "  (στον υπολογιστή, από τη ρίζα)"
    )
    return 0


def save_rows(
    path, rows, live_rows, args, budget: SPEC.BudgetLedger, gen, ret, *,
    complete: bool, expected_rows: int, expected_primary_rows: int,
    include_distract: bool, live_target: int, aborted_reason: str | None = None,
) -> None:
    meta = {
        "model": MODEL, "seed": SEED, "draws": DRAWS, "in_usd": args.in_usd, "out_usd": args.out_usd,
        "cached_share": args.cached_share,
        "complete": complete,
        "aborted_reason": aborted_reason,
        "include_distract": include_distract,
        "phase": getattr(args, "phase", "all"),
        "arms": list(getattr(args, "arms", PRIMARY_ARMS)),
        "safety_sizes": list(getattr(args, "safety_list", SPEC.SAFETY_SIZES)),
        "expected_rows": expected_rows,
        "expected_primary_rows": expected_primary_rows,
        "actual_rows": len(rows),
        "live_timing_target": live_target,
        "live_timing_done": len(live_rows),
        "spent_usd": round(budget.total_usd, 4),
        "run_usd": round(budget.run_usd, 4),
        "budget": budget.snapshot(),
        "gen_new": gen.new, "gen_frozen": gen.hits, "ret_new": ret.misses, "ret_frozen": ret.hits,
    }
    SPEC._atomic_write_json(path, {"meta": meta, "rows": rows, "live_timing": live_rows})


def summary(rows, live_rows, budget: SPEC.BudgetLedger, gen, ret) -> None:
    med = statistics.median
    print("\n" + "#" * 96)
    print(f"{'σκέλος · σύστημα':<24}{'κλήσεις':>8}{'σιωπές':>8}{'tokens (διάμ.)':>16}{'cache':>8}"
          f"{'$ (διάμ.)':>11}{'TTFT διάμ./p90':>17}")
    for arm in ARMS:
        for system in ("rag", "full"):
            rs = [r for r in rows if r["arm"] == arm and r["system"] == system]
            called = [r for r in rs if not r["cut"]]
            if not rs:
                continue
            tok = [r["in_tokens"] for r in called if r["in_tokens"]]
            share = sum(r["cached_tokens"] for r in called) / max(1, sum(tok))
            tt = sorted(r["ttft"] for r in called if r["ttft"] is not None)
            p90 = tt[min(len(tt) - 1, int(0.9 * len(tt)))] if tt else None
            print(f"{arm + ' · ' + system:<24}{len(rs):>8}{len(rs) - len(called):>8}"
                  f"{(med(tok) if tok else 0):>16,.0f}{share:>8.0%}"
                  f"{(med(r['usd'] for r in called) if called else 0):>11.4f}"
                  f"{(f'{med(tt):.2f} / {p90:.2f}' if tt else '-'):>17}")
    if live_rows:
        print("\nΖωντανό δείγμα χρόνου χρήστη (έναρξη ερώτησης -> 1ο ορατό κείμενο):")
        for s in (1, 2, 3):
            sub = [r for r in live_rows if r["size"] == s]
            if not sub:
                continue
            r_tt = sorted(r["rag_e2e_ttft"] for r in sub)
            f_tt = sorted(r["full_e2e_ttft"] for r in sub)
            print(f"  {s} άρθρ{'ο' if s == 1 else 'α'} (n={len(sub)}): "
                  f"RAG διάμ. {med(r_tt):.2f}s / max {max(r_tt):.2f}s · "
                  f"FULL διάμ. {med(f_tt):.2f}s / max {max(f_tt):.2f}s")
    b = budget.snapshot()
    print(f"\nGemini: γέννηση {gen.new} νέες / {gen.hits} παγωμένες · αναζήτηση {ret.misses} νέες / "
          f"{ret.hits} παγωμένες")
    print(f"Δαπάνη: τρέξιμο {b['run_usd']:.4f} $ · σωρευτικά {b['total_usd']:.4f} / {b['max_usd']:.2f} $ "
          f"(γέννηση {b['gen_usd']:.4f} · ανάκτηση {b['ret_usd']:.4f} · ζωντανό {b['live_usd']:.4f})")
    print("#" * 96)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="N ερωτήσεις ανά σκέλος (δοκιμή -> answers_smoke.json)")
    ap.add_argument("--max-usd", type=float, default=8.0, help="σωρευτικό πλαφόν κόστους Gemini (διατηρείται στο resume)")
    ap.add_argument("--in-usd", type=float, default=0.75, help="$ ανά 1M tokens εισόδου (3.8 Flash ως 31/12)")
    ap.add_argument("--out-usd", type=float, default=3.75, help="$ ανά 1M tokens εξόδου (μαζί με thinking)")
    ap.add_argument("--cached-share", type=float, default=0.10, help="τιμή των tokens από το cache, ως μέρος της εισόδου")
    ap.add_argument("--live-timing", type=int, default=4, help="ζωντανές ερωτήσεις ανά μέγεθος για πραγματικό E2E χρόνο (0=χωρίς)")
    ap.add_argument("--no-distract", action="store_true", help="παραλείπει το χωριστό σκέλος distractor")
    ap.add_argument("--phase", choices=("A", "B", "all"), default="all",
                    help="A = ποιότητα/κόστος/χρόνος (one, two, three) · B = ασφάλεια · all = όλα μαζί")
    ap.add_argument("--safety-sizes", default="1,2,3",
                    help="Φάση Β: μεγέθη ασφάλειας, π.χ. 2,3 = μόνο όσα πέρασαν τη Φάση Α")
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

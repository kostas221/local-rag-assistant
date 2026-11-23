"""RAG ή ΟΛΟ το κείμενο των ΕΠΙΛΕΓΜΕΝΩΝ άρθρων; Μέρος 1 από 2: οι απαντήσεις (στο Docker).
Προκαταχώριση: docs/LONGCONTEXT_VS_RAG_PREREG_2026-11-23.md. Μέρος 2 (κρίση GPT-4.1 + πίνακας + κανόνας):
judge_lc_vs_rag.py, στον υπολογιστή.

ΤΙ ΚΑΝΕΙ: για κάθε ερώτηση, με επιλεγμένα ΜΟΝΟ τα άρθρα της, δύο απαντήσεις από τον ΠΡΑΓΜΑΤΙΚΟ ask_ai
(ίδιο prompt, ίδιες ετικέτες [S#], ίδιο μοντέλο):
    rag   search_documents(q, target_doc_ids=επιλεγμένα) -> ask_ai(precomputed=οι σελίδες της αναζήτησης)
    full  ask_ai(precomputed=ΟΛΕΣ οι σελίδες των επιλεγμένων, με τη σειρά) — χωρίς αναζήτηση, χωρίς φύλακα
Οι σελίδες του «full» είναι το ίδιο κείμενο με τις σελίδες της αναζήτησης (probe_decomp_answers.page_of).

ΣΚΕΛΗ (μέγεθος = πόσα άρθρα επιλέγονται):
    one     36 κύριου (main_set_papers.json) + 16 πινάκων (golden_tables, πεδίο doc)        1 άρθρο
    two     29 (golden_multihop_v2)                                                       2 άρθρα
    three   12 (golden_three_papers)                                                      3 άρθρα
    safety  42 κοντινές + 5 άσχετες του κύριου, ΚΑΘΕ ΜΙΑ σε 1, 2 και 3 άρθρα: το άρθρο του κοντινού
            θέματος (source_doc· στις άσχετες ένα τυχαίο, seed 20261123) και τα επόμενα 0/1/2 στην
            αλφαβητική σειρά (κυκλικά) — λίγοι συνδυασμοί, άρα το cache του Gemini δουλεύει όπως σε
            πραγματική χρήση (πολλές ερωτήσεις στα ίδια άρθρα).
Κληρώσεις: 2 (safety: 1). Η αναζήτηση του RAG γίνεται ΜΙΑ φορά ανά ερώτηση· οι κληρώσεις αφορούν τη γέννηση.
Σειρά: ανά σκέλος, ομαδοποιημένα κατά επιλογή άρθρων (ίδιο πρόθεμα prompt -> cache, όπως στην εφαρμογή).

ΠΑΓΩΜΑ (ξανατρέξιμο = 0 κλήσεις· κρατούν απάντηση + ουρά του prompt από την ερώτηση και μετά, ΟΧΙ
κείμενο άρθρων — δημόσιο repo):
    runs/longctx/generation_calls.json   γέννηση, κλειδί κλήρωση + prompt (+ usage, TTFT της πρώτης φοράς)
    runs/longctx/retrieval_calls.json    μετάφραση / αναδιατύπωση / κριτής επάρκειας του RAG· όσες υπάρχουν
                                         ήδη στο πάγωμα του scoreboard ξαναχρησιμοποιούνται (όχι αντιγραφή)
ΚΟΣΤΟΣ: τιμές ανά 1M tokens ως παράμετροι (default 3.8 Flash ως 31/12: 0.75 / 3.75· tokens από το cache του
Gemini στο 10%). --max-usd (default 8): σταματά πριν το ξεπεράσει· ό,τι πληρώθηκε είναι σωσμένο.
ΧΡΟΝΟΣ: TTFT = από την αρχή του ask_ai ως την 1η λέξη, ΜΟΝΟ σε νέες κλήσεις (σώζεται στο πάγωμα). Η
αναζήτηση του RAG μετριέται χωριστά (search_s) και ΔΕΝ μπαίνει στο TTFT.

    # δοκιμή: 2 ερωτήσεις ανά σκέλος -> runs/longctx/answers_smoke.json (κόστος, cache, χρόνος)
    docker compose -f docker-compose.yml -f docker-compose.upgrade.yml run --rm --no-deps \\
        -e DATABASE_URL=sqlite:// -e GEMINI_MODEL=gemini-3.8-flash -v eval_tmp:/tmp backend \\
        python -u evaluation/probe_lc_vs_rag.py --limit 2
    # όλο -> runs/longctx/answers.json
    docker compose -f docker-compose.yml -f docker-compose.upgrade.yml run --rm --no-deps \\
        -e DATABASE_URL=sqlite:// -e GEMINI_MODEL=gemini-3.8-flash -v eval_tmp:/tmp backend \\
        python -u evaluation/probe_lc_vs_rag.py
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
import probe_decomp_answers as PA
import scoreboard as S
import scoreboard_answers as SA

import gemini_rest

HERE = os.path.dirname(os.path.abspath(__file__))
LC_DIR = os.path.join(HERE, "runs", "longctx")
GEN_PATH = os.path.join(LC_DIR, "generation_calls.json")
RET_PATH = os.path.join(LC_DIR, "retrieval_calls.json")
MODEL = "gemini-3.8-flash"
SEED = 20261123
DRAWS = {"one": 2, "two": 2, "three": 2, "safety": 1}
ARMS = tuple(DRAWS)
SIZE = {"one": 1, "two": 2, "three": 3}


def load_jsonl(name: str) -> list[dict]:
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def cyclic(files: list[str], src: str, k: int) -> list[str]:
    i = files.index(src)
    return [files[(i + j) % len(files)] for j in range(k)]


def mh_docs(t: dict) -> list[str]:
    """Τα δύο άρθρα με τη σειρά που τα ονομάζει η ερώτηση (όπως ο κριτής του scoreboard)."""
    return [ep.rsplit(":", 1)[0] for ep in SA.mention_order(t)]


def build_items(files: list[str]) -> list[dict]:
    main = {t["id"]: t for t in load_jsonl("golden_set_50.jsonl")}
    with open(os.path.join(HERE, "main_set_papers.json"), encoding="utf-8") as f:
        papers = json.load(f)["papers"]
    items = []
    for qid, doc in papers.items():
        items.append({"arm": "one", "set": "main", "t": main[qid], "docs": [doc]})
    for t in load_jsonl("golden_tables.jsonl"):
        items.append({"arm": "one", "set": "tables", "t": t, "docs": [t["doc"]]})
    for t in load_jsonl("golden_multihop_v2.jsonl"):
        items.append({"arm": "two", "set": "mh_new", "t": t, "docs": mh_docs(t)})
    for t in load_jsonl("golden_three_papers.jsonl"):
        items.append({"arm": "three", "set": "t3p", "t": t, "docs": list(t["docs"])})
    rng = random.Random(SEED)
    safety = [("near_ooc", t) for t in load_jsonl("golden_near_ooc.jsonl")]
    safety += [("ooc", t) for t in main.values() if t.get("category") == "out_of_corpus"]
    for sk, t in safety:
        src = t.get("source_doc") or rng.choice(files)
        for k in (1, 2, 3):
            items.append({"arm": "safety", "set": sk, "t": t, "docs": cyclic(files, src, k)})
    for it in items:
        it["size"] = len(it["docs"])
        assert all(d in files for d in it["docs"]), (it["t"]["id"], it["docs"])
    # Ανά σκέλος, ομαδοποίηση κατά επιλογή (σταθερή ταξινόμηση: μέσα στην ομάδα, η σειρά του αρχείου)
    return sorted(items, key=lambda it: (ARMS.index(it["arm"]), it["size"], it["docs"]))


class DrawStream:
    """Πάγωμα της ΓΕΝΝΗΣΗΣ με κλειδί κλήρωση + prompt. Κρατά usage, finish και TTFT της πρώτης φοράς."""

    def __init__(self, real):
        self.real = real
        self.data: dict = {}
        if os.path.exists(GEN_PATH):
            with open(GEN_PATH, encoding="utf-8") as f:
                self.data = json.load(f)
        self.draw = 1
        self.new = self.hits = 0
        self.errors: list[str] = []
        self.last: dict | None = None
        self.last_fresh = False

    async def __call__(self, prompt, **kw):
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
            self.errors.append(f"γέννηση: {type(e).__name__}: {e}"[:200])
            raise
        if not text.strip():
            self.errors.append("γέννηση: κενή απάντηση")
            return
        self.new += 1
        entry = {"draw": self.draw, "tail": S.safe_tail(prompt), "out": text, "finish": finish,
                 "usage": usage, "ttft": round(ttft, 3) if ttft else None,
                 "total_s": round(time.perf_counter() - t0, 3)}
        self.data[key] = entry
        self.last, self.last_fresh = entry, True

    def save(self) -> None:
        tmp = GEN_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, GEN_PATH)


class RetrievalCalls(S.Frozen):
    """Το πάγωμα του scoreboard (μόνο ανάγνωση) + δικό μας αρχείο για όσες κλήσεις είναι νέες (με
    επιλεγμένα άρθρα αλλάζει το πεδίο της μετάφρασης, άρα και το prompt της)."""

    def __init__(self):
        super().__init__(RET_PATH, fresh=False)
        self.base = {}
        if os.path.exists(S.FROZEN_PATH):
            with open(S.FROZEN_PATH, encoding="utf-8") as f:
                self.base = json.load(f)
        self.data = {**self.base, **self.data}

    def save(self) -> None:
        own = {k: v for k, v in self.data.items() if k not in self.base}
        tmp = RET_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(own, f, ensure_ascii=False, indent=1)
        os.replace(tmp, RET_PATH)


def usd(usage: dict, args) -> float:
    p, c_out, th = gemini_rest.usage_tokens(usage) if usage else (0, 0, 0)
    cached = usage.get("cachedContentTokenCount") or 0
    return ((p - cached) * args.in_usd + cached * args.in_usd * args.cached_share
            + (c_out + th) * args.out_usd) / 1e6


async def answer(ai_core, question: str, pages: list) -> str:
    out = ""
    async for ev in ai_core.ask_ai(question, None, user_id=E.TEST_USER, precomputed=pages):
        if ev.get("type") == "text":
            out += ev.get("data", "")
    return out


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
    ret = RetrievalCalls()
    ret.install()
    gen = DrawStream(gemini_rest.stream_generate)
    gemini_rest.stream_generate = gen

    metas = {}
    for m in ai_core._get_bm25_index()["metas"]:
        metas.setdefault((m["file_name"], int(m["page"])), m)
    files = sorted({f for f, _p in metas})
    doc_id = {m["file_name"]: m["doc_id"] for m in metas.values()}
    full_cache: dict = {}

    def full_pages(docs: list[str]) -> list:
        key = tuple(docs)
        if key not in full_cache:
            full_cache[key] = [PA.page_of(ai_core, metas[(f, p)]) for f in docs
                               for (ff, p) in sorted(metas) if ff == f]
        return full_cache[key]

    items = build_items(files)
    if args.limit:
        # οι πρώτες N ερωτήσεις κάθε σκέλους (στην ασφάλεια: και στα τρία μεγέθη)
        keep: dict = {}
        for it in items:
            ids = keep.setdefault(it["arm"], [])
            if it["t"]["id"] not in ids and len(ids) < args.limit:
                ids.append(it["t"]["id"])
        items = [it for it in items if it["t"]["id"] in keep[it["arm"]]]
    n_calls = sum(DRAWS[it["arm"]] * 2 for it in items)
    print(f"\n{len(items)} ερωτήσεις-επιλογές · έως {n_calls} γεννήσεις · άρθρα: {len(files)} · "
          f"όριο κόστους {args.max_usd} $\n", flush=True)

    out_path = os.path.join(LC_DIR, "answers_smoke.json" if args.limit else "answers.json")
    rows, spent = [], 0.0
    for n, it in enumerate(items, 1):
        if spent >= args.max_usd:
            print(f"!! ΟΡΙΟ ΚΟΣΤΟΥΣ {args.max_usd} $ — σταματάω (ό,τι πληρώθηκε είναι σωσμένο)")
            break
        t, q = it["t"], it["t"]["question"]
        t0 = time.perf_counter()
        rag_pages = await ai_core.search_documents(q, None, user_id=E.TEST_USER,
                                                   target_doc_ids=[doc_id[d] for d in it["docs"]])
        search_s = time.perf_counter() - t0
        if ret.errors:
            print(f"!! {t['id']}: {ret.errors[-1]} — σταματάω")
            break
        pages = {"rag": rag_pages, "full": full_pages(it["docs"])}
        for draw in range(1, DRAWS[it["arm"]] + 1):
            gen.draw = draw
            for system in ("full", "rag"):
                n_err = len(gen.errors)
                gen.last = None
                ans = await answer(ai_core, q, pages[system])
                if len(gen.errors) > n_err:
                    print(f"!! {t['id']} {system}: {gen.errors[-1]} — σταματάω")
                    gen.save()
                    ret.save()
                    return 1
                cut = not pages[system]
                entry = gen.last or {}
                usage = entry.get("usage") or {}
                p, c_out, th = gemini_rest.usage_tokens(usage) if usage else (None, None, None)
                cost = usd(usage, args) if usage else 0.0
                if gen.last_fresh and not cut:
                    spent += cost
                rows.append({
                    "arm": it["arm"], "set": it["set"], "id": t["id"], "size": it["size"],
                    "lang": S.lang_of(t, q), "docs": ";".join(it["docs"]), "draw": draw, "system": system,
                    "cut": int(cut),
                    "second_search": int(any(m.get("second_search") for _x, m in pages[system])),
                    "n_pages": len(pages[system]),
                    "pages": ";".join(f"{m.get('file_name')}:{m.get('page')}" for _x, m in pages[system])
                             if system == "rag" else "",
                    "in_tokens": p, "cached_tokens": usage.get("cachedContentTokenCount") or 0,
                    "out_tokens": (c_out or 0) + (th or 0) if usage else None,
                    "usd": round(cost, 6), "fresh": int(gen.last_fresh and not cut),
                    "finish": entry.get("finish"),
                    "ttft": entry.get("ttft"), "total_s": entry.get("total_s"),
                    "search_s": round(search_s, 3) if system == "rag" else None,
                    "refusal_re": int(bool(E.REFUSAL.search(ans))),
                    "answer": ans,
                })
                r = rows[-1]
                print(f"  [{n}/{len(items)}] {it['arm']:<6} {t['id']:<5} {it['size']}α κλ.{draw} {system:<4} "
                      f"{'ΣΙΩΠΗ' if cut else ''}in {r['in_tokens']} (cache {r['cached_tokens']}) "
                      f"out {r['out_tokens']} · {r['usd']:.4f} $ · TTFT {r['ttft']} "
                      f"{'νέα' if r['fresh'] else 'παγ.'} · σύνολο {spent:.3f} $", flush=True)
        gen.save()
        ret.save()
        save_rows(out_path, rows, args, spent, gen, ret)

    gen.save()
    ret.save()
    save_rows(out_path, rows, args, spent, gen, ret)
    summary(rows, spent, gen, ret)
    print(f"→ {out_path}\nΕπόμενο: python backend/evaluation/judge_lc_vs_rag.py"
          + (" --smoke" if args.limit else "") + "  (στον υπολογιστή, Git Bash, από τη ρίζα)")
    return 0


def save_rows(path, rows, args, spent, gen, ret) -> None:
    meta = {"model": MODEL, "seed": SEED, "draws": DRAWS, "in_usd": args.in_usd, "out_usd": args.out_usd,
            "cached_share": args.cached_share, "spent_usd": round(spent, 4),
            "gen_new": gen.new, "gen_frozen": gen.hits, "ret_new": ret.misses, "ret_frozen": ret.hits}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": rows}, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def summary(rows, spent, gen, ret) -> None:
    med = statistics.median
    print("\n" + "#" * 90)
    print(f"{'σκέλος · σύστημα':<22}{'κλήσεις':>8}{'σιωπές':>8}{'tokens (διάμ.)':>16}{'cache':>8}"
          f"{'$ (διάμ.)':>11}{'TTFT διάμ./p90':>17}")
    for arm in ARMS:
        for system in ("rag", "full"):
            rs = [r for r in rows if r["arm"] == arm and r["system"] == system]
            called = [r for r in rs if not r["cut"]]
            if not rs:
                continue
            tok = [r["in_tokens"] for r in called if r["in_tokens"]]
            share = sum(r["cached_tokens"] for r in called) / max(1, sum(tok))
            tt = sorted(r["ttft"] for r in called if r["ttft"])
            p90 = tt[min(len(tt) - 1, int(0.9 * len(tt)))] if tt else None
            print(f"{arm + ' · ' + system:<22}{len(rs):>8}{len(rs) - len(called):>8}"
                  f"{(med(tok) if tok else 0):>16,.0f}{share:>8.0%}"
                  f"{(med(r['usd'] for r in called) if called else 0):>11.4f}"
                  f"{(f'{med(tt):.2f} / {p90:.2f}' if tt else '-'):>17}")
    print(f"\nGemini: γέννηση {gen.new} νέες / {gen.hits} παγωμένες · αναζήτηση {ret.misses} νέες / "
          f"{ret.hits} παγωμένες · κόστος τρεξίματος ~{spent:.3f} $")
    print("#" * 90)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="N ερωτήσεις ανά σκέλος (δοκιμή -> answers_smoke.json)")
    ap.add_argument("--max-usd", type=float, default=8.0, help="σταματά πριν ξεπεράσει αυτό το κόστος Gemini")
    ap.add_argument("--in-usd", type=float, default=0.75, help="$ ανά 1M tokens εισόδου (3.8 Flash ως 31/12)")
    ap.add_argument("--out-usd", type=float, default=3.75, help="$ ανά 1M tokens εξόδου (μαζί με thinking)")
    ap.add_argument("--cached-share", type=float, default=0.10, help="τιμή των tokens από το cache, ως μέρος της εισόδου")
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

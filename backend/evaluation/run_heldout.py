"""Βήμα 7: οι απαντήσεις του παγωμένου συστήματος στο κλειστό σετ (Docker).

    # 1. στο Git Bash: έλεγχος ότι τρέχει το c15a395 (γράφει private/preflight.json)
    python backend/evaluation/heldout_preflight.py [--smoke]
    # 2. έλεγχοι, βάση και router χωρίς καμία κλήση API
    docker compose -f docker-compose.yml -f docker-compose.upgrade.yml exec backend \\
        python evaluation/run_heldout.py [--smoke] --dry-run
    # 3. οι απαντήσεις (συνεχίζει από όπου σταμάτησε)
    docker compose -f docker-compose.yml -f docker-compose.upgrade.yml exec backend \\
        python evaluation/run_heldout.py [--smoke]

ΤΙ ΚΑΝΕΙ (docs/HELDOUT_SET_PROTOCOL.md §4, §4α· κανόνες στην προκαταχώριση του βήματος 7)
  · Ελέγχει: αποτυπώματα των αρχείων του συστήματος (ίδια με το preflight), του CSV, των πηγών και των
    PDF, και ότι οι ρυθμίσεις που ισχύουν ΠΡΑΓΜΑΤΙΚΑ είναι οι παγωμένες. Σε κάθε διαφορά σταματά.
  · Βάση: δική της (/tmp, μέσα στο container), με τον κώδικα εισαγωγής της εφαρμογής. Οι caches της
    κανονικής βάσης δεν γράφονται.
  · Τρέχει ΟΠΩΣ Η ΕΦΑΡΜΟΓΗ (main.py -> ai_core.ask_ai, persona Researcher, χωρίς ιστορικό):
      alpha  όλες οι ερωτήσεις, με τσεκαρισμένα ΟΛΑ τα έγγραφα (η προεπιλογή του UI: όλοι οι αριθμοί
             εγγράφων περνούν από τον router, που για 11 άρθρα πάει σε RAG)·
      beta   οι ερωτήσεις δύο άρθρων, με επιλεγμένα ΜΟΝΟ τα δύο άρθρα τους·
      closed οι ερωτήσεις με απάντηση (όχι near), στο ίδιο μοντέλο ΧΩΡΙΣ έγγραφα: τι ξέρει από μνήμη.
  · Για κάθε απάντηση: διαδρομή (φύλακας, δεύτερη αναζήτηση, όλο το κείμενο), σελίδες, αν η σελίδα-
    τεκμήριο ήταν στους υποψηφίους πριν από τον reranker / έφτασε στο μοντέλο / παρατέθηκε, χρόνος,
    tokens, έλεγχος χωρίων του βήματος 6 και αριθμοί που λείπουν από την παρατιθέμενη σελίδα.
  · Σφάλμα API σε ΟΠΟΙΑΔΗΠΟΤΕ κλήση της ερώτησης -> η ερώτηση ξανατρέχει, έως 3 φορές· μετά
    καταγράφεται «τεχνική αποτυχία». Κάθε πραγματική απάντηση μετράει όπως βγήκε (και κομμένη).
  · Σώζει μετά από κάθε ερώτηση: όσες ολοκληρώθηκαν και σώθηκαν δεν ξαναγίνονται. Μια διακοπή ανάμεσα
    στην κλήση και το σώσιμο μπορεί να χρεωθεί δύο φορές.

ΕΞΟΔΟΣ: μόνο id, σκέλος, κατάσταση και πλήθη. Τα κείμενα μόνο στο private/answers.json (εκτός git).
"""
import argparse
import asyncio
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import heldout_common as H  # noqa: E402

TEST_USER = 999_996          # εικονικός ιδιοκτήτης της απομονωμένης βάσης (authz)
DOC_ID_BASE = 7000
STORES = {"heldout": "/tmp/heldout_store", "smoke": "/tmp/heldout_smoke_store"}  # noqa: S108
CLOSED_PROMPT = (
    "Answer the following question from your own knowledge. Answer in the same language as the "
    "question. Be concise: at most 5 sentences. If you do not know the answer, say so.\n\n"
    "Question: {question}"
)


# ------------------------------------------------------------------------------------------ έλεγχοι

def check_files(mode: str, items: list[dict], papers: list[str]) -> list[str]:
    cfg = H.MODES[mode]
    errors = []
    pre_path = os.path.join(H.private_dir(mode), "preflight.json")
    if not os.path.exists(pre_path):
        return [f"λείπει το {pre_path}: τρέξε πρώτα το heldout_preflight.py στο Git Bash"]
    import json
    with open(pre_path, encoding="utf-8") as f:
        pre = json.load(f)
    if pre.get("mode") != mode:
        errors.append(f"το preflight.json είναι για {pre.get('mode')}, όχι για {mode}")
    for rel, sha in (pre.get("system_files") or {}).items():
        path = os.path.join(H.BACKEND, rel)
        if not os.path.exists(path) or H.sha256_file(path) != sha:
            errors.append(f"το αρχείο συστήματος {rel} διαφέρει από το preflight")
    if H.sha256_file(cfg["csv"]) != pre.get("csv_sha"):
        errors.append("το CSV άλλαξε μετά το preflight")
    if mode == "heldout":
        if H.sha256_file(cfg["csv"]) != cfg["csv_sha"]:
            errors.append("το CSV δεν είναι το κλειδωμένο της 19/11")
        if H.sha256_file(cfg["sources"]) != cfg["sources_sha"]:
            errors.append("το heldout_sources.json δεν είναι το κλειδωμένο της 19/11")
        with open(cfg["sources"], encoding="utf-8") as f:
            sources = json.load(f)
        if set(papers) != set(sources):
            errors.append("ο φάκελος των PDF δεν έχει ακριβώς τα 11 άρθρα του heldout_sources.json")
        for name, meta in sources.items():
            path = os.path.join(cfg["papers"], name)
            if not os.path.exists(path):
                errors.append(f"λείπει το PDF {name}")
            elif H.sha256_file(path) != meta["sha256"]:
                errors.append(f"το PDF {name} έχει άλλο SHA-256 από το heldout_sources.json")
    needed = {d for it in items for d in (it["doc1"], it["doc2"]) if d}
    missing = sorted(needed - set(papers))
    if missing:
        errors.append(f"ερωτήσεις σε άρθρα που δεν υπάρχουν: {', '.join(missing)}")
    return errors


def actual_config(ai_core) -> dict:
    return {
        "GEMINI_MODEL": ai_core.GEMINI_MODEL,
        "TEMPERATURE": ai_core.GENERATION_CONFIG.temperature,
        "MAX_OUTPUT_TOKENS": ai_core.GENERATION_CONFIG.max_output_tokens,
        "THINKING_BUDGET": ai_core.THINKING_BUDGET,
        "USE_REST_GENERATION": ai_core.USE_REST_GENERATION,
        "EMBED_MODEL_NAME": ai_core.EMBED_MODEL_NAME,
        "RERANKER_MODEL": ai_core.RERANKER_MODEL,
        "MIN_RERANK_SCORE": ai_core.MIN_RERANK_SCORE,
        "CORRECTIVE_MIN_SCORE": ai_core.CORRECTIVE_MIN_SCORE,
        "ENABLE_CORRECTIVE": ai_core.ENABLE_CORRECTIVE,
        "ENABLE_SUFFICIENCY_VETO": ai_core.ENABLE_SUFFICIENCY_VETO,
        "STRIP_HIDDEN_TEXT": ai_core.STRIP_HIDDEN_TEXT,
        "ENABLE_FULL_TEXT": ai_core.ENABLE_FULL_TEXT,
        "FULL_TEXT_MAX_TOKENS": ai_core.FULL_TEXT_MAX_TOKENS,
        "ENABLE_PERDOC": ai_core.ENABLE_PERDOC,
        "PERDOC_EXTRA": ai_core.PERDOC_EXTRA,
        "RERANK_CANDIDATES": ai_core.RERANK_CANDIDATES,
        "DENSE_CANDIDATES": ai_core.DENSE_CANDIDATES,
        "EXPAND_INPUT": ai_core.EXPAND_INPUT,
        "MAX_PAGES": ai_core.MAX_PAGES,
        "PAGE_SCORE_TOP_K": ai_core.PAGE_SCORE_TOP_K,
        "MAX_PAGES_PER_FILE": ai_core.MAX_PAGES_PER_FILE,
        "USE_BGE_SPARSE": ai_core.USE_BGE_SPARSE,
        "CITATION_RULE_SET": bool(ai_core.CITATION_RULE),
    }


# ------------------------------------------------------------------------------------------ κατάσκοποι

class Spy:
    """Τι έγινε μέσα σε ΜΙΑ ερώτηση: κλήσεις reranker (υποψήφιοι), βοηθητικές κλήσεις, σφάλματα API.
    Μόνο καταγράφει: τα ορίσματα και τα αποτελέσματα περνούν αναλλοίωτα."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.rerank, self.aux, self.api_errors = [], [], []


SPY = Spy()


def _aux_kind(prompt: str) -> str:
    if prompt.startswith("You are a translation assistant"):
        return "translation"
    if prompt.startswith("The following search query"):
        return "corrective"
    if prompt.startswith("Answer the following question from your own knowledge"):
        return "closed_book"
    return "other"          # κριτής επάρκειας, έλεγχος χωρίων


def install_spies(ai_core) -> None:
    import gemini_rest

    orig_predict = ai_core.reranker.predict

    def predict_spy(pairs, **kw):
        scores = orig_predict(pairs, **kw)
        SPY.rerank.append({"max": float(max(scores)) if len(scores) else None,
                           "texts": [p[1] for p in pairs]})
        return scores

    ai_core.reranker.predict = predict_spy

    orig_stream = gemini_rest.stream_generate

    async def stream_spy(prompt, *a, **kw):
        try:
            async for ev in orig_stream(prompt, *a, **kw):
                yield ev
        except Exception as e:      # σφάλμα δικτύου/API μετά τις επαναλήψεις του gemini_rest
            SPY.api_errors.append(type(e).__name__)
            raise

    gemini_rest.stream_generate = stream_spy

    orig_once = gemini_rest.generate_once

    async def once_spy(prompt, **kw):
        SPY.aux.append({"kind": _aux_kind(prompt), "chars": len(prompt)})
        return await orig_once(prompt, **kw)

    gemini_rest.generate_once = once_spy


# -------------------------------------------------------------------------------------------- βάση

def open_store(ai_core, mode: str, papers: list[str], rebuild: bool) -> dict:
    """Απομονωμένη βάση με τα PDF του σετ, με τον κώδικα εισαγωγής της εφαρμογής (ingest_pdf, όπως το
    main.process_document). Επιστρέφει {'doc_ids': αρχείο -> doc_id, 'text_pages': κείμενο chunk ->
    [(αρχείο, σελίδα)], 'chunks': πλήθος}."""
    import chromadb
    path = STORES[mode]
    if rebuild and os.path.exists(path):
        shutil.rmtree(path)
    col = chromadb.PersistentClient(path=path).get_or_create_collection(
        name="heldout", embedding_function=ai_core.sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"})
    ai_core.collection = col
    pdir = H.MODES[mode]["papers"]
    if col.count() == 0:
        # Ο χρόνος εισαγωγής είναι κι αυτός αποτέλεσμα: τόσο περιμένει και ο χρήστης που ανεβάζει το PDF
        # (ίδιος κώδικας). Στη δοκιμή του εργαλείου: 418 chunks σε ~38 λεπτά σε CPU.
        print(f"Χτίσιμο βάσης: {len(papers)} PDF (σε CPU μπορεί να πάρει ώρα, 0 κλήσεις API)...", flush=True)
        build = []
        for i, name in enumerate(papers):
            report = {}
            t0 = time.perf_counter()
            ok = ai_core.ingest_pdf(os.path.join(pdir, name), name, TEST_USER,
                                    doc_id=DOC_ID_BASE + i, report=report)
            sec = round(time.perf_counter() - t0, 1)
            build.append({"file": name, "ok": ok, "pages_text": report.get("pages_text"),
                          "pages_total": report.get("pages_total"), "seconds": sec})
            print(f"  {name}: {'OK' if ok else 'ΚΕΝΟ'} · σελίδες με κείμενο "
                  f"{report.get('pages_text', '?')}/{report.get('pages_total', '?')} · {sec:.0f} s", flush=True)
        H.write_json_atomic(os.path.join(H.MODES[mode]["out"], "store_build.json"),
                            {"when": time.strftime("%Y-%m-%d %H:%M"), "docs": build})
    ai_core._bump_corpus_version()
    got = col.get(include=["documents", "metadatas"])
    doc_ids, text_pages = {}, {}
    for text, meta in zip(got["documents"], got["metadatas"]):
        doc_ids.setdefault(meta["file_name"], meta["doc_id"])
        text_pages.setdefault(text, set()).add((meta["file_name"], int(meta["page"])))
    expected = {name: DOC_ID_BASE + i for i, name in enumerate(papers)}
    if doc_ids != expected:
        raise SystemExit("η βάση δεν έχει τα αναμενόμενα έγγραφα/doc_id: τρέξε με --rebuild-store")
    return {"doc_ids": doc_ids, "text_pages": text_pages, "chunks": len(got["ids"])}


# ------------------------------------------------------------------------------------------ διαδρομές

def targets(item: dict, arm: str, store: dict) -> tuple[list[str], list[int]]:
    """Ό,τι στέλνει το UI: alpha = όλα τα τσεκαρισμένα (προεπιλογή), beta = μόνο τα δύο άρθρα."""
    files = sorted(store["doc_ids"]) if arm == "alpha" else [item["doc1"], item["doc2"]]
    return files, [store["doc_ids"][f] for f in files]


def router_check(ai_core, mode: str, items: list[dict], store: dict) -> list[str]:
    """Χωρίς κλήσεις API: η alpha πρέπει να πηγαίνει σε RAG (όλα τα έγγραφα δεν χωράνε) και η beta στο
    όλο το κείμενο (και τα 12 ζεύγη χωράνε στα 70k, βλ. προκαταχώριση). Στο smoke μόνο αναφέρεται."""
    errors = []
    files, ids = targets(items[0], "alpha", store)
    if ai_core._full_text_pages(files, TEST_USER, ids):
        errors.append("alpha: όλα τα έγγραφα πάνε στο όλο το κείμενο (αναμενόταν RAG)")
    pairs = [it for it in items if it["type"] == "two_papers"]
    full = 0
    for it in pairs:
        files, ids = targets(it, "beta", store)
        full += bool(ai_core._full_text_pages(files, TEST_USER, ids))
    print(f"Router: alpha -> RAG · beta -> όλο το κείμενο σε {full}/{len(pairs)} ζεύγη")
    if mode == "heldout" and full != len(pairs):
        errors.append(f"beta: {len(pairs) - full} ζεύγη δεν πάνε στο όλο το κείμενο (αναμενόταν 0)")
    return errors


def evidence(item: dict, sources: list[dict], cited: list[str], store: dict) -> list[dict]:
    """Για κάθε σελίδα-τεκμήριο: ήταν στους υποψηφίους του 1ου περάσματος του reranker; σε οποιοδήποτε
    πέρασμα; έφτασε στο μοντέλο; παρατέθηκε; (None όταν δεν υπάρχει reranker, π.χ. όλο το κείμενο.)"""
    wanted = [(item["doc1"], item["page1"])]
    if item["type"] == "two_papers":
        wanted.append((item["doc2"], item["page2"]))

    def pages_of(texts):
        return {fp for t in texts for fp in store["text_pages"].get(t, ())}

    first = pages_of(SPY.rerank[0]["texts"]) if SPY.rerank else None
    anyp = pages_of(t for call in SPY.rerank for t in call["texts"]) if SPY.rerank else None
    final = {(s["file"], s["page"]) for s in sources}
    cset = set(cited)
    cited_pages = {(s["file"], s["page"]) for s in sources if s["label"] in cset}
    out = []
    for f, p in wanted:
        if not f or p is None:
            out.append(None)
            continue
        out.append({"file": f, "page": p,
                    "in_candidates_first": None if first is None else (f, p) in first,
                    "in_candidates_any": None if anyp is None else (f, p) in anyp,
                    "in_final": (f, p) in final, "cited": (f, p) in cited_pages})
    return out


async def passage_check(ai_core, item: dict, answer: str, sources: list[dict], cited: list[str]) -> dict:
    """Ο έλεγχος χωρίων της εφαρμογής (βήμα 6, get_source_passages, αμετάβλητος) και, χωρίς κλήση,
    οι αριθμοί κάθε πρότασης που λείπουν από τις σελίδες που παραθέτει (claim_support, βήμα 5)."""
    import claim_support as CS
    import source_passages as SP
    last = ""
    for attempt in range(H.MAX_ATTEMPTS):
        try:
            res = await ai_core.get_source_passages(item["question"], answer, sources, TEST_USER)
            srcs = [s for s in sources if s.get("label") in set(cited)]
            texts = await asyncio.to_thread(ai_core._page_texts_for, srcs, TEST_USER)
            claims, _prompt = SP.prepare(item["question"], answer, texts)
            allowed = CS.numbers_in(item["question"])
            missing = []
            for c in claims:
                if c.kind == "claim" and c.labels:
                    cited_texts = [texts[lab] for lab in c.labels if lab in texts]
                    nums = CS.unsupported_numbers(c.text, cited_texts, allowed) if cited_texts else []
                    if nums:
                        missing.append({"i": c.i, "numbers": nums})
            return {"status": res.get("status"), "items": res.get("items", []),
                    "numbers_missing": missing, "attempts": attempt + 1}
        except Exception as e:
            last = type(e).__name__
            await asyncio.sleep(15)
    return {"status": "failed", "error": last, "items": [], "numbers_missing": [],
            "attempts": H.MAX_ATTEMPTS}


async def answer_once(ai_core, item: dict, arm: str, store: dict) -> dict:
    files, ids = targets(item, arm, store)
    SPY.reset()
    t0 = time.perf_counter()
    answer, sources, metrics, statuses = "", [], {}, []
    async for ev in ai_core.ask_ai(item["question"], target_filenames=files, history=[],
                                   user_id=TEST_USER, persona="Researcher", target_doc_ids=ids):
        kind = ev.get("type")
        if kind == "sources":
            sources = ev.get("data") or []
        elif kind == "text":
            answer += ev.get("data", "")
        elif kind == "metrics":
            metrics = ev.get("data") or {}
        elif kind == "status":
            statuses.append(ev.get("reason"))
    wall = time.perf_counter() - t0
    technical = bool(SPY.api_errors) or any(t in answer for t in H.API_ERROR_TEXTS)
    srcs = [{"label": s.get("label"), "file": s.get("file"), "page": s.get("page"),
             "full_text": bool(s.get("full_text")), "second_search": bool(s.get("second_search"))}
            for s in sources]
    import source_passages as SP
    cited = SP.cited_labels(answer)
    if any(s["full_text"] for s in srcs):
        outcome = "full_text"
    elif not srcs:
        outcome = "cut"
    elif SPY.rerank and SPY.rerank[0]["max"] is not None and SPY.rerank[0]["max"] >= ai_core.MIN_RERANK_SCORE:
        outcome = "passed_gate"
    else:
        outcome = "passed_corrective"
    own = {item["doc1"], item["doc2"]} - {None}
    cited_files = sorted({s["file"] for s in srcs if s["label"] in set(cited)})
    return {
        "technical": technical, "api_errors": list(SPY.api_errors), "answer": answer,
        "translation": ai_core._translation_cache.get(item["question"], ""),
        "sources": srcs, "cited": cited, "cited_files": cited_files,
        "other_papers_cited": [f for f in cited_files if f not in own],
        "outcome": outcome, "best1": SPY.rerank[0]["max"] if SPY.rerank else None,
        "n_rerank_calls": len(SPY.rerank), "n_pages": len(srcs),
        "full_text": outcome == "full_text",
        "second_search": any(s["second_search"] for s in srcs),
        "evidence": evidence(item, srcs, cited, store),
        "finish_reason": metrics.get("finish_reason"), "incomplete": bool(statuses),
        "prompt_tokens": metrics.get("prompt_tokens"), "completion_tokens": metrics.get("completion_tokens"),
        "thinking_tokens": metrics.get("thinking_tokens"),
        "retrieval_s": metrics.get("retrieval_s"), "generation_s": metrics.get("generation_s"),
        "wall_s": round(wall, 2),
        "aux": [dict(a) for a in SPY.aux],
    }


async def closed_once(ai_core, item: dict) -> dict:
    import gemini_rest
    SPY.reset()
    t0 = time.perf_counter()
    answer, technical, err = "", False, ""
    try:
        answer = await gemini_rest.generate_once(
            CLOSED_PROMPT.format(question=item["question"]), model=ai_core.GEMINI_MODEL,
            api_key=ai_core.GEMINI_API_KEY, temperature=ai_core.GENERATION_CONFIG.temperature,
            max_output_tokens=ai_core.GENERATION_CONFIG.max_output_tokens,
            thinking_budget=ai_core.THINKING_BUDGET)
    except Exception as e:
        technical, err = True, type(e).__name__
    return {"technical": technical or bool(SPY.api_errors), "api_errors": list(SPY.api_errors) + ([err] if err else []),
            "answer": answer, "wall_s": round(time.perf_counter() - t0, 2), "aux": [dict(a) for a in SPY.aux],
            "outcome": "closed_book", "sources": [], "cited": []}


def est_usd(rec: dict, in_usd: float, out_usd: float) -> float:
    """ΕΚΤΙΜΗΣΗ κόστους (για το όριο ασφαλείας, όχι για αναφορά): tokens της απάντησης από το usage, οι
    βοηθητικές κλήσεις από το μήκος του prompt (~3,8 χαρακτήρες/token) και ~400 tokens εξόδου η καθεμία."""
    aux = rec.get("aux", []) + rec.get("passage_aux", [])
    tin = (rec.get("prompt_tokens") or 0) + sum(a["chars"] for a in aux) / 3.8
    tout = (rec.get("completion_tokens") or 0) + (rec.get("thinking_tokens") or 0) + 400 * len(aux)
    return (tin * in_usd + tout * out_usd) / 1e6


# -------------------------------------------------------------------------------------------- κύριο

async def run(args) -> int:
    mode = "smoke" if args.smoke else "heldout"
    cfg = H.MODES[mode]
    items = H.load_items(cfg["csv"])
    papers = sorted(f for f in os.listdir(cfg["papers"]) if f.lower().endswith(".pdf"))
    errors = check_files(mode, items, papers)
    if errors:
        print("ΣΤΑΜΑΤΑ:")
        for e in errors:
            print("  ✗", e)
        return 1
    print(f"✓ {mode}: αρχεία συστήματος ίδια με το preflight · {len(items)} ερωτήσεις · {len(papers)} PDF")

    print("Φόρτωση μοντέλων (40-60 s χωρίς έξοδο)...", flush=True)
    import ai_core
    actual = actual_config(ai_core)
    diff = {k: (v, actual.get(k)) for k, v in H.FROZEN_CONFIG.items() if actual.get(k) != v}
    if diff:
        print("ΣΤΑΜΑΤΑ: οι ρυθμίσεις διαφέρουν από το παγωμένο σύστημα (μάλλον από το .env):")
        for k, (want, got) in diff.items():
            print(f"  ✗ {k}: αναμενόταν {want!r}, ισχύει {got!r}")
        return 1
    print("✓ ρυθμίσεις ίδιες με το παγωμένο σύστημα")
    ai_core._save_translation_cache = lambda: None      # η κανονική βάση δεν γράφεται
    ai_core._save_query_emb_cache = lambda: None

    store = open_store(ai_core, mode, papers, args.rebuild_store)
    print(f"✓ βάση: {store['chunks']} chunks σε {len(store['doc_ids'])} έγγραφα")
    errors = router_check(ai_core, mode, items, store)
    if errors:
        print("ΣΤΑΜΑΤΑ:")
        for e in errors:
            print("  ✗", e)
        return 1
    if args.dry_run:
        print("Dry run: κανένα σφάλμα, καμία κλήση API.")
        return 0

    install_spies(ai_core)
    out_path = os.path.join(H.private_dir(mode), "answers.json")
    import json
    data = {"meta": {}, "rows": {}}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
    data["meta"] = {**data.get("meta", {}), "mode": mode, "config": actual, "test_user": TEST_USER,
                    "doc_ids": store["doc_ids"], "chunks": store["chunks"],
                    "closed_prompt": CLOSED_PROMPT, "last_run": time.strftime("%Y-%m-%d %H:%M")}
    data["meta"].setdefault("started", time.strftime("%Y-%m-%d %H:%M"))

    plan = []
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        for it in items:
            if arm == "beta" and it["type"] != "two_papers":
                continue
            if arm == "closed" and it["type"] == "near":
                continue
            plan.append((arm, it))
    spent = sum(r.get("est_usd", 0.0) for r in data["rows"].values())
    todo = [(a, it) for a, it in plan if f"{a}:{it['id']}" not in data["rows"]]
    print(f"Σχέδιο: {len(plan)} απαντήσεις · έτοιμες {len(plan) - len(todo)} · μένουν {len(todo)} · "
          f"εκτίμηση κόστους ως τώρα {spent:.2f} $ (όριο {args.max_usd} $)")

    for n, (arm, it) in enumerate(todo, 1):
        if spent >= args.max_usd:
            print(f"ΣΤΑΜΑΤΑ: έφτασε το όριο κόστους ({spent:.2f} $). Ό,τι έγινε σώθηκε.")
            break
        key = f"{arm}:{it['id']}"
        rec, attempts = None, 0
        for attempts in range(1, H.MAX_ATTEMPTS + 1):
            rec = await (closed_once(ai_core, it) if arm == "closed" else answer_once(ai_core, it, arm, store))
            spent += est_usd(rec, args.in_usd, args.out_usd)
            if not rec["technical"]:
                break
            print(f"  {key}: σφάλμα API ({', '.join(rec['api_errors']) or 'κείμενο σφάλματος'}), "
                  f"προσπάθεια {attempts}/{H.MAX_ATTEMPTS}", flush=True)
            await asyncio.sleep(30)
        rec["status"] = "technical_failure" if rec["technical"] else "ok"
        rec["attempts"] = attempts
        if rec["status"] == "ok" and arm != "closed":
            n_aux = len(SPY.aux)
            rec["passages"] = await passage_check(ai_core, it, rec["answer"], rec["sources"], rec["cited"])
            rec["passage_aux"] = [dict(a) for a in SPY.aux[n_aux:]]
            spent += est_usd({"passage_aux": rec["passage_aux"]}, args.in_usd, args.out_usd)
        rec.update({"key": key, "arm": arm, "id": it["id"], "type": it["type"], "lang": it["lang"],
                    "est_usd": round(est_usd(rec, args.in_usd, args.out_usd), 5)})
        data["rows"][key] = rec
        H.write_json_atomic(out_path, data)
        ev = "".join("·" if e is None else ("Τ" if e["in_final"] else "-") for e in rec.get("evidence") or [])
        ps = (rec.get("passages") or {}).get("status", "")
        print(f"[{n}/{len(todo)}] {arm:<6} {it['id']:<6} {rec['status']:<17} {rec.get('outcome', ''):<17} "
              f"σελ {rec.get('n_pages', 0):>2} τεκμ {ev or '·'} παραπ {len(rec.get('cited') or [])} "
              f"χωρία {ps:<8} {rec['wall_s']:>5.1f}s ~{spent:.2f}$", flush=True)

    rows = data["rows"]
    done = [r for r in rows.values() if r["status"] == "ok"]
    tech = [r["key"] for r in rows.values() if r["status"] != "ok"]
    print(f"\nΤέλος: {len(done)} απαντήσεις, {len(tech)} τεχνικές αποτυχίες"
          + (f" ({', '.join(tech)})" if tech else "") + f" · εκτίμηση κόστους {spent:.2f} $")
    print(f"Απαντήσεις (ιδιωτικό, εκτός git): {os.path.relpath(out_path, H.BACKEND)}")
    print("Επόμενο, στο Git Bash: python backend/evaluation/judge_heldout.py" + (" --smoke" if args.smoke else ""))
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="δοκιμή του εργαλείου στα άρθρα cloud")
    ap.add_argument("--dry-run", action="store_true", help="μόνο έλεγχοι, βάση και router· 0 κλήσεις")
    ap.add_argument("--rebuild-store", action="store_true")
    ap.add_argument("--arms", default="alpha,beta,closed")
    ap.add_argument("--max-usd", type=float, default=5.0, help="όριο ΕΚΤΙΜΩΜΕΝΟΥ κόστους Gemini")
    ap.add_argument("--in-usd", type=float, default=0.30, help="$ ανά 1M tokens εισόδου (εκτίμηση)")
    ap.add_argument("--out-usd", type=float, default=2.50, help="$ ανά 1M tokens εξόδου (εκτίμηση)")
    args = ap.parse_args()
    bad = [a for a in args.arms.split(",") if a.strip() and a.strip() not in H.ARMS]
    if bad:
        ap.error(f"άγνωστα σκέλη: {bad}")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())

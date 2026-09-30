"""Φάση 3.2α — LlamaIndex με τις ΠΡΟΕΠΙΛΟΓΕΣ του: απαντήσεις στις ίδιες ερωτήσεις (χωρίς κριτή).

ΤΙ ΕΙΝΑΙ: ό,τι παίρνει κάποιος που ακολουθεί το «starter tutorial» του LlamaIndex πάνω στα ΙΔΙΑ 7 PDF:
    SimpleDirectoryReader (pypdf, ένα έγγραφο ανά σελίδα) -> VectorStoreIndex (SentenceSplitter 1024
    tokens / επικάλυψη 200) -> as_query_engine() (top-2 κομμάτια, response_mode «compact», το δικό του
    πρότυπο ερώτησης). ΚΑΜΙΑ ρύθμιση δεν αλλάζει, με ΔΥΟ αναγκαστικές εξαιρέσεις:
        embeddings  bge-m3 (όπως το δικό μας) — η προεπιλογή θέλει κλειδί OpenAI
        LLM         Gemini 2.5 Flash (όπως το δικό μας) με τις προεπιλογές του GoogleGenAI
    Έτσι η διαφορά μετράει τη ΣΩΛΗΝΩΣΗ (κομμάτιασμα, ανάκτηση, πρότυπο), όχι το μοντέλο.

ΓΙΑΤΙ ΧΩΡΙΣΤΟ SCRIPT: τρέχει σε μιας-χρήσης container με το llama-index εγκατεστημένο και ΔΕΝ φορτώνει
το ai_core (δύο αντίγραφα του bge-m3 = ~4.6 GB, δεν χωράνε). Ο κριτής τρέχει μετά, στο κανονικό
περιβάλλον: compare_systems.py --system llamaindex.

ΕΞΟΔΟΣ: runs/compare/llamaindex_answers.json — ανά ερώτηση: απάντηση, τα κομμάτια που διάβασε το
μοντέλο (κείμενο + αρχείο + σελίδα), tokens από το usage του Gemini, χρόνος. Συνεχίζει από όπου
σταμάτησε (ένα 429/402 δεν χάνει όσα πληρώθηκαν). Το ευρετήριο σώζεται στο runs/compare/llamaindex_store/
(αντίγραφα papers — gitignored) και ξαναχρησιμοποιείται.

ΤΡΕΞΙΜΟ (από τη ρίζα του repo, Git Bash) — δοκιμή 2 ανά σετ, μετά όλο:
    MSYS_NO_PATHCONV=1 docker compose run --rm --no-deps -u root backend sh -c \
      "pip install -q --no-warn-conflicts -r evaluation/requirements_compare.txt && python -u evaluation/compare_llamaindex.py --limit 2"
    (το ίδιο χωρίς --limit). Το pip αναβαθμίζει το pydantic σε 2.x ΜΕΣΑ στο προσωρινό container — σπάει
    fastapi/chromadb εκεί, που αυτό το script δεν φορτώνει· το image δεν αλλάζει.
    Μοντέλο: ΜΟΝΟ το τοπικό αντίγραφο στο cache (HF_HUB_OFFLINE)· αν λείπει, σταματάει με διάγνωση.
"""
import argparse
import glob
import importlib.metadata as md
import json
import os
import time

# ΠΡΙΝ από κάθε import του huggingface_hub / transformers: ΚΑΜΙΑ λήψη. Το 1ο τρέξιμο (30/9) με το
# όνομα «BAAI/bge-m3» ΞΑΝΑΚΑΤΕΒΑΣΕ το μοντέλο (2.27 GB) και μετά ένα model.safetensors στο παρασκήνιο
# (αυτόματη μετατροπή του transformers για .bin) -> «Killed» από μνήμη. Φορτώνουμε το ΤΟΠΙΚΟ αντίγραφο.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(HERE, "test_papers", "cloud")
OUT_DIR = os.path.join(HERE, "runs", "compare")
OUT_PATH = os.path.join(OUT_DIR, "llamaindex_answers.json")
STORE = os.path.join(OUT_DIR, "llamaindex_store")
SETS = {"main": "golden_set_50.jsonl", "mh_new": "golden_multihop_v2.jsonl",
        "near_ooc": "golden_near_ooc.jsonl", "tables": "golden_tables.jsonl"}
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED = "BAAI/bge-m3"
PACKAGES = ("llama-index-core", "llama-index-embeddings-huggingface", "llama-index-llms-google-genai",
            "llama-index-readers-file", "google-genai", "pypdf", "sentence-transformers")

_USAGE: list[dict] = []
_KEYS = {"prompt_token_count": "promptTokenCount", "candidates_token_count": "candidatesTokenCount",
         "thoughts_token_count": "thoughtsTokenCount",
         "cached_content_token_count": "cachedContentTokenCount"}


def _find_usage(obj) -> dict | None:
    """Το usage του Gemini μέσα στο raw της απάντησης, όπου κι αν το βάζει η έκδοση του wrapper."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    if isinstance(obj, dict):
        for snake, camel in _KEYS.items():
            if snake in obj or camel in obj:
                return {c: obj.get(s, obj.get(c)) or 0 for s, c in _KEYS.items()}
        for v in obj.values():
            u = _find_usage(v)
            if u:
                return u
    if isinstance(obj, (list, tuple)):
        for v in obj:
            u = _find_usage(v)
            if u:
                return u
    return None


def install_usage_handler() -> None:
    from llama_index.core.instrumentation import get_dispatcher
    from llama_index.core.instrumentation.event_handlers import BaseEventHandler

    class UsageHandler(BaseEventHandler):
        @classmethod
        def class_name(cls) -> str:
            return "UsageHandler"

        def handle(self, event, **kwargs):
            if not type(event).__name__.endswith("EndEvent"):
                return
            resp = getattr(event, "response", None)
            u = _find_usage(getattr(resp, "raw", None)) if resp is not None else None
            if u:
                _USAGE.append(u)

    get_dispatcher().add_event_handler(UsageHandler())


def local_bge_m3() -> tuple[str | None, list[str]]:
    """Το ΠΛΗΡΕΣ τοπικό αντίγραφο του bge-m3 στο cache (ό,τι φορτώνει ήδη η εφαρμογή) + διάγνωση.
    Πλήρες = modules.json, config, tokenizer, pooling ΚΑΙ βάρη. Το os.path.exists ακολουθεί το symlink
    του snapshot στο blob — ένα μισοκατεβασμένο blob (.incomplete) δεν έχει ακόμα symlink."""
    need = ("modules.json", "config.json", "tokenizer.json", os.path.join("1_Pooling", "config.json"))
    roots = dict.fromkeys([os.path.join(os.getenv("HF_HOME", "/home/appuser/.cache/huggingface"), "hub"),
                           "/home/appuser/.cache/huggingface/hub", "/root/.cache/huggingface/hub"])
    report, complete = [], []
    for root in roots:
        repo = os.path.join(root, "models--BAAI--bge-m3")
        if not os.path.isdir(repo):
            report.append(f"{repo}: δεν υπάρχει")
            continue
        ref, main_ref = os.path.join(repo, "refs", "main"), "-"
        if os.path.exists(ref):
            with open(ref, encoding="utf-8") as f:
                main_ref = f.read().strip()[:10]
        partial = len(glob.glob(os.path.join(repo, "blobs", "*.incomplete")))
        report.append(f"{repo}: refs/main {main_ref} · μισοκατεβασμένα blobs {partial}")
        for snap in sorted(glob.glob(os.path.join(repo, "snapshots", "*"))):
            weights = [w for w in ("model.safetensors", "pytorch_model.bin")
                       if os.path.exists(os.path.join(snap, w))]
            missing = [f for f in need if not os.path.exists(os.path.join(snap, f))]
            ok = bool(weights) and not missing
            report.append(f"  snapshot {os.path.basename(snap)[:10]}: βάρη {weights or '-'} · "
                          f"λείπουν {missing or '-'} -> {'ΠΛΗΡΕΣ' if ok else 'ελλιπές'}")
            if ok:
                complete.append(snap)
    return (complete[0] if complete else None), report


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_index(model_path: str):
    from llama_index.core import (
        Settings,
        SimpleDirectoryReader,
        StorageContext,
        VectorStoreIndex,
        load_index_from_storage,
    )
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.google_genai import GoogleGenAI

    # embed_batch_size 10 -> 2: ρύθμιση ΜΝΗΜΗΣ, όχι ποιότητας (ίδια διανύσματα ανά κείμενο). Με 10
    # κομμάτια των 1024 tokens μαζί, η προσοχή του bge-m3 σε CPU θέλει ~0.7 GB επιπλέον ανά στρώμα.
    Settings.embed_model = HuggingFaceEmbedding(model_name=model_path, embed_batch_size=2)
    Settings.llm = GoogleGenAI(model=MODEL, api_key=os.getenv("GEMINI_API_KEY"))
    if os.path.exists(os.path.join(STORE, "docstore.json")):
        print(f"Ευρετήριο από {STORE}", flush=True)
        return load_index_from_storage(StorageContext.from_defaults(persist_dir=STORE))
    docs = SimpleDirectoryReader(PDF_DIR, required_exts=[".pdf"]).load_data()
    print(f"{len(docs)} έγγραφα (σελίδες) — embeddings με {EMBED} σε CPU, ~5-15 λεπτά...", flush=True)
    t0 = time.perf_counter()
    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    index.storage_context.persist(STORE)
    print(f"Ευρετήριο: {len(index.docstore.docs)} κομμάτια σε {time.perf_counter() - t0:.0f} s", flush=True)
    return index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="N ανά σετ (δοκιμή)")
    args = ap.parse_args()
    if not os.getenv("GEMINI_API_KEY"):
        print("!! Λείπει GEMINI_API_KEY")
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"system": "llamaindex", "rows": {}}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as f:
            out = json.load(f)
    out["versions"] = {p: md.version(p) for p in PACKAGES}
    print("Εκδόσεις: " + " · ".join(f"{p} {v}" for p, v in out["versions"].items()), flush=True)
    model_path, report = local_bge_m3()
    print("bge-m3 στο cache:\n  " + "\n  ".join(report), flush=True)
    if not model_path:
        print("!! Δεν βρέθηκε ΠΛΗΡΕΣ τοπικό αντίγραφο του bge-m3 — σταματάω (χωρίς λήψεις, βλ. πάνω)")
        return 1
    out["config"] = {"reader": "SimpleDirectoryReader (defaults)", "splitter": "defaults",
                     "query_engine": "as_query_engine() defaults",
                     "embed_model": f"{EMBED} @ {os.path.basename(model_path)[:10]} (τοπικό)",
                     "embed_batch_size": 2, "llm": MODEL}

    install_usage_handler()
    index = build_index(model_path)
    qe = index.as_query_engine()
    out["config"]["similarity_top_k"] = getattr(qe.retriever, "similarity_top_k", None)

    tests = [(sk, t) for sk, fn in SETS.items()
             for t in load_jsonl(os.path.join(HERE, fn))[:args.limit or None]]
    todo = [(sk, t) for sk, t in tests if f"{sk}:{t['id']}" not in out["rows"]]
    print(f"\n{len(tests)} ερωτήσεις · έτοιμες {len(tests) - len(todo)} · νέες {len(todo)}\n", flush=True)
    for sk, t in todo:
        _USAGE.clear()
        t0 = time.perf_counter()
        try:
            resp = qe.query(t["question"])
        except Exception as e:                      # 429 / 402 / δίκτυο: σώσε ό,τι έγινε και σταμάτα
            print(f"!! {sk} {t['id']}: {type(e).__name__}: {str(e)[:300]} — σταματάω (τα έτοιμα σώθηκαν)")
            break
        dt = time.perf_counter() - t0
        nodes = [{"text": n.node.get_content(), "file_name": n.node.metadata.get("file_name"),
                  "page": n.node.metadata.get("page_label"), "score": n.score}
                 for n in resp.source_nodes]
        usage = _USAGE[-1] if _USAGE else None
        out["rows"][f"{sk}:{t['id']}"] = {"set": sk, "id": t["id"], "question": t["question"],
                                          "answer": str(resp), "pages": nodes, "usage": usage,
                                          "llm_calls": len(_USAGE), "seconds": round(dt, 2)}
        tmp = OUT_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        os.replace(tmp, OUT_PATH)
        tok = (f"in {usage['promptTokenCount']} out {usage['candidatesTokenCount']} "
               f"think {usage['thoughtsTokenCount']}") if usage else "usage: —"
        where = ", ".join(f"{n['file_name']}:{n['page']}" for n in nodes)   # 3.11: όχι ίδια εισαγωγικά μέσα
        print(f"  {sk:<8} {t['id']:<6} {dt:5.1f}s · {tok} · {where}", flush=True)
    done = sum(1 for sk, t in tests if f"{sk}:{t['id']}" in out["rows"])
    print(f"\nΈτοιμες {done}/{len(tests)} · {OUT_PATH}")
    return 0 if done == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())

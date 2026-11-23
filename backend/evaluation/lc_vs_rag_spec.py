"""Κοινή προδιαγραφή πειράματος RAG vs όλο το κείμενο (μόνο stdlib — τρέχει και στο Docker και στον υπολογιστή).

Μία πηγή αλήθειας για:
- build_items / expected_row_keys / validate_answers (ακριβής έλεγχος πληρότητας: 654 πρωτεύουσες + 56 distractor)
- BudgetLedger (σωρευτικό όριο κόστους σε δίσκο που επιβιώνει resume + έλεγχος με περιθώριο ΠΡΙΝ από κάθε
  πληρωμένη κλήση: γέννηση, ανάκτηση ή ζωντανή χρονομέτρηση)
- επιλογή του χωριστού σκέλους distractor (14 γνωστές μονοαρθρικές ερωτήσεις σε 2 και 3 άρθρα)
- επιλογή στρωματοποιημένου δείγματος ζωντανής χρονομέτρησης (el + en ανά μέγεθος)
"""
from __future__ import annotations

import ast
import json
import os
import random
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LC_DIR = os.path.join(HERE, "runs", "longctx")
BUDGET_PATH = os.path.join(LC_DIR, "budget.json")

SEED = 20261123
PRIMARY_DRAWS = {"one": 2, "two": 2, "three": 2, "safety": 1}
DRAWS = {**PRIMARY_DRAWS, "distract": 1}
PRIMARY_ARMS = tuple(PRIMARY_DRAWS)
ALL_ARMS = tuple(DRAWS)
QUALITY_ARMS = ("one", "two", "three")
SYSTEMS = ("rag", "full")


def _literal_from(path: str, name: str):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} δεν βρέθηκε στο {path}")


ALIASES: dict[str, tuple[str, ...]] = _literal_from(os.path.join(HERE, "scoreboard_answers.py"), "ALIASES")
NAMES: dict[str, str] = _literal_from(os.path.join(HERE, "scoreboard_answers.py"), "NAMES")
CANONICAL_FILES: list[str] = sorted(NAMES)


def load_jsonl(name: str, base_dir: str = HERE) -> list[dict]:
    with open(os.path.join(base_dir, name), encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def lang_of(t: dict, question: str) -> str:
    if t.get("lang"):
        return t["lang"]
    return "el" if any("Ͱ" <= ch <= "Ͽ" for ch in question) else "en"


def cyclic(files: list[str], src: str, k: int) -> list[str]:
    i = files.index(src)
    return [files[(i + j) % len(files)] for j in range(k)]


def mention_order(t: dict) -> list[str]:
    """Οι σελίδες-τεκμήρια με τη σειρά που αναφέρεται το paper τους στην ερώτηση (ίδιο με scoreboard_answers)."""
    q = t["question_en"].lower()
    pos = {}
    for ep in t["evidence_pages"]:
        doc = ep.rsplit(":", 1)[0]
        found = [q.find(a) for a in ALIASES[doc] if a in q]
        if not found:
            raise ValueError(f"{t['id']}: κανένα όνομα του {doc} στην ερώτηση")
        pos[ep] = min(found)
    if len(set(pos.values())) != len(pos):
        raise ValueError(f"{t['id']}: δύο papers στην ίδια θέση")
    return sorted(t["evidence_pages"], key=pos.get)


def mh_docs(t: dict) -> list[str]:
    return [ep.rsplit(":", 1)[0] for ep in mention_order(t)]


def distractor_selection(papers: dict[str, str], main: dict[str, dict], per_paper: int = 2) -> list[tuple[str, str]]:
    """Ντετερμινιστική επιλογή γνωστών μονοαρθρικών ερωτήσεων (2 ανά άρθρο = 14 συνολικά), ισορροπημένες
    σε el/en όπου ένα άρθρο έχει και τις δύο γλώσσες."""
    by_doc: dict[str, list[tuple[str, str]]] = {}
    for qid, doc in papers.items():
        q = main[qid]["question"]
        by_doc.setdefault(doc, []).append((qid, lang_of(main[qid], q)))
    chosen: list[tuple[str, str]] = []
    for doc in sorted(by_doc):
        pool = by_doc[doc]
        picked: list[str] = []
        # Προτίμησε 1 αγγλική + 1 ελληνική όταν υπάρχουν και οι δύο
        for want_lang in ("en", "el"):
            hit = next((qid for qid, lg in pool if lg == want_lang and qid not in picked), None)
            if hit is not None and len(picked) < per_paper:
                picked.append(hit)
        for qid, _lg in pool:
            if len(picked) >= per_paper:
                break
            if qid not in picked:
                picked.append(qid)
        for qid in picked:
            chosen.append((qid, doc))
    return chosen


def build_items(files: list[str], *, include_distract: bool = True, base_dir: str = HERE) -> list[dict]:
    """Χτίζει τα 4 προκαταχωρισμένα σκέλη (234 items -> 654 γραμμές) και προαιρετικά το χωριστό σκέλος
    distractor (28 items -> 56 γραμμές) στο ΤΕΛΟΣ, ώστε να μην αγγίζει ποτέ το προκαταχωρισμένο τεστ."""
    files = list(files)
    main = {t["id"]: t for t in load_jsonl("golden_set_50.jsonl", base_dir)}
    with open(os.path.join(base_dir, "main_set_papers.json"), encoding="utf-8") as f:
        papers = json.load(f)["papers"]
    items: list[dict] = []
    for qid, doc in papers.items():
        items.append({"arm": "one", "set": "main", "t": main[qid], "docs": [doc], "target_doc": doc})
    for t in load_jsonl("golden_tables.jsonl", base_dir):
        items.append({"arm": "one", "set": "tables", "t": t, "docs": [t["doc"]], "target_doc": t["doc"]})
    for t in load_jsonl("golden_multihop_v2.jsonl", base_dir):
        items.append({"arm": "two", "set": "mh_new", "t": t, "docs": mh_docs(t)})
    for t in load_jsonl("golden_three_papers.jsonl", base_dir):
        items.append({"arm": "three", "set": "t3p", "t": t, "docs": list(t["docs"])})
    rng = random.Random(SEED)
    safety = [("near_ooc", t) for t in load_jsonl("golden_near_ooc.jsonl", base_dir)]
    safety += [("ooc", t) for t in main.values() if t.get("category") == "out_of_corpus"]
    for sk, t in safety:
        src = t.get("source_doc") or rng.choice(files)
        for k in (1, 2, 3):
            items.append({"arm": "safety", "set": sk, "t": t, "docs": cyclic(files, src, k)})
    if include_distract:
        for qid, doc in distractor_selection(papers, main):
            for k in (2, 3):
                items.append({
                    "arm": "distract",
                    "set": "main",
                    "t": main[qid],
                    "docs": cyclic(files, doc, k),
                    "target_doc": doc,
                })
    for it in items:
        it["size"] = len(it["docs"])
        assert all(d in files for d in it["docs"]), (it["t"]["id"], it["docs"])
    return sorted(items, key=lambda it: (ALL_ARMS.index(it["arm"]), it["size"], it["docs"]))


def row_key(r: dict) -> tuple:
    return (r["arm"], r["set"], r["id"], int(r["size"]), r["docs"], int(r["draw"]), r["system"])


def expected_row_keys(
    files: list[str] | None = None,
    *,
    include_distract: bool = False,
    base_dir: str = HERE,
) -> set[tuple]:
    items = build_items(files or CANONICAL_FILES, include_distract=include_distract, base_dir=base_dir)
    keys: set[tuple] = set()
    for it in items:
        docs_s = ";".join(it["docs"])
        for draw in range(1, DRAWS[it["arm"]] + 1):
            for system in SYSTEMS:
                keys.add((it["arm"], it["set"], it["t"]["id"], it["size"], docs_s, draw, system))
    return keys


def validate_answers(
    data: dict,
    *,
    smoke: bool = False,
    files: list[str] | None = None,
    base_dir: str = HERE,
) -> list[str]:
    """Ελέγχει πληρότητα και ακεραιότητα του answers.json ΠΡΙΝ από κρίση ή απόφαση."""
    errors: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        return ["μη έγκυρη δομή answers.json (λείπει η λίστα rows)"]
    rows = data["rows"]
    meta = data.get("meta") or {}
    if not rows:
        return ["το αρχείο δεν περιέχει καμία γραμμή"]

    seen: set[tuple] = set()
    dups: list[tuple] = []
    empty_ans: list[tuple] = []
    for idx, r in enumerate(rows):
        if not isinstance(r, dict):
            errors.append(f"γραμμή #{idx}: δεν είναι αντικείμενο JSON")
            continue
        missing_fields = [k for k in ("arm", "set", "id", "size", "docs", "draw", "system", "cut") if k not in r]
        if missing_fields:
            errors.append(f"γραμμή #{idx}: λείπουν πεδία {missing_fields}")
            continue
        k = row_key(r)
        if k in seen:
            dups.append(k)
        seen.add(k)
        if not r.get("cut") and not (r.get("answer") or "").strip():
            empty_ans.append(k)
    if dups:
        errors.append(f"βρέθηκαν {len(dups)} διπλότυπες γραμμές (π.χ. {dups[:3]})")
    if empty_ans:
        errors.append(f"βρέθηκαν {len(empty_ans)} μη-κομμένες γραμμές με κενή απάντηση (π.χ. {empty_ans[:3]})")
    if smoke:
        return errors

    if meta.get("complete") is not True:
        errors.append(f"το meta.complete είναι {meta.get('complete')!r} (όχι True) — το τρέξιμο είναι ημιτελές")

    has_distract = any(r.get("arm") == "distract" for r in rows if isinstance(r, dict))
    expect_distract = bool(meta.get("include_distract")) or has_distract

    primary_exp = expected_row_keys(files, include_distract=False, base_dir=base_dir)
    primary_got = {k for k in seen if k[0] in PRIMARY_ARMS}
    miss_p = sorted(primary_exp - primary_got)
    extra_p = sorted(primary_got - primary_exp)
    if miss_p or extra_p:
        errors.append(
            f"προκαταχωρισμένο σετ: βρέθηκαν {len(primary_got)}/{len(primary_exp)} μοναδικές γραμμές "
            f"(λείπουν {len(miss_p)}, πλεονάζουν {len(extra_p)}"
            + (f", π.χ. λείπει {miss_p[0]}" if miss_p else "")
            + (f", π.χ. πλεονάζει {extra_p[0]}" if extra_p else "")
            + ")"
        )

    if expect_distract:
        all_exp = expected_row_keys(files, include_distract=True, base_dir=base_dir)
        dist_exp = all_exp - primary_exp
        dist_got = {k for k in seen if k[0] == "distract"}
        miss_d = sorted(dist_exp - dist_got)
        extra_d = sorted(dist_got - dist_exp)
        if miss_d or extra_d:
            errors.append(
                f"σκέλος distractor: βρέθηκαν {len(dist_got)}/{len(dist_exp)} μοναδικές γραμμές "
                f"(λείπουν {len(miss_d)}, πλεονάζουν {len(extra_d)})"
            )

    unknown_arms = sorted({k[0] for k in seen if k[0] not in ALL_ARMS})
    if unknown_arms:
        errors.append(f"άγνωστα σκέλη: {unknown_arms}")
    return errors


def select_live_timing_items(items: list[dict], per_size: int = 4) -> list[dict]:
    """Επιλέγει ντετερμινιστικά έως `per_size` ερωτήσεις ποιότητας ανά μέγεθος (1, 2, 3 άρθρα),
    ισορροπώντας ελληνικές και αγγλικές ερωτήσεις, για ζωντανή μέτρηση χρόνου από έναρξη έως 1η λέξη."""
    if per_size <= 0:
        return []
    selected: list[dict] = []
    for arm in QUALITY_ARMS:
        pool = [it for it in items if it["arm"] == arm]
        el_pool = [it for it in pool if lang_of(it["t"], it["t"]["question"]) == "el"]
        en_pool = [it for it in pool if lang_of(it["t"], it["t"]["question"]) != "el"]
        picked: list[dict] = []
        seen_ids: set[str] = set()
        # Εναλλάξ el / en ώστε το δείγμα να πιάνει ΚΑΙ ζωντανές μεταφράσεις ΚΑΙ αγγλικές ερωτήσεις
        max_len = max(len(el_pool), len(en_pool))
        for i in range(max_len):
            for bucket in (el_pool, en_pool):
                if len(picked) >= per_size:
                    break
                if i < len(bucket) and bucket[i]["t"]["id"] not in seen_ids:
                    picked.append(bucket[i])
                    seen_ids.add(bucket[i]["t"]["id"])
            if len(picked) >= per_size:
                break
        selected.extend(picked)
    return selected


def usage_tokens(usage: dict | None) -> tuple[int, int, int]:
    """(prompt_tokens, completion_tokens, thinking_tokens) — ίδια αριθμητική με gemini_rest.usage_tokens."""
    if not usage:
        return 0, 0, 0
    prompt = usage.get("promptTokenCount") or 0
    completion = usage.get("candidatesTokenCount") or 0
    total = usage.get("totalTokenCount") or 0
    thinking = usage.get("thoughtsTokenCount")
    if thinking is None:
        thinking = max(0, total - prompt - completion)
    return int(prompt), int(completion), int(thinking)


def usage_usd(usage: dict | None, in_usd: float, out_usd: float, cached_share: float) -> float:
    if not usage:
        return 0.0
    p, c_out, th = usage_tokens(usage)
    cached = min(p, int(usage.get("cachedContentTokenCount") or 0))
    return ((p - cached) * in_usd + cached * in_usd * cached_share + (c_out + th) * out_usd) / 1e6


def estimate_call_usd(prompt: str, max_output_tokens: int, in_usd: float, out_usd: float) -> float:
    """Συντηρητική εκτίμηση ανώτατου κόστους για ΜΙΑ επικείμενη κλήση (0% cache, ~3 χαρ./token,
    πλήρες max_output_tokens). Εξασφαλίζει περιθώριο πριν αποσταλεί το HTTP αίτημα."""
    est_in_tokens = max(1, (len(prompt) + 2) // 3)
    est_out_tokens = max(256, int(max_output_tokens or 1024))
    return (est_in_tokens * in_usd + est_out_tokens * out_usd) / 1e6


def _atomic_write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


class BudgetLedger:
    """Σωρευτικό μητρώο δαπάνης Gemini που επιβιώνει επανεκκινήσεις (resume) και ελέγχει ΠΡΙΝ από
    ΚΑΘΕ πληρωμένη κλήση (γέννηση, ανάκτηση ή ζωντανή χρονομέτρηση) με συντηρητικό περιθώριο."""

    def __init__(
        self,
        path: str,
        *,
        max_usd: float,
        in_usd: float,
        out_usd: float,
        cached_share: float,
        seed_files: tuple[tuple[str, str], ...] = (),
    ):
        self.path = path
        self.max_usd = float(max_usd)
        self.in_usd = float(in_usd)
        self.out_usd = float(out_usd)
        self.cached_share = float(cached_share)
        self.run_usd = 0.0
        self.exceeded: str | None = None
        self.state = {
            "total_usd": 0.0,
            "gen_usd": 0.0,
            "ret_usd": 0.0,
            "live_usd": 0.0,
            "gen_calls": 0,
            "ret_calls": 0,
            "live_calls": 0,
        }
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            for k in self.state:
                if k in loaded:
                    self.state[k] = loaded[k]
        elif seed_files:
            # Αν λείπει το budget.json αλλά υπάρχουν παγωμένες κλήσεις από προηγούμενο τρέξιμο,
            # υπολόγισε τη σωρευτική δαπάνη από τα αποθηκευμένα usage ώστε να μην μηδενιστεί το πλαφόν.
            self._bootstrap_from_caches(seed_files)

    def _bootstrap_from_caches(self, seed_files: tuple[tuple[str, str], ...]) -> None:
        for kind, fpath in seed_files:
            if not os.path.exists(fpath):
                continue
            with open(fpath, encoding="utf-8") as f:
                raw = json.load(f)
            entries = raw.values() if isinstance(raw, dict) else []
            for entry in entries:
                if isinstance(entry, dict) and entry.get("usage"):
                    c = usage_usd(entry["usage"], self.in_usd, self.out_usd, self.cached_share)
                    self.state["total_usd"] = round(self.state["total_usd"] + c, 6)
                    self.state[f"{kind}_usd"] = round(self.state[f"{kind}_usd"] + c, 6)
                    self.state[f"{kind}_calls"] += 1

    @property
    def total_usd(self) -> float:
        return float(self.state["total_usd"])

    def check_before_call(self, prompt: str, max_output_tokens: int, label: str = "") -> float:
        """Ελέγχει αν η επόμενη πληρωμένη κλήση χωράει στο συνολικό πλαφόν. Αν όχι, σημειώνει
        self.exceeded και πετά RuntimeError ΠΡΙΝ γίνει οποιοδήποτε αίτημα δικτύου."""
        est = estimate_call_usd(prompt, max_output_tokens, self.in_usd, self.out_usd)
        if self.total_usd + est > self.max_usd:
            who = f" ({label})" if label else ""
            msg = (
                f"ΟΡΙΟ ΚΟΣΤΟΥΣ{who}: σωρευτικά {self.total_usd:.4f} $ + εκτίμηση επόμενης κλήσης "
                f"{est:.4f} $ > πλαφόν {self.max_usd:.2f} $ (σε αυτό το τρέξιμο {self.run_usd:.4f} $)"
            )
            self.exceeded = msg
            raise RuntimeError(msg)
        return est

    def record(self, kind: str, usage: dict | None) -> float:
        """Καταγράφει την πραγματική χρέωση μιας πληρωμένης κλήσης και σώζει ατομικά στο δίσκο."""
        cost = usage_usd(usage, self.in_usd, self.out_usd, self.cached_share)
        self.run_usd += cost
        self.state["total_usd"] = round(self.state["total_usd"] + cost, 6)
        usd_key = f"{kind}_usd"
        calls_key = f"{kind}_calls"
        if usd_key in self.state:
            self.state[usd_key] = round(self.state[usd_key] + cost, 6)
            self.state[calls_key] += 1
        self.save()
        return cost

    def snapshot(self) -> dict:
        return {
            **self.state,
            "run_usd": round(self.run_usd, 6),
            "max_usd": self.max_usd,
        }

    def save(self) -> None:
        _atomic_write_json(self.path, self.state)

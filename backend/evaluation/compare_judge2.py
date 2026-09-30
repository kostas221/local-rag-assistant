"""Φάση 3.2β — ΔΕΥΤΕΡΟΣ ΚΡΙΤΗΣ από ΑΛΛΗ εταιρεία (OpenAI) πάνω στις ΙΔΙΕΣ απαντήσεις της 3.2α.

ΤΟ ΕΡΩΤΗΜΑ: η σύγκριση ours / naive / llamaindex κρίθηκε από Gemini, πάνω σε απαντήσεις Gemini. Αλλάζει η
κατάταξη αν κρίνει μοντέλο άλλης οικογένειας; Και: ο κριτής του κύριου σετ (v1) κρίνει ΩΣ ΠΡΟΣ ΤΟ CONTEXT
(βάζει 5/5/5/5 στο «δεν το βρίσκω» όταν η ανάκτηση απέτυχε) — εδώ κρίνει ΜΟΝΟ ως προς την ΑΠΑΝΤΗΣΗ ΑΝΑΦΟΡΑΣ.

ΤΙ ΚΑΝΕΙ: 0 νέες απαντήσεις. Διαβάζει τις απαντήσεις (ours: scoreboard · naive/llamaindex: runs/compare) και
τις κρίνει ξανά:
    main 50      ΝΕΟΣ κριτής ως προς την αναφορά: correct · partial · refused · wrong
    mh_new 29    ΝΕΟΣ κριτής ανά paper ως προς την αναφορά: correct · partial · declared_missing · omitted · wrong
    near_ooc 42  ο κριτής της Φάσης 0.1 ΑΥΤΟΥΣΙΟΣ (το ίδιο κείμενο, διαβασμένο από το eval_near_ooc.py με ast)·
                 η σιωπή του φύλακα (ours, 2) μετράει σωστό χωρίς κρίση, όπως στον 1ο κριτή
    tables       δεν χρειάζεται κριτή (ακριβείς τιμές)
Η στήριξη στο context (faithfulness / «χωρίς στήριξη») ΜΕΝΕΙ από τον 1ο κριτή: ο 2ος δεν βλέπει context,
επίτηδες — μετράει ΟΡΘΟΤΗΤΑ, όχι θεμελίωση.
ΣΥΜΦΩΝΙΑ ΚΡΙΤΩΝ: ανά μονάδα (μισό στα δύο papers, ερώτηση στις κοντινές) ποσοστό + Cohen κ.

ΤΡΕΧΕΙ ΣΤΟΝ ΥΠΟΛΟΓΙΣΤΗ, όχι στο Docker (μόνο βασική βιβλιοθήκη της Python). Κλειδί: OPENAI_API_KEY από το
περιβάλλον ή από το .env της ρίζας. Πάγωμα: runs/compare/judge2_cache.json (κλειδί μοντέλο + prompt) —
ξανατρέξιμο = 0 κλήσεις. Φύλακας κόστους --max-usd (default 3, με ΕΚΤΙΜΗΜΕΝΕΣ τιμές ανά μοντέλο).

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (30/9/2026):
    ΚΑΤΑΤΑΞΗ ίδια με τον 1ο κριτή σε κύριο και δύο papers: ours > naive ≈ llamaindex.
    κύριο «correct»: ours 38-44/45 · naive 30-38 · llamaindex 26-35 (οι 5 λάθος αρνήσεις του llamaindex
        -> «refused», όχι correct)
    δύο papers σωστά μισά: ours 38-48/56 · naive 16-26 · llamaindex 16-26
    κοντινές: ίδιο σύνολο ±2 ανά σύστημα · συμφωνία με τον 1ο κριτή ≥ 90%
    συμφωνία στα μισά των δύο papers: 75-90%, κ 0.5-0.8 (το «partial» είναι θολό όριο)
ΚΑΝΟΝΑΣ (γραμμένος πριν): αν ο 2ος κριτής ΑΝΤΙΣΤΡΕΦΕΙ τη σειρά ours / άλλο σε κύριο ή δύο papers πάνω από
    τον θόρυβο (±3 / ±4) -> το αποτέλεσμα της 3.2α ΔΕΝ δημοσιεύεται ως έχει, γράφεται η διαφωνία.
    Ίδια σειρά -> «δύο κριτές από δύο εταιρείες συμφωνούν».

ΑΠΟΤΕΛΕΣΜΑ (30/9/2026, gpt-4.1, runs/compare/judge2_gpt-4.1.json · 360 κρίσεις · ~0.67 $):
                                        ours      naive     llamaindex
    κύριο: correct                      42/45     34/45     26/45     (partial 2 / 7 / 15 · refused 1 / 4 / 4)
    δύο papers: σωστά μισά              45/58     18/58     17/58
    δύο papers: «και τα δύο σωστά»      18/29      3/29      2/29
    κοντινές: σωστή άρνηση              40/42     41/42     41/42
    ζευγαρωτά (ours κερδίζει/χάνει): κύριο 11/3 (p=0.06) και 17/1 (p<0.001) · δύο papers 28/1 και 28/0 ·
    κοντινές 1/2 και 0/1 (θόρυβος).
    ΚΑΝΟΝΑΣ -> ΙΔΙΑ ΚΑΤΑΤΑΞΗ με τον 1ο κριτή: «δύο κριτές από δύο εταιρείες συμφωνούν».
    ΣΥΜΦΩΝΙΑ ΚΡΙΤΩΝ: δύο papers 166/174 μισά (95%, κ=0.91) · κοντινές 123/124 (99%, κ=0.85).
    Κύριο: και οι δύο «τέλειο/correct» 96 · ΜΟΝΟ ο 1ος 17 — 16 από αυτά naive/llamaindex (οι αρνήσεις και
    οι μισές απαντήσεις που ο 1ος κριτής, ως προς το context, επιβράβευε· επιβεβαιώνει το εύρημα της 3.2α)
    · ours q032 (partial: λείπει η σημείωση για τις τριμηνιαίες πωλήσεις — αυστηρό αλλά σωστό).
    Πρόβλεψη: 9/10 — μόνο η συμφωνία στα δύο papers βγήκε ΥΨΗΛΟΤΕΡΗ (95% / κ 0.91 έναντι 75-90% / 0.5-0.8).
    Όριο: ο 2ος κριτής δεν βλέπει context -> δεν κρίνει θεμελίωση (μένει από τον 1ο).

ΤΡΕΞΙΜΟ (Git Bash, από τη ρίζα του repo):
    python backend/evaluation/compare_judge2.py --list-models          # δωρεάν: ποια μοντέλα βλέπει το κλειδί
    python backend/evaluation/compare_judge2.py --limit 2              # δοκιμή: 2 ανά σετ ανά σύστημα
    python backend/evaluation/compare_judge2.py                        # όλο
"""
import argparse
import ast
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(HERE, "runs")
CMP = os.path.join(RUNS, "compare")
CACHE_PATH = os.path.join(CMP, "judge2_cache.json")
SOURCES = {
    "ours": {"main": "scoreboard/answers_baseline_main.json", "mh_new": "scoreboard/answers_perdoc_mh_new.json",
             "near_ooc": "scoreboard/answers_baseline_near_ooc.json"},
    "naive": {sk: f"compare/naive_{sk}.json" for sk in ("main", "mh_new", "near_ooc")},
    "llamaindex": {sk: f"compare/llamaindex_{sk}.json" for sk in ("main", "mh_new", "near_ooc")},
}
GOLD = {"main": "golden_set_50.jsonl", "mh_new": "golden_multihop_v2.jsonl", "near_ooc": "golden_near_ooc.jsonl"}
PREFER = ("gpt-4.1", "gpt-4o", "gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini")
# ΕΚΤΙΜΗΣΗ $ ανά 1M tokens (είσοδος, έξοδος) — μόνο για τον φύλακα κόστους, όχι για αναφορά
PRICE = {"gpt-4.1": (2.0, 8.0), "gpt-4o": (2.5, 10.0), "gpt-5-mini": (0.25, 2.0), "gpt-5": (1.25, 10.0),
         "gpt-4.1-mini": (0.4, 1.6), "gpt-4o-mini": (0.15, 0.6)}
NOISE = {"main": 3, "mh_new": 4, "near_ooc": 2}

MAIN_J2 = """You are grading one answer of a question-answering system that must answer ONLY from a \
fixed set of research papers.

QUESTION: {question}
REFERENCE ANSWER (written by reading the complete papers; may be in Greek): {reference}

SYSTEM ANSWER (may be in Greek; markers like [S3] are citations and do not matter):
<<<
{answer}
>>>

Classify the SYSTEM ANSWER against the REFERENCE ANSWER with exactly one label:
- "correct": it gives the key information of the reference answer. Missing a secondary detail is fine.
- "partial": it gives some of the key information but misses an important part of it.
- "refused": it says that the documents do not contain the answer (or that it cannot find it) and \
gives no substantive answer.
- "wrong": its main claim contradicts the reference answer, or it answers a different question.
If the REFERENCE ANSWER itself says that the papers do not contain the answer, then an answer that \
says so is "correct".
Judge ONLY against the REFERENCE ANSWER. Ignore length, style, language and citations.

Return ONLY JSON, no prose:
{{"label": "correct|partial|refused|wrong", "evidence": "<ONE sentence copied VERBATIM from the \
SYSTEM ANSWER that decided the label, or empty>"}}
"""

MH_J2 = """You are grading one answer of a question-answering system that must answer ONLY from a \
fixed set of research papers.

The QUESTION needs information from TWO papers:
  PAPER 1 = {name1}
  PAPER 2 = {name2}

QUESTION: {question}
REFERENCE ANSWER (written by reading both complete papers; may be in Greek): {reference}
Key terms expected in the part about PAPER 1: {kw1}
Key terms expected in the part about PAPER 2: {kw2}

SYSTEM ANSWER (may be in Greek; markers like [S3] are citations and do not matter):
<<<
{answer}
>>>

For EACH paper, label the part of the SYSTEM ANSWER about that paper with exactly one of:
- "correct": it gives that paper's part of the reference answer.
- "partial": it gives only some of that part, or related information from that paper that misses \
the main point of the reference.
- "declared_missing": it clearly says that the documents do not contain that part.
- "omitted": it does not address that paper's part and does not say that it is missing.
- "wrong": it states something about that paper's part that contradicts the reference, or \
attributes to that paper something that belongs to another paper.
If the answer gives some useful information for a part AND says the rest is missing, the label is \
"partial". Judge ONLY against the REFERENCE ANSWER. Ignore length, style, language and citations.

Return ONLY JSON, no prose:
{{"paper_1": "<label>", "paper_2": "<label>",
 "evidence_1": "<ONE sentence copied VERBATIM from the SYSTEM ANSWER that decided the label of PAPER 1, or empty>",
 "evidence_2": "<the same for PAPER 2>"}}
"""


def literal_from(path: str, name: str):
    """Μια σταθερά (string / dict) από άλλο αρχείο ΧΩΡΙΣ import (εκείνα φορτώνουν gemini_rest κ.λπ.)."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} δεν βρέθηκε στο {path}")


NEAR_J = literal_from(os.path.join(HERE, "eval_near_ooc.py"), "JUDGE_PROMPT")
NAMES = literal_from(os.path.join(HERE, "scoreboard_answers.py"), "NAMES")


def api_key() -> str | None:
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                m = re.match(r"\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.+?)\s*$", line)
                if m:
                    return m.group(1).strip().strip('"').strip("'")
    return None


def http(url: str, key: str, body: dict | None = None) -> dict:
    # S310: το url είναι ΠΑΝΤΑ ένα από τα δύο σταθερά https://api.openai.com/... αυτού του αρχείου
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,  # noqa: S310
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:  # noqa: S310
        return json.load(r)


def list_models(key: str) -> list[str]:
    return sorted(m["id"] for m in http("https://api.openai.com/v1/models", key)["data"])


class Judge:
    def __init__(self, model: str, key: str, max_usd: float):
        self.model, self.key, self.max_usd = model, key, max_usd
        self.data = {}
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, encoding="utf-8") as f:
                self.data = json.load(f)
        self.new = self.hits = 0
        self.usd = 0.0

    def save(self) -> None:
        os.makedirs(CMP, exist_ok=True)
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, CACHE_PATH)

    def __call__(self, prompt: str) -> dict:
        k = hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()
        if k in self.data:
            self.hits += 1
            return json.loads(self.data[k]["out"])
        if self.usd >= self.max_usd:
            raise RuntimeError(f"όριο κόστους {self.max_usd} $")
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}}
        if not self.model.startswith(("o1", "o3", "o4", "gpt-5")):
            body["temperature"] = 0
        for attempt in range(4):
            try:
                d = http("https://api.openai.com/v1/chat/completions", self.key, body)
                break
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")[:300]
                if e.code == 429 and "insufficient_quota" not in msg and attempt < 3:
                    time.sleep(5 * (attempt + 1))
                    continue
                if e.code >= 500 and attempt < 3:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenAI {e.code}: {msg}") from e
        out = d["choices"][0]["message"]["content"]
        u = d.get("usage") or {}
        pin, pout = PRICE.get(self.model, (2.5, 10.0))
        self.usd += ((u.get("prompt_tokens") or 0) * pin + (u.get("completion_tokens") or 0) * pout) / 1e6
        self.new += 1
        self.data[k] = {"model": self.model, "tail": prompt[-150:], "out": out, "usage": u}
        return json.loads(out)


def load_jsonl(name: str) -> dict:
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return {t["id"]: t for t in (json.loads(x) for x in f if x.strip())}


def load_rows(system: str, sk: str) -> dict:
    with open(os.path.join(RUNS, SOURCES[system][sk]), encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)["rows"]}


def squash(text: str) -> str:
    return " " + " ".join(re.findall(r"\w+", (text or "").lower())) + " "


def evfound(ev: str, answer: str) -> int | None:
    """1/0 = το απόσπασμα βρέθηκε / ΔΕΝ βρέθηκε αυτούσιο· None = ο κριτής δεν έδωσε (επιτρέπεται)."""
    if not (ev or "").strip():
        return None
    return int(len(ev.split()) >= 3 and squash(ev) in squash(answer))


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def binom_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def judge_all(judge: Judge, limit: int) -> list[dict]:
    gold = {sk: load_jsonl(fn) for sk, fn in GOLD.items()}
    out = []
    for system in SOURCES:
        for sk in ("main", "mh_new", "near_ooc"):
            rows = load_rows(system, sk)
            ours = load_rows("ours", sk) if sk == "mh_new" else {}
            for qid in list(gold[sk])[:limit or None]:
                t, r = gold[sk][qid], rows.get(qid)
                if not r or not (r.get("answer") or "").strip():
                    print(f"  {system:<10} {sk:<8} {qid:<5} — χωρίς απάντηση (σφάλμα γέννησης), παραλείπεται")
                    continue
                ans = r["answer"]
                row = {"system": system, "set": sk, "id": qid, "lang": r.get("lang", "")}
                if sk == "main":
                    j = judge(MAIN_J2.format(question=t["question"], reference=t["reference_answer"],
                                             answer=ans))
                    row |= {"category": t.get("category", ""), "label": str(j.get("label", "")).lower(),
                            "evfound": evfound(j.get("evidence"), ans),
                            "j1_perfect": int(all(r.get(s) == 5 for s in
                                                  ("accuracy", "completeness", "relevance", "faithfulness")))}
                elif sk == "mh_new":
                    d1 = r.get("doc_1") or ours.get(qid, {}).get("doc_1")
                    d2 = r.get("doc_2") or ours.get(qid, {}).get("doc_2")
                    j = judge(MH_J2.format(name1=NAMES[d1], name2=NAMES[d2], question=t["question_en"],
                                           reference=t["reference_answer"],
                                           kw1=", ".join(t["keywords_by_doc"][d1]),
                                           kw2=", ".join(t["keywords_by_doc"][d2]), answer=ans))
                    row |= {"doc_1": d1, "doc_2": d2}
                    for h in (1, 2):
                        row |= {f"label_{h}": str(j.get(f"paper_{h}", "")).lower(),
                                f"j1_label_{h}": r.get(f"label_{h}", ""),
                                f"evfound_{h}": evfound(j.get(f"evidence_{h}"), ans)}
                else:
                    if r.get("label") == "cut":
                        row |= {"label": "cut", "j1_label": "cut", "evfound": 1}
                    else:
                        j = judge(NEAR_J.format(focus=t["focus"], note=t.get("review_note") or "",
                                                question=t["question"], question_en=t["question_en"],
                                                answer=ans))
                        row |= {"label": str(j.get("label", "")).lower(), "j1_label": r.get("label", ""),
                                "evfound": evfound(j.get("evidence"), ans)}
                out.append(row)
                judge.save()
                lab = row.get("label") or f"{row.get('label_1')}/{row.get('label_2')}"
                print(f"  {system:<10} {sk:<8} {qid:<5} {lab:<22} · νέες {judge.new} · ~{judge.usd:.3f} $", flush=True)
    return out


def report(rows: list[dict], model: str) -> dict:
    def of(system, sk):
        return [r for r in rows if r["system"] == system and r["set"] == sk]

    systems = list(SOURCES)
    summ = {}
    w0, w = 52, 14
    print("\n" + "#" * (w0 + w * len(systems)))
    print(f"ΔΕΥΤΕΡΟΣ ΚΡΙΤΗΣ: {model} (ως προς την ΑΠΑΝΤΗΣΗ ΑΝΑΦΟΡΑΣ, χωρίς context)")
    print(f"{'':<{w0}}" + "".join(f"{s:>{w}}" for s in systems))
    lines = []
    for s in systems:
        m = [r for r in of(s, "main") if r["category"] != "out_of_corpus"]
        ooc = [r for r in of(s, "main") if r["category"] == "out_of_corpus"]
        mh = of(s, "mh_new")
        halves = [r[f"label_{h}"] for r in mh for h in (1, 2)]
        nr = of(s, "near_ooc")
        summ[s] = {
            "κύριο: correct": f"{sum(r['label'] == 'correct' for r in m)}/{len(m)}",
            "κύριο: partial · refused · wrong": " · ".join(str(sum(r["label"] == x for r in m))
                                                        for x in ("partial", "refused", "wrong")),
            "κύριο εκτός σώματος: σωστή άρνηση": f"{sum(r['label'] in ('correct', 'refused') for r in ooc)}/{len(ooc)}",
            "δύο papers: σωστά μισά": f"{halves.count('correct')}/{len(halves)}",
            "δύο papers: «και τα δύο σωστά»": f"{sum(r['label_1'] == r['label_2'] == 'correct' for r in mh)}/{len(mh)}",
            "δύο papers: μισά «wrong»": f"{halves.count('wrong')}/{len(halves)}",
            "κοντινές: σωστή άρνηση": f"{sum(r['label'] in ('correct', 'cut') for r in nr)}/{len(nr)}",
            "κοντινές: ήπια διαρροή · διαρροή": f"{sum(r['label'] == 'soft_leak' for r in nr)} · "
                                                 f"{sum(r['label'] == 'leak' for r in nr)}",
        }
        lines = list(summ[s])
    for lab in lines:
        print(f"{lab:<{w0}}" + "".join(f"{summ[s][lab]:>{w}}" for s in systems))
    print("#" * (w0 + w * len(systems)))

    def units(system, sk):
        if sk == "main":
            return {r["id"]: r["label"] == "correct" for r in of(system, sk) if r["category"] != "out_of_corpus"}
        if sk == "mh_new":
            return {f"{r['id']}:{h}": r[f"label_{h}"] == "correct" for r in of(system, sk) for h in (1, 2)}
        return {r["id"]: r["label"] in ("correct", "cut") for r in of(system, sk)}

    print("\nΖΕΥΓΑΡΩΤΑ με τον 2ο κριτή (ours κερδίζει / χάνει · p · πάνω από θόρυβο;)")
    for s in systems[1:]:
        for sk in ("main", "mh_new", "near_ooc"):
            ua, ub = units("ours", sk), units(s, sk)
            common = ua.keys() & ub.keys()
            win = sorted(k for k in common if ua[k] and not ub[k])
            lose = sorted(k for k in common if ub[k] and not ua[k])
            diff = len(win) - len(lose)
            v = ("ours ΜΠΡΟΣΤΑ" if diff > NOISE[sk] else f"{s} ΜΠΡΟΣΤΑ" if -diff > NOISE[sk] else "θόρυβος")
            print(f"  {s:<11} {sk:<9} {len(win):>3} / {len(lose):<3} · p={binom_p(len(win), len(lose)):.3f} · {v}"
                  + (f"   χάνει στα: {' '.join(lose)}" if lose else ""))

    print("\nΣΥΜΦΩΝΙΑ ΤΩΝ ΔΥΟ ΚΡΙΤΩΝ (ανά μονάδα, όλα τα συστήματα μαζί)")
    a = [r[f"j1_label_{h}"] == "correct" for r in rows if r["set"] == "mh_new" for h in (1, 2)]
    b = [r[f"label_{h}"] == "correct" for r in rows if r["set"] == "mh_new" for h in (1, 2)]
    k = kappa(a, b)
    if a:
        print(f"  δύο papers, «σωστό μισό»: {sum(x == y for x, y in zip(a, b))}/{len(a)} "
              f"({100 * sum(x == y for x, y in zip(a, b)) / len(a):.0f}%) · κ={k:.2f}" if k is not None else "")
    nr = [r for r in rows if r["set"] == "near_ooc" and r["label"] != "cut"]
    a = [r["j1_label"] == "correct" for r in nr]
    b = [r["label"] == "correct" for r in nr]
    k = kappa(a, b)
    if a:
        print(f"  κοντινές, «σωστή άρνηση»: {sum(x == y for x, y in zip(a, b))}/{len(a)} "
              f"({100 * sum(x == y for x, y in zip(a, b)) / len(a):.0f}%) · κ={k:.2f}" if k is not None else "")
    m = [r for r in rows if r["set"] == "main" and r["category"] != "out_of_corpus"]
    if m:
        both = sum(r["j1_perfect"] and r["label"] == "correct" for r in m)
        only1 = [f"{r['system']}:{r['id']}" for r in m if r["j1_perfect"] and r["label"] != "correct"]
        only2 = [f"{r['system']}:{r['id']}" for r in m if not r["j1_perfect"] and r["label"] == "correct"]
        print(f"  κύριο: 1ος «5/5/5/5» και 2ος «correct» {both} · μόνο ο 1ος {len(only1)} · μόνο ο 2ος {len(only2)}")
        print(f"    μόνο ο 1ος (ύποπτα: η άρνηση/παράλειψη που ο 1ος επιβράβευσε): {' '.join(only1)}")
    bad = [f"{r['system']}:{r['set']}:{r['id']}" for r in rows
           if r.get("evfound") == 0 or r.get("evfound_1") == 0 or r.get("evfound_2") == 0]
    unknown = [f"{r['system']}:{r['id']}" for r in rows
               if any(x and x not in ("correct", "partial", "refused", "wrong", "declared_missing", "omitted",
                                      "soft_leak", "leak", "cut")
                      for x in (r.get("label"), r.get("label_1"), r.get("label_2")))]
    print(f"\nαπόσπασμα-απόδειξη που ΔΕΝ βρέθηκε αυτούσιο στην απάντηση: {len(bad)}"
          + (f" · ΑΓΝΩΣΤΕΣ ετικέτες: {unknown}" if unknown else ""))
    return summ


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="auto", help="μοντέλο OpenAI (auto = το 1ο διαθέσιμο από " + ", ".join(PREFER) + ")")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="N ανά σετ ανά σύστημα (δοκιμή· δεν γράφει αποτελέσματα)")
    ap.add_argument("--max-usd", type=float, default=3.0)
    args = ap.parse_args()
    key = api_key()
    if not key:
        print("!! Λείπει OPENAI_API_KEY (περιβάλλον ή .env της ρίζας)")
        return 1
    try:
        models = list_models(key)
    except urllib.error.HTTPError as e:
        print(f"!! OpenAI {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
        return 1
    if args.list_models:
        print("\n".join(m for m in models if m.startswith(("gpt", "o1", "o3", "o4", "chatgpt"))))
        return 0
    model = next((m for m in PREFER if m in models), None) if args.model == "auto" else args.model
    if not model or model not in models:
        print(f"!! Το μοντέλο {model or args.model} δεν είναι διαθέσιμο σε αυτό το κλειδί (--list-models)")
        return 1
    print(f"Κριτής: {model} · εκτιμώμενη τιμή {PRICE.get(model, '?')} $/1M · όριο {args.max_usd} $")
    judge = Judge(model, key, args.max_usd)
    try:
        rows = judge_all(judge, args.limit)
    except RuntimeError as e:
        judge.save()
        print(f"!! {e} — σταματάω (ό,τι κρίθηκε σώθηκε· το ξανατρέξιμο συνεχίζει με 0 κόστος)")
        return 1
    judge.save()
    summ = report(rows, model)
    print(f"\nOpenAI: {judge.new} νέες κλήσεις / {judge.hits} παγωμένες · ~{judge.usd:.3f} $ (εκτίμηση)")
    if args.limit:
        print("(--limit: δεν γράφεται αρχείο αποτελεσμάτων)")
        return 0
    out = os.path.join(CMP, f"judge2_{model}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": model, "when": time.strftime("%Y-%m-%d %H:%M"), "summary": summ, "rows": rows},
                  f, ensure_ascii=False, indent=1)
    print(f"Σώθηκε: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

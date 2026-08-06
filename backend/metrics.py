"""Συσσωρευμένα metrics σε Prometheus text format — ΜΗΔΕΝ εξαρτήσεις.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ:
Το σύστημα ήδη μετράει ΜΙΑ ερώτηση (latency ανά φάση, tokens -> logs + UI). Δεν
ξέρει τίποτα για το ΣΥΝΟΛΟ. Τρία ερωτήματα που δεν μπορούσαν να απαντηθούν:
  - πόσο συχνά κόβει το relevance gate στην πραγματική χρήση;
  - πόσο συχνά τρέχει ο corrective agent και πόσο συχνά πετυχαίνει;
  - έγινε το σύστημα πιο αργό απ' ό,τι ήταν;
Το πρώτο είναι το πιο ενδιαφέρον νούμερο του συστήματος — μετρήθηκε ΜΟΝΟ σε
golden sets, ποτέ σε πραγματικούς χρήστες.

ΓΙΑΤΙ ΟΧΙ prometheus_client:
+1 εξάρτηση για ~90 γραμμές κειμένου. Το exposition format είναι απλό text και
σταθερό εδώ και χρόνια. Το endpoint είναι συμβατό με Prometheus/Grafana ό,τι κι
αν τρέξει από πάνω. (Το Langfuse είχε ήδη απορριφθεί: 6 containers για 50 traces.)

THREAD SAFETY: ο reranker και το retrieval τρέχουν σε asyncio.to_thread, άρα οι
μετρητές γράφονται από ΠΟΛΛΑ threads. Ένα Lock γύρω από dict updates — το κόστος
είναι νανοδευτερόλεπτα έναντι 450ms retrieval.

ΠΡΟΣΟΧΗ: in-process, όπως το rate limiting. Χάνεται σε restart και ΔΕΝ
μοιράζεται μεταξύ workers. Με `--workers 1` δεν έχει σημασία· αν ποτέ ανέβουν
οι workers, θέλει Redis ή push σε Pushgateway.
"""
import threading
import time

_lock = threading.Lock()
_counters: dict[tuple, float] = {}
_summaries: dict[tuple, list] = {}   # key -> [count, sum, max]
_START = time.time()

# Τεκμηρίωση ανά metric — μπαίνει στο # HELP της εξόδου ώστε να μη χρειάζεται
# να ανοίξει κανείς αυτό το αρχείο για να καταλάβει τι βλέπει στο Grafana.
_HELP = {
    "rag_queries_total": "Ερωτήσεις που έφτασαν στο retrieval",
    "rag_gate_blocked_total": "Φορές που το relevance gate έκοψε (1ο pass)",
    "rag_corrective_attempts_total": "Φορές που ξεκίνησε corrective retry",
    "rag_corrective_success_total": "Corrective retries που πέρασαν το αυστηρό κατώφλι",
    "rag_corrective_skipped_total": "Corrective retries που δεν έτρεξαν καν (ανά αιτία)",
    "rag_answers_total": "Απαντήσεις που στάλθηκαν στον χρήστη (ανά έκβαση)",
    "rag_gemini_errors_total": "Σφάλματα από το Gemini κατά τη γέννηση",
    "rag_tokens_total": "Tokens ανά είδος (FinOps)",
    "rag_retrieval_seconds": "Χρόνος φάσης ανάκτησης",
    "rag_generation_seconds": "Χρόνος φάσης γέννησης",
}


def _key(name: str, labels: dict | None):
    return (name, tuple(sorted((labels or {}).items())))


def inc(name: str, labels: dict | None = None, value: float = 1.0):
    """Μονότονος μετρητής. Ποτέ δεν πρέπει να σκάσει: τα metrics δεν αξίζουν
    ούτε μία αποτυχημένη απάντηση."""
    with _lock:
        k = _key(name, labels)
        _counters[k] = _counters.get(k, 0.0) + value


def observe(name: str, value: float, labels: dict | None = None):
    """Summary: count + sum + max. ΟΧΙ πλήρη histogram buckets — για p95 θα
    χρειαζόταν προκαθορισμένα buckets, και δεν έχουμε ακόμα δεδομένα παραγωγής
    για να τα διαλέξουμε σωστά. Το max πιάνει ήδη το χειρότερο περιστατικό."""
    with _lock:
        k = _key(name, labels)
        s = _summaries.get(k)
        if s is None:
            _summaries[k] = [1, value, value]
        else:
            s[0] += 1
            s[1] += value
            s[2] = max(s[2], value)


def reset():
    """ΜΟΝΟ για tests — η παραγωγή δεν μηδενίζει ποτέ μετρητές."""
    with _lock:
        _counters.clear()
        _summaries.clear()


def _fmt_labels(label_tuple) -> str:
    if not label_tuple:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in label_tuple)
    return "{" + inner + "}"


def _fmt_value(v: float) -> str:
    """Ακέραιοι χωρίς επιστημονική σημειογραφία: το %g έβγαζε 1.4014e+06 για
    1.401.400 tokens. Έγκυρο για τον scraper, αλλά αδιάβαστο για άνθρωπο — και
    το FinOps νούμερο είναι ακριβώς αυτό που κοιτάει άνθρωπος."""
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def render() -> str:
    """Prometheus text exposition format (version 0.0.4)."""
    with _lock:
        counters = dict(_counters)
        summaries = {k: list(v) for k, v in _summaries.items()}

    lines: list[str] = []
    seen: set[str] = set()

    for name in sorted({k[0] for k in counters}):
        if name not in seen:
            lines.append(f"# HELP {name} {_HELP.get(name, name)}")
            lines.append(f"# TYPE {name} counter")
            seen.add(name)
        for (n, lbl), v in sorted(counters.items()):
            if n == name:
                lines.append(f"{name}{_fmt_labels(lbl)} {_fmt_value(v)}")

    for name in sorted({k[0] for k in summaries}):
        lines.append(f"# HELP {name} {_HELP.get(name, name)}")
        lines.append(f"# TYPE {name} summary")
        for (n, lbl), (cnt, total, _mx) in sorted(summaries.items()):
            if n != name:
                continue
            lb = _fmt_labels(lbl)
            lines.append(f"{name}_count{lb} {cnt:g}")
            lines.append(f"{name}_sum{lb} {total:.6f}")
        # Το _max δεν είναι στο πρότυπο του summary — εκτίθεται ως ξεχωριστό
        # gauge. Το # TYPE δηλώνεται ΜΙΑ φορά ανά metric family (έξω από τον
        # βρόχο των labels): επανάληψη = διπλή δήλωση, που ο scraper απορρίπτει.
        lines.append(f"# TYPE {name}_max gauge")
        for (n, lbl), (_c, _s, mx) in sorted(summaries.items()):
            if n == name:
                lines.append(f"{name}_max{_fmt_labels(lbl)} {mx:.6f}")

    lines.append("# HELP rag_uptime_seconds Χρόνος ζωής της διεργασίας")
    lines.append("# TYPE rag_uptime_seconds gauge")
    lines.append(f"rag_uptime_seconds {time.time() - _START:.1f}")
    return "\n".join(lines) + "\n"


def snapshot() -> dict:
    """Ίδια δεδομένα σε JSON — για ανθρώπους και για tests. Τα παράγωγα
    ποσοστά υπολογίζονται εδώ ΜΙΑ φορά, ώστε να μη χρειάζεται ο καθένας να
    ξαναγράφει τον ίδιο διαιρέτη (και να τον κάνει λάθος)."""
    with _lock:
        counters = {f"{k[0]}{_fmt_labels(k[1])}": v for k, v in _counters.items()}
        summaries = {
            f"{k[0]}{_fmt_labels(k[1])}": {
                "count": v[0], "sum": round(v[1], 3),
                "avg": round(v[1] / v[0], 3) if v[0] else 0.0,
                "max": round(v[2], 3)}
            for k, v in _summaries.items()}
        q = _counters.get(_key("rag_queries_total", None), 0.0)
        blocked = _counters.get(_key("rag_gate_blocked_total", None), 0.0)
        att = _counters.get(_key("rag_corrective_attempts_total", None), 0.0)
        ok = _counters.get(_key("rag_corrective_success_total", None), 0.0)

    return {
        "counters": counters,
        "summaries": summaries,
        "derived": {
            "gate_block_rate": round(blocked / q, 4) if q else None,
            "corrective_success_rate": round(ok / att, 4) if att else None,
            "answered_after_correction": ok,
        },
        "uptime_seconds": round(time.time() - _START, 1),
    }

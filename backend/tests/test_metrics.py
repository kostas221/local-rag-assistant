"""Metrics: συσσώρευση, Prometheus format, thread safety.

ΤΙ ΠΡΟΣΤΑΤΕΥΕΙ: τα metrics είναι ο ΜΟΝΟΣ τρόπος να δεις τι κάνει το σύστημα
στην παραγωγή. Δύο σιωπηλές αποτυχίες:
  1. λάθος format -> ο scraper δεν διαβάζει τίποτα, και δεν το μαθαίνεις ποτέ
     (κανείς δεν κοιτά το /metrics με το μάτι)
  2. χαμένες μετρήσεις υπό concurrency -> τα νούμερα είναι λάθος αλλά φαίνονται
     εύλογα, που είναι χειρότερο από το να λείπουν

Μηδέν εξαρτήσεις (ούτε ai_core) -> τρέχει στο fast CI job.
"""
import threading

import metrics


def setup_function():
    metrics.reset()


def test_counter_accumulates():
    for _ in range(3):
        metrics.inc("rag_queries_total")
    snap = metrics.snapshot()
    assert snap["counters"]["rag_queries_total"] == 3


def test_labels_are_separate_series():
    """Ίδιο όνομα + άλλο label = ΑΛΛΗ σειρά. Αλλιώς το
    rag_corrective_skipped_total{reason=...} θα άθροιζε τις αιτίες σε ένα
    νούμερο και δεν θα ήξερες ΓΙΑΤΙ δεν έτρεξε ο agent."""
    metrics.inc("rag_corrective_skipped_total", {"reason": "identical"})
    metrics.inc("rag_corrective_skipped_total", {"reason": "identical"})
    metrics.inc("rag_corrective_skipped_total", {"reason": "disabled"})
    c = metrics.snapshot()["counters"]
    assert c['rag_corrective_skipped_total{reason="identical"}'] == 2
    assert c['rag_corrective_skipped_total{reason="disabled"}'] == 1


def test_summary_tracks_count_sum_max():
    for v in (0.5, 1.5, 1.0):
        metrics.observe("rag_retrieval_seconds", v)
    s = metrics.snapshot()["summaries"]["rag_retrieval_seconds"]
    assert s["count"] == 3
    assert s["sum"] == 3.0
    assert s["avg"] == 1.0
    assert s["max"] == 1.5


def test_derived_rates():
    """Τα δύο νούμερα για τα οποία φτιάχτηκε ολόκληρο το module."""
    for _ in range(10):
        metrics.inc("rag_queries_total")
    for _ in range(4):
        metrics.inc("rag_gate_blocked_total")
        metrics.inc("rag_corrective_attempts_total")
    metrics.inc("rag_corrective_success_total")
    d = metrics.snapshot()["derived"]
    assert d["gate_block_rate"] == 0.4
    assert d["corrective_success_rate"] == 0.25


def test_derived_rates_are_none_without_traffic():
    """Μηδέν ερωτήσεις -> None, ΟΧΙ 0.0. Το 0% block rate και το «δεν ξέρω
    ακόμα» είναι διαφορετικά πράγματα· ένα dashboard που τα μπερδεύει δείχνει
    «όλα καλά» σε σύστημα που δεν δέχτηκε ποτέ κίνηση."""
    d = metrics.snapshot()["derived"]
    assert d["gate_block_rate"] is None
    assert d["corrective_success_rate"] is None


def test_prometheus_format_is_parseable():
    """Κάθε γραμμή είναι είτε σχόλιο είτε `όνομα[{labels}] τιμή`. Αν σπάσει
    αυτό, ο scraper πετά ΟΛΟ το scrape, όχι μόνο τη μία γραμμή."""
    metrics.inc("rag_queries_total", value=7)
    metrics.inc("rag_tokens_total", {"kind": "prompt"}, 1234)
    metrics.observe("rag_generation_seconds", 2.5)
    text = metrics.render()

    assert "# TYPE rag_queries_total counter" in text
    assert "rag_queries_total 7" in text
    assert 'rag_tokens_total{kind="prompt"} 1234' in text
    assert "rag_generation_seconds_count 1" in text
    assert "rag_generation_seconds_sum 2.500000" in text

    for line in text.strip().split("\n"):
        if line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        assert len(parts) == 2, f"μη έγκυρη γραμμή: {line!r}"
        float(parts[1])  # σκάει αν η τιμή δεν είναι αριθμός


def test_every_metric_has_help_text():
    """Metric χωρίς # HELP είναι metric που κανείς δεν θα καταλάβει σε 6 μήνες."""
    metrics.inc("rag_gate_blocked_total")
    metrics.observe("rag_retrieval_seconds", 1.0)
    lines = metrics.render().split("\n")
    names = {ln.split()[2] for ln in lines if ln.startswith("# HELP")}
    for expected in ("rag_gate_blocked_total", "rag_retrieval_seconds"):
        assert expected in names
        help_line = next(ln for ln in lines if ln.startswith(f"# HELP {expected} "))
        assert len(help_line.split(" ", 3)[3]) > 10, "κενή/άχρηστη περιγραφή"


def test_concurrent_increments_lose_nothing():
    """20 threads x 500 -> ΑΚΡΙΒΩΣ 10.000. Ο reranker και το retrieval τρέχουν
    σε asyncio.to_thread, άρα αυτό ΔΕΝ είναι θεωρητικό σενάριο."""
    def worker():
        for _ in range(500):
            metrics.inc("rag_queries_total")
            metrics.observe("rag_retrieval_seconds", 0.1)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = metrics.snapshot()
    assert snap["counters"]["rag_queries_total"] == 10_000
    assert snap["summaries"]["rag_retrieval_seconds"]["count"] == 10_000

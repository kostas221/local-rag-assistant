"""Load test: προσομοιώνει N ταυτόχρονους χρήστες που ρωτούν το /chat, ώστε να
δεις πώς συμπεριφέρεται το σύστημα όταν π.χ. 5 άτομα ρωτούν την ΙΔΙΑ στιγμή
(πριν το deploy στον server της σχολής).

Μετράει ανά αίτημα 3 χρόνους που αντιστοιχούν στις φάσεις του pipeline:
  - t_sources : ώρα μέχρι το 1ο packet (sources) ≈ φάση ΑΝΑΚΤΗΣΗΣ (retrieval)
  - t_first   : ώρα μέχρι το 1ο token κειμένου ≈ έναρξη ΠΑΡΑΓΩΓΗΣ (Gemini)
  - t_total   : συνολικός χρόνος απάντησης
Και συγκεντρωτικά: median / p95 + ποσοστό σφαλμάτων + throughput.

Δεν χρειάζεται νέα dependency (μόνο requests). Τρέξε από τον host:
    python backend/loadtest.py --register --users 5
ή μέσα στο container:
    docker compose exec backend python loadtest.py --register --users 5

ΣΗΜ: αν ΔΕΝ υπάρχουν "ready" PDFs, το gate γυρνά αμέσως "no documents" και
παρακάμπτει retrieval+generation — οπότε ανέβασε 1-2 PDF πρώτα για ρεαλιστική μέτρηση.
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def _pct(values, p):
    """Απλό percentile (nearest-rank) — αρκεί για load test, χωρίς numpy."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, round((p / 100) * (len(s) - 1))))
    return s[k]


def _login(base, username, password, do_register):
    if do_register:
        # Best-effort: αν υπάρχει ήδη ο χρήστης, το 400 αγνοείται.
        requests.post(f"{base}/register", json={
            "username": username, "email": f"{username}@loadtest.local",
            "password": password}, timeout=30)
    res = requests.post(f"{base}/login",
                        data={"username": username, "password": password}, timeout=30)
    if res.status_code == 401:
        raise SystemExit(
            f"❌ Login απέτυχε (401) για '{username}': λάθος username/password.\n"
            "   • Έλεγξε για κενά/κόμμα στον κωδικό.\n"
            "   • Ή φτιάξε στο UI λογαριασμό 'loadtester' (password 'loadtest123'),\n"
            "     ανέβασέ του ΕΝΑ PDF, και τρέξε ΧΩΡΙΣ --username/--password.")
    res.raise_for_status()
    return res.json()["access_token"]


def _ready_filenames(base, headers):
    res = requests.get(f"{base}/documents", headers=headers, timeout=30)
    res.raise_for_status()
    return [d["file_name"] for d in res.json() if d.get("status") == "ready"]


def _one_request(base, headers, conv_id, question, targets, persona):
    """Στέλνει ΕΝΑ streaming /chat και μετράει τους 3 χρόνους φάσης."""
    payload = {
        "conversation_id": conv_id,
        "question": question,
        "history": [],
        "target_filenames": targets,
        "persona": persona,
    }
    t0 = time.perf_counter()
    t_sources = t_first = None
    chars = 0
    try:
        with requests.post(f"{base}/chat", json=payload, headers=headers,
                           stream=True, timeout=180) as r:
            if r.status_code != 200:
                return {"ok": False, "status": r.status_code,
                        "err": r.text[:160]}
            for line in r.iter_lines():
                if not line:
                    continue
                now = time.perf_counter() - t0
                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "sources" and t_sources is None:
                    t_sources = now
                elif data.get("type") == "text":
                    if t_first is None:
                        t_first = now
                    chars += len(data.get("data", ""))
        return {"ok": True, "t_sources": t_sources, "t_first": t_first,
                "t_total": time.perf_counter() - t0, "chars": chars}
    except Exception as e:
        return {"ok": False, "err": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--username", default="loadtester")
    ap.add_argument("--password", default="loadtest123")
    ap.add_argument("--register", action="store_true",
                    help="Φτιάξε τον χρήστη πρώτα (αγνοεί 'υπάρχει ήδη').")
    ap.add_argument("--users", type=int, default=5,
                    help="Ταυτόχρονοι χρήστες (concurrency).")
    ap.add_argument("--requests", type=int, default=None,
                    help="Συνολικά αιτήματα (default = --users, δηλ. ένα κύμα).")
    ap.add_argument("--question", default="What is serverless computing?")
    ap.add_argument("--persona", default="Concise",
                    help="Concise = πιο μικρές απαντήσεις -> πιο γρήγορο test.")
    args = ap.parse_args()

    total = args.requests or args.users
    base = args.base_url.rstrip("/")

    print(f"--> Login ({args.username}) @ {base}")
    token = _login(base, args.username, args.password, args.register)
    headers = {"Authorization": f"Bearer {token}"}

    targets = _ready_filenames(base, headers)
    if not targets:
        print(f"⚠️  Ο λογαριασμός '{args.username}' δεν βλέπει κανένα 'ready' PDF.\n"
              "    Τα PDF είναι ΙΔΙΩΤΙΚΑ ανά χρήστη — αν τα ανέβασες από το UI με\n"
              "    άλλο λογαριασμό, τρέξε με τα ΔΙΚΑ ΣΟΥ στοιχεία:\n"
              "      --username <ο λογαριασμός σου> --password <ο κωδικός σου>\n"
              "    (Χωρίς ready PDF το gate γυρνά αμέσως 'no documents' και ΔΕΝ\n"
              "    μετριέται retrieval/generation.)")
    else:
        print(f"--> {len(targets)} ready PDFs: {', '.join(targets[:3])}"
              f"{'…' if len(targets) > 3 else ''}")

    conv_id = requests.post(f"{base}/conversations", headers=headers,
                            json={"title": "loadtest"}, timeout=30).json()["id"]

    # Warmup (δεν μετριέται): «ζεσταίνει» μοντέλα/caches ώστε το 1ο αίτημα του
    # κύματος να μην πληρώνει το cold start.
    print("--> Warmup (discarded)…")
    _one_request(base, headers, conv_id, args.question, targets, args.persona)

    print(f"--> Εκτέλεση: {total} αιτήματα, concurrency={args.users}\n")
    wall0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.users) as pool:
        futs = [pool.submit(_one_request, base, headers, conv_id,
                            args.question, targets, args.persona)
                for _ in range(total)]
        for fut in as_completed(futs):
            results.append(fut.result())
    wall = time.perf_counter() - wall0

    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]

    def stat(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        if not vals:
            return "—"
        return f"median {statistics.median(vals):.2f}s | p95 {_pct(vals, 95):.2f}s | max {max(vals):.2f}s"

    print("=" * 64)
    print(f"ΑΠΟΤΕΛΕΣΜΑΤΑ — {len(ok)}/{total} ΟΚ, {len(bad)} σφάλματα, "
          f"concurrency={args.users}")
    print("=" * 64)
    print(f"  Retrieval (t_sources): {stat('t_sources')}")
    print(f"  Time-to-1st-token    : {stat('t_first')}")
    print(f"  Total response       : {stat('t_total')}")
    print(f"  Throughput           : {len(ok) / wall:.2f} req/s "
          f"(wall {wall:.1f}s)")
    if bad:
        print("\n  Σφάλματα (δείγμα):")
        for r in bad[:5]:
            print(f"    - {r.get('status', '')} {r.get('err', '')}")
    print("=" * 64)
    print("Tip: τρέξε με --users 1 και μετά --users 5 — αν το 'Total' ανεβαίνει "
          "πολύ, βλέπεις serialization (CPU-bound retrieval -> βλ. βελτίωση #2).")


if __name__ == "__main__":
    main()

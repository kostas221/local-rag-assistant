"""Βήμα 7, έλεγχος ΠΡΙΝ από το τρέξιμο (host, Git Bash, μόνο stdlib):

    python backend/evaluation/heldout_preflight.py            # κλειστό σετ
    python backend/evaluation/heldout_preflight.py --smoke    # δοκιμή του εργαλείου (άρθρα cloud)

ΓΙΑΤΙ ΕΔΩ ΚΑΙ ΟΧΙ ΣΤΟ DOCKER: το container βλέπει μόνο το `backend/` (χωρίς `.git`), άρα δεν μπορεί να
ελέγξει μόνο του ότι τρέχει το παγωμένο σύστημα. Εδώ ελέγχουμε με το git και γράφουμε το αποτύπωμα
(SHA-256) κάθε αρχείου του συστήματος στο `private/preflight.json`. Το `run_heldout.py` ξαναϋπολογίζει
τα ίδια αποτυπώματα μέσα στο container και σταματά σε κάθε διαφορά: έτσι αποδεικνύεται ότι έτρεξαν
ακριβώς τα αρχεία του `c15a395`.

Έλεγχοι (σταματά στο πρώτο σφάλμα, χωρίς να τυπώσει περιεχόμενο ερωτήσεων):
  1. το `c15a395` είναι πρόγονος του HEAD·
  2. κανένα αρχείο του συστήματος (backend εκτός evaluation/ και tests/) δεν διαφέρει από το `c15a395`,
     ούτε στα commits ούτε στο working tree, και δεν υπάρχει νέο αρχείο συστήματος εκτός git·
  3. (κλειστό σετ) τα SHA-256 του CSV και του heldout_sources.json είναι αυτά της 19/11.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heldout_common as H

ROOT = os.path.dirname(H.BACKEND)


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")  # noqa: S603, S607
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")       # Git Bash στα Windows: αλλιώς σπάει στο «✓»
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="δοκιμή του εργαλείου στα άρθρα cloud")
    args = ap.parse_args()
    mode = "smoke" if args.smoke else "heldout"
    cfg = H.MODES[mode]
    errors = []

    head = git("rev-parse", "HEAD").strip()
    freeze = git("rev-parse", H.FREEZE_COMMIT).strip()
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", freeze, head], cwd=ROOT)  # noqa: S603, S607
    if anc.returncode != 0:
        errors.append(f"το {H.FREEZE_COMMIT} δεν είναι πρόγονος του HEAD")

    tracked = [p for p in git("ls-files", "backend").splitlines() if H.is_system_file(p)]
    changed = [p for p in git("diff", "--name-only", freeze, "--", "backend").splitlines()
               if H.is_system_file(p)]
    if changed:
        errors.append(f"αρχεία συστήματος διαφέρουν από το {H.FREEZE_COMMIT}: {', '.join(changed)}")
    untracked = [ln[3:] for ln in git("status", "--porcelain", "--", "backend").splitlines()
                 if ln.startswith("??") and H.is_system_file(ln[3:])]
    if untracked:
        errors.append(f"νέα αρχεία συστήματος εκτός git: {', '.join(untracked)}")

    if mode == "heldout":
        for key, path in (("csv_sha", cfg["csv"]), ("sources_sha", cfg["sources"])):
            got = H.sha256_file(path)
            if got != cfg[key]:
                errors.append(f"{os.path.basename(path)}: SHA-256 {got[:12]}… ≠ κλείδωμα {cfg[key][:12]}…")
    elif not os.path.exists(cfg["csv"]):
        errors.append(f"λείπει το {cfg['csv']}")

    if errors:
        print("ΣΤΑΜΑΤΑ — ο έλεγχος απέτυχε:")
        for e in errors:
            print("  ✗", e)
        return 1

    files = {}
    for rel in tracked:
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            files[rel[len("backend/"):]] = H.sha256_file(path)
    out = {"mode": mode, "head": head, "freeze_commit": freeze, "when": time.strftime("%Y-%m-%d %H:%M"),
           "system_files": files,
           "csv_sha": H.sha256_file(cfg["csv"]),
           "sources_sha": H.sha256_file(cfg["sources"]) if cfg["sources"] else None}
    path = os.path.join(H.private_dir(mode), "preflight.json")
    H.write_json_atomic(path, out)
    print(f"✓ {mode}: HEAD {head[:7]}, σύστημα ίδιο με το {H.FREEZE_COMMIT} ({len(files)} αρχεία)")
    if mode == "heldout":
        print("✓ CSV και heldout_sources.json: ίδια αποτυπώματα με το κλείδωμα της 19/11")
    print(f"  γράφτηκε {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

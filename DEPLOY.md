# DEPLOY.md — Οδηγός ανάπτυξης (Z-AI Platform)

Ο **κώδικας & το config** του repo είναι έτοιμα για deploy. Ό,τι μένει είναι
**ενέργειες πάνω στον server** (δεν γίνονται από το repo). Προτεραιότητες: 🔴 BLOCKING · 🟡 IMPORTANT · 🟢 nice-to-have.

---

## 📋 Βήμα 0 — Τι να μάθω από τον καθηγητή/admin (ΠΡΙΝ αγγίξω τον server)

Πριν κάνω οποιαδήποτε ενέργεια, μαζεύω αυτές τις πληροφορίες. Κάθε απάντηση «κλειδώνει» μια απόφαση παρακάτω.

### 🔴 Make-or-break (χωρίς αυτά δεν τρέχει τίποτα)

**1. Εξερχόμενο internet / Gemini.**
> *Ρώτα:* «Ο server έχει **εξερχόμενη** (outbound) πρόσβαση στο internet; Μπορεί να καλέσει το Google Gemini API (`generativelanguage.googleapis.com`); Χρειάζεται **proxy**;»
- **Γιατί:** Η ανάκτηση (embeddings, BM25, reranker) είναι 100% τοπική, αλλά η **παραγωγή της απάντησης, η μετάφραση ελληνικών ερωτήσεων και το judging** καλούν το Gemini στο cloud. Αν ο server είναι κλειστός προς τα έξω (πολύ συχνό σε πανεπιστημιακά δίκτυα), η εφαρμογή θα κάνει retrieval αλλά **δεν θα βγάζει απάντηση** — θα κολλάει/σφάλλει στη γέννηση.
- **Τι αλλάζει:** Αν θέλει proxy → βάζω `HTTPS_PROXY` env. Αν είναι τελείως κλειστό → ζητάω **firewall exception** για το Google API, αλλιώς το app δεν λειτουργεί. Γρήγορο test στον server: `curl https://generativelanguage.googleapis.com`.

**2. GPU ή CPU;**
> *Ρώτα:* «Το μηχάνημα έχει **NVIDIA GPU**; Αν ναι, ποια κάρτα και πόση **VRAM**; Πόσα **CPU cores** και πόση **RAM** μου αναλογούν;»
- **Γιατί:** Το βαρύ κομμάτι (reranker cross-encoder) είναι ~10-50× ταχύτερο σε GPU. Καθορίζει όλο το performance path (βλ. ενότητα CPU vs GPU παρακάτω). Η RAM είναι κρίσιμη: τα μοντέλα θέλουν **~5GB**.
- **Τι αλλάζει:** GPU → passthrough στο container + CUDA torch, και τρέχω `RERANK_CANDIDATES=15` ελεύθερα. CPU → optimization (ONNX) + `RERANK_CANDIDATES=10` υπό φόρτο.

**3. Docker + δικαιώματα.**
> *Ρώτα:* «Είναι εγκατεστημένα **Docker** και **Docker Compose**; Έχω δικαίωμα να τρέχω containers (sudo ή στο `docker` group); Αν υπάρχει GPU, υπάρχει **nvidia-container-toolkit**;»
- **Γιατί:** Όλη η εφαρμογή τρέχει **μόνο** μέσω Docker Compose. Χωρίς αυτά ή χωρίς δικαιώματα, δεν ξεκινάς καν.
- **Τι αλλάζει:** Αν λείπουν → ζητάω εγκατάσταση (ή τρέχω rootless docker αν επιτρέπεται).

### 🟡 Πρόσβαση χρηστών & δίκτυο

**4. Πώς θα φτάνουν οι χρήστες;**
> *Ρώτα:* «Ο server έχει **δημόσια IP/domain** ή είναι προσβάσιμος μόνο μέσα στο **δίκτυο/VPN** της σχολής;»
- **Γιατί:** Αλλάζει τελείως πώς δοκιμάζουν οι χρήστες την εφαρμογή. Δημόσιο = χρειάζεσαι domain + HTTPS. Intranet/VPN = δουλεύει μόνο για όσους είναι στο δίκτυο.
- **Τι αλλάζει:** Δημόσιο → στήνω reverse proxy + HTTPS σε πραγματικό domain. Intranet → απλούστερο, ίσως αρκεί IP:port μέσα στο δίκτυο.

**5. Ports & HTTPS.**
> *Ρώτα:* «Ποια **ports** μπορώ να ανοίξω στο firewall; Υπάρχει ήδη **reverse proxy** (nginx) ή **πιστοποιητικό** σχολής, ή τα στήνω εγώ;»
- **Γιατί:** Το UI ακούει στο **8501**, το API στο **8000**. Για πραγματικούς χρήστες θες ένα proxy (Caddy/nginx) μπροστά που δίνει `https://…` (port 443) — αλλιώς οι κωδικοί ταξιδεύουν **plaintext**.
- **Τι αλλάζει:** Αν επιτρέπεται μόνο 443 → όλα πίσω από proxy (το repo ήδη το υποστηρίζει: `ALLOWED_ORIGINS`, frontend σε `127.0.0.1`).

### 🟢 Πόροι & διάρκεια

**6. Disk & persistence.**
> *Ρώτα:* «Πόσος **χώρος δίσκου** μου αναλογεί; Είναι **persistent** μετά από reboot; Υπάρχουν **backups**;»
- **Γιατί:** Χρειάζεσαι ~5GB για μοντέλα + χώρο για PDFs χρηστών + Postgres + ChromaDB volumes. Αν ο χώρος δεν είναι persistent, **χάνεις δεδομένα & μοντέλα** σε κάθε restart (και ξανακατεβαίνουν 4.5GB).
- **Τι αλλάζει:** Persistent volumes υποχρεωτικά· αλλιώς προσωρινή λύση μόνο για demo.

**7. Κοινόχρηστος / quotas.**
> *Ρώτα:* «Το μηχάνημα είναι **κοινόχρηστο**; Υπάρχουν όρια **RAM/CPU**; Θα τερματίζεται το container αν ξεπεράσω όριο;»
- **Γιατί:** Με 5 ταυτόχρονους χρήστες + βαριά μοντέλα, ένα όριο RAM μπορεί να ρίξει το container (**OOM kill**).
- **Τι αλλάζει:** Αν υπάρχουν quotas → ρυθμίζω `RERANK_CANDIDATES` χαμηλά + memory limits στο compose.

**8. Uptime & auto-restart.**
> *Ρώτα:* «Ο server θα μένει **αναμμένος 24/7**; Κάνει **auto-restart** στο reboot; Ποιος είναι ο **admin** για προβλήματα και πώς συνδέομαι (**SSH**);»
- **Γιατί:** Ακαδημαϊκά μηχανήματα κάνουν sleep/reboot/συντήρηση. Αν σβήσει πριν την παρουσίαση ή ενώ το δοκιμάζουν χρήστες, χάθηκε το demo.
- **Τι αλλάζει:** Βάζω `restart: unless-stopped` στα containers + επιβεβαιώνω ότι σηκώνονται μόνα τους μετά από reboot.

### 💡 Έννοιες να ξέρω μπαίνοντας στη συζήτηση
- **Inbound vs Outbound:** *inbound* = να φτάνουν οι **χρήστες σ' εμένα** (ports/firewall). *outbound* = να φτάνω **εγώ στο Gemini**. Χρειάζομαι **και τα δύο** — είναι ξεχωριστά πράγματα.
- **«Ο server έχει GPU» ≠ «το container βλέπει GPU».** Χρειάζεται ρητό **passthrough** στο docker-compose + nvidia toolkit. Γι' αυτό ρωτάω και τα δύο.
- **Reverse proxy / HTTPS:** ένα nginx/Caddy μπροστά που δίνει `https://…` και προωθεί εσωτερικά στο 8501. Απαραίτητο για πραγματικούς χρήστες (κρυπτογράφηση κωδικών).

---

## ✅ Ολοκληρώθηκαν (code / config — στο repo)

**Ασφάλεια & deploy-hardening**
- **Πόρτες:** `db` (5432) & `backend` (8000) δεμένα σε `127.0.0.1` (όχι εκτεθειμένα στο δίκτυο)· `frontend` (8501) = το UI.
- **CORS env-configurable:** `ALLOWED_ORIGINS` (comma-separated, default τοπικό Streamlit).
- **Rate-limiting login** με bounded memory (όχι unbounded leak).
- **Orphaned upload files:** το `delete_document` σβήνει πλέον και το PDF από τον δίσκο.
- **Ingest recovery στο boot:** ορφανά «processing» → «failed» μετά από restart.
- Required env (raise on missing): `SECRET_KEY`, `DATABASE_URL`.

**Πυρήνας RAG & performance**
- Unit tests για relevance gate + RRF.
- **BM25 index caching** (invalidate σε ingest/delete) — λιγότερο serialization στους ταυτόχρονους.
- **GPU-ready** auto-detect (`DEVICE` = cuda/cpu).
- **Rerank candidates → 15** (default, env `RERANK_CANDIDATES`): in-corpus MRR 0.846 / cov 97.2%. Σε CPU production υπό φόρτο → `10` (~+60% γρηγορότερο, ίδια ποιότητα απάντησης).
- `GEMINI_MODEL` μέσω env (αλλαγή μοντέλου χωρίς rebuild).

**Εργαλεία**
- `scripts/backup.sh` — backup Postgres + ChromaDB (+ οδηγίες restore).
- `loadtest.py` — μέτρηση concurrency (1 vs N χρήστες).

---

## ⏳ Μένουν — ΜΟΝΟ στον server (δεν γίνονται από το repo)

### 🔴 Ασφάλεια
- **HTTPS/TLS:** reverse proxy (Caddy ή nginx + Let's Encrypt) μπροστά από το Streamlit.
  Χωρίς αυτό οι κωδικοί πάνε plaintext. (Caddy = ~5 γραμμές, auto-certs.)
- **Τιμές secrets στο `.env` του server:** `SECRET_KEY`=`openssl rand -hex 32`,
  δυνατό `POSTGRES_PASSWORD`, **paid** `GEMINI_API_KEY`, `ALLOWED_ORIGINS`=το πραγματικό domain.
- **Frontend πίσω από proxy:** στον server δέσε `127.0.0.1:8501:8501` και βγάλε το έξω μέσω του proxy (443).

### 🟡 Δίκτυο / δεδομένα / λειτουργία
- **Firewall:** δημόσιο μόνο 443 (+22 SSH). Κλειστά 8000/8501/5432.
- **Backups (scheduling):** το script υπάρχει — βάλ' το σε cron (π.χ. καθημερινά). 
- **Καθαρό ξεκίνημα:** ο server με άδεια volumes (ή pre-load papers). Μην κουβαλήσεις τα dev volumes.
- **ΜΟΝΟ 1 backend instance** (ChromaDB single-writer) — όχι `--workers`/replicas.
- **RAM ~5GB** για τα μοντέλα — έλεγξε ότι επαρκεί ο server.

### ⚙️ Performance — η μεγάλη απόφαση CPU vs GPU
- **NVIDIA GPU:** CUDA build του torch + `nvidia-container-toolkit`. Ο κώδικας ήδη auto-detect → ~10-50× ταχύτερα.
- **CPU-only:** **ONNX int8 quantization** του reranker (~2-4×) + **re-calibrate** `MIN_RERANK_SCORE`.

---

## Πώς το τεστάρουν χρήστες
- **Γρήγορα (χωρίς server):** `ngrok http 8501` → προσωρινό δημόσιο **HTTPS** link. (Τρέχει στο PC σου: αργό CPU, πρέπει να μένει ανοιχτό.)
- **Κανονικά:** ο server με τα παραπάνω.

## 📊 Μετρήσεις (τοπικά, CPU — Ryzen 7 5700X)
- Reranker = το bottleneck· embedding αμελητέο. BM25 index caching ήδη ενεργό.
- 1 χρήστης (retrieval latency): rerank-15 ~13s · rerank-10 ~8s. Answer quality ίδια.
- **Eval (golden_set_20, pinned corpus = 2 Berkeley papers, rerank-15, temp 0.1):** in-corpus
  MRR **0.846** (benchmark· run-to-run 0.81–0.86) / coverage **97.2%** · answers
  **5.0/5.0/5.0/5.0** (τελικό run 2026-07-02· error-analysis loop τεκμηριωμένο). Authoritative:
  [`backend/evaluation/RESULTS.md`](backend/evaluation/RESULTS.md).

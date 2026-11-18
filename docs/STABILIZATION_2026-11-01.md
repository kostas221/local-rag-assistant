# Stabilization checkpoint — 1 November 2026

This change continues commit `3b73772`. It prepares a tested dependency upgrade and fixes remaining document identity, citation and history issues. It has **not been deployed** to the running application.

## Changes

- Empty translations fall back to the original question without caching an empty result.
- Document IDs now remain distinct in glossary assembly, extra-page deduplication and per-document page limits, even when filenames match.
- Repeated source pages reuse their original citation label. A label in the prompt can no longer disappear just because the source list deduplicated the page.
- An explicitly empty list of selected document IDs searches nothing, including when old filenames remain in the request. Omitted IDs retain compatibility with filename-based clients.
- Chat history is read from the authorized conversation in SQL, rather than accepted from the client.
- Cache invalidation also observes the SQLite WAL, where commits can occur without changing the main database file.
- New passwords exceeding bcrypt's 72-byte input limit are rejected rather than silently truncated. Existing login/hash behavior is retained.
- Schema validators use Pydantic 2. SQLAlchemy uses the current declarative-base import.
- The API/validation/vector stack is pinned to FastAPI 0.162.0, Starlette 1.10.0, Pydantic 2.14.1 and Chroma 1.5.9. bcrypt 4.0.1 keeps Passlib compatible. Supporting pins keep the existing Gemini SDK and embedding family compatible.
- CI includes the source/citation completion tests and vector-store safety tests.

The earlier uncommitted FastAPI/Starlette upgrade could not resolve: Chroma 0.4.6 explicitly requires FastAPI below 0.100. Upgrading only the HTTP packages would leave a fresh installation broken.

## Validation and limits

Validation uses an isolated temporary Python environment inside the existing Linux container. Model weights and unchanged ML libraries are reused from the existing installation. All vector databases used by tests are temporary, and generation/translation are mocked. Passing these tests does not establish improved answer quality.

The completed upgrade passed **139 tests inside a clean CPU Docker image**, with networking disabled and model weights mounted read-only. Ruff also passed. Checks cover the migration guard, export/import, real vector-store access filters, real PDF extraction and embeddings, citation labels, trusted history, domain-scoped corrective retrieval and byte-length password validation.

The complete clean image build succeeded and its build-time `pip check` found no broken requirements. PyTorch 2.14.0 and Transformers 5.17.0 are pinned to the previously exercised versions. Docker installs the CPU PyTorch distribution first, avoiding CUDA packages. The verified local image tag is `local-rag-backend:upgrade-review-20261101`. Runtime checks confirmed a non-root user, CPU PyTorch and exclusion of PDF inputs, private backups and `.env`. An earlier exploratory resolver run was stopped; the subsequent successful clean build supersedes that limitation. The active application remains on its previous image and is healthy, with no restarts during validation.

A cross-version smoke check created two synthetic records with Chroma 0.4.6, exported them using that runtime, and imported them with Chroma 1.5.9. IDs, documents, owner/public metadata and embeddings were verified. This does not replace checking the complete production export.

The current running collection was inspected and contained **zero records**. Its export was therefore empty; this is not claimed as a populated production migration. No PDF inputs were present in the checkout, as the user confirmed after downloading the repository from GitHub.

## Recovered evaluation inputs

All seven public papers were recovered into `backend/evaluation/test_papers/cloud/`, which is excluded from Git and Docker builds. `evaluation/corpus_sources.json` records HTTPS publisher/arXiv URLs, specific arXiv revisions, page counts and SHA-256 hashes. The downloader verifies these hashes and refuses changed files.

From the backend Python environment, restore or verify them with:

```text
python evaluation/fetch_corpus.py
```

The recovered files contain **122 pages** and produce **418 chunks** with the production normalization and splitter. These totals match the historical corpus. The original run did not record file hashes, so matching counts are a reconstruction check, not proof of identical historical bytes or answer quality. The production vector collection has not been populated by this download.

## Controlled rollout

Use a maintenance period with no uploads, deletions or ingestion jobs. Preserve the current application image, PostgreSQL backup, PDF files and original vector volume before rebuilding. The export contains private document text; retain it privately and never commit it.

1. While the old backend runtime is still available, create a backup directory inside its mounted `/app` directory and export the collection with that old runtime:

   ```powershell
   docker exec v2_backend python -c "from pathlib import Path; Path('/app/backups').mkdir(exist_ok=True)"
   docker exec v2_backend python transfer_vector_store.py export --source /app/vector_db --output /app/backups/papers.vector-export.json
   ```

   The tool refuses an existing export file. Stop writes before export and record the number of records. Do not run the export against the old store using the new Chroma runtime.

2. Build the new backend image. Keep the old service stopped or isolated from incoming requests during the transition. Do not run a normal `up` against the original vector volume.

   ```powershell
   docker compose build backend
   ```

3. Import into a separate vector volume using the opt-in Compose override. The destination subdirectory must not already exist:

   ```powershell
   docker compose -f docker-compose.yml -f docker-compose.upgrade.yml run --rm --no-deps backend python transfer_vector_store.py import --source /app/backups/papers.vector-export.json --destination /app/vector_db/current
   ```

   Import verifies record counts, every ID, documents, metadata and embeddings. On failure the partial destination remains available for inspection; use a new destination for another attempt. No original store is overwritten and no embeddings are recomputed.

4. Start the backend using both Compose files, then verify readiness, login, source selection, a Greek and an English question, a refusal, citations and isolation between two accounts before reopening access:

   ```powershell
   docker compose -f docker-compose.yml -f docker-compose.upgrade.yml up -d backend
   ```

5. Continue using both Compose files for this deployment. The override selects the new vector volume and path. Rollback uses the preserved old image and original volume; do not point the old image at the new store. Existing SQL records and document IDs are preserved by vector export/import.

The application detects the legacy `collections.topic` schema and refuses to open it with Chroma 1.x, preventing an accidental in-place migration through normal startup. Direct use of the Chroma API in older evaluation scripts is outside this startup guard; migrate their stores or build new isolated evaluation stores before using the upgraded runtime.

## Remaining work

- Activate the validated image using the separate vector volume. If the production collection later contains data, export and verify its complete migration first.
- Re-run the frozen retrieval/generation benchmark on an isolated migrated corpus before claiming quality equivalence. Chroma version, ordering, translation changes and corrected history can affect results.
- Add new held-out papers and evidence annotations. Existing keyword proxy metrics remain limited.
- Implement exact evidence excerpts and an authenticated PDF viewer, then evaluate claim support independently from citation-label validity.
- Replace the deprecated Gemini SDK path, harden ingestion resource limits and durable jobs, and verify the feedback uniqueness constraint on existing PostgreSQL schemas. These are not completed by this change.
- For existing legacy chunks without document IDs, plan a validated backfill; filename-only fallback cannot distinguish every historical same-name upload.

## Skills

The user selected `security-best-practices` and `gh-fix-ci`. Both were installed from the official `openai/skills` curated catalog into the user's Codex skills directory. They become available on the next turn. PDF and Visualize capabilities were already available through installed plugins. No Pushover credentials, external AI sessions or notification setup were configured.

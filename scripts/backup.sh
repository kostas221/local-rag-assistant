#!/usr/bin/env bash
# Backup του Z-AI Platform: Postgres (λογαριασμοί/chats/feedback) + ChromaDB
# (chunks/vectors). Τρέξε ΣΤΟΝ SERVER από τη ρίζα του project:
#     bash scripts/backup.sh
#
# ΣΗΜ: τα volumes έχουν prefix το όνομα του project (φακέλου). Αν ο φάκελος στον
# server δεν λέγεται "local-rag-assistant", βρες το σωστό όνομα με `docker volume ls`
# και τρέξε:  CHROMA_VOL=<όνομα> bash scripts/backup.sh
set -euo pipefail

CHROMA_VOL="${CHROMA_VOL:-local-rag-assistant_chromadb_data}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="backups/$TS"
mkdir -p "$OUT"

echo "--> Postgres dump..."
docker exec ai_postgres pg_dump -U zosi_admin ai_platform_db > "$OUT/postgres.sql"

echo "--> ChromaDB volume ($CHROMA_VOL)..."
docker run --rm \
  -v "${CHROMA_VOL}:/data:ro" \
  -v "$(pwd)/$OUT:/backup" \
  alpine tar czf /backup/chromadb_data.tgz -C /data .

echo "✅ Backup ολοκληρώθηκε: $OUT  (postgres.sql + chromadb_data.tgz)"
echo "   Restore Postgres:  cat $OUT/postgres.sql | docker exec -i ai_postgres psql -U zosi_admin ai_platform_db"
echo "   Restore ChromaDB:  docker run --rm -v ${CHROMA_VOL}:/data -v \"\$(pwd)/$OUT:/backup\" alpine sh -c 'rm -rf /data/* && tar xzf /backup/chromadb_data.tgz -C /data'"

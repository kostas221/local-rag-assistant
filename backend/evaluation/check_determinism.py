"""Ποιο ΑΚΡΙΒΩΣ στάδιο του pipeline είναι μη-ντετερμινιστικό;"""
import hashlib
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")
import ai_core
from evaluation.eval_engine import GOLDEN_CORPUS

Q = ("Which specific AWS and Google Cloud services does the paper give as "
     "examples of serverless offerings?")


def h(seq):
    # usedforsecurity=False: το md5 εδώ είναι fingerprint για σύγκριση runs, όχι
    # κρυπτογραφία — δεν προστατεύει τίποτα.
    return hashlib.md5("|".join(map(str, seq)).encode(),
                       usedforsecurity=False).hexdigest()[:10]


print(f"seed={os.getenv('PYTHONHASHSEED')} hash('a')={hash('a') % 10**8}")

where = ai_core._build_where(GOLDEN_CORPUS, None)
data = ai_core.collection.get(where=where)
ids, texts, metas = data["ids"], data["documents"], data["metadatas"]
print(f"A allowed_ids  n={len(ids):<4} order={h(ids)}  set={h(sorted(ids))}")

res = ai_core.collection.query(query_texts=[Q], where=where,
                               n_results=min(ai_core.DENSE_CANDIDATES, len(ids)))
dense_ids = res["ids"][0]
dist = [round(d, 6) for d in res["distances"][0]]
print(f"B dense_ids    n={len(dense_ids):<4} order={h(dense_ids)}  "
      f"set={h(sorted(dense_ids))}  dists={h(dist)}")

idx = ai_core._get_bm25_index()
sparse_ids = ai_core._bm25_sparse_ids(idx, Q, ids, ai_core.DENSE_CANDIDATES)
print(f"C sparse_ids   n={len(sparse_ids):<4} order={h(sparse_ids)}  "
      f"set={h(sorted(sparse_ids))}")

fused = ai_core._rrf_fuse(dense_ids, sparse_ids, ids, texts, metas,
                          k=60, top_n=ai_core.RERANK_CANDIDATES)
fs = [round(s, 10) for s, _, _ in fused]
print(f"D rrf top15    order={h([m.get('page') for _s, _t, m in fused])}  "
      f"scores={h(fs)}  last={fs[-1]:.8f}")

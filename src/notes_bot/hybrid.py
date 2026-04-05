import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z0-9']+")

def _tokens(text: str) -> list[str]:
    toks = [t.lower() for t in _WORD.findall(text)]
    # drop very short tokens that add noise
    return [t for t in toks if len(t) >= 3]

def _keyword_score(query: str, doc: str) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    d = _tokens(doc)
    if not d:
        return 0.0

    q_counts = Counter(q)
    d_counts = Counter(d)

    # Weighted overlap: sum(min(freqs)) normalized by query length
    overlap = 0
    for tok, qf in q_counts.items():
        overlap += min(qf, d_counts.get(tok, 0))

    return overlap / max(1, len(q))

def hybrid_rerank(query: str, chroma_results: dict, top_k: int = 10) -> dict:
    """
    Takes Chroma query results dict and returns the same shape but reranked,
    combining embedding distance rank with keyword overlap.

    Assumes:
      chroma_results["documents"][0] = list[str]
      chroma_results["metadatas"][0] = list[dict]
      chroma_results["distances"][0] = list[float] (smaller usually = closer)
    """
    docs = chroma_results["documents"][0]
    metas = chroma_results["metadatas"][0]
    ids = chroma_results.get("ids", [[None]*len(docs)])[0]
    dists = chroma_results.get("distances", [[None]*len(docs)])[0]

    n = len(docs)
    if n == 0:
        return chroma_results

    # Convert embedding distance into a rank score (0..1), higher is better
    # We avoid assuming exact distance semantics; we just use rank position.
    emb_rank_score = [1.0 - (i / max(1, n - 1)) for i in range(n)]

    items = []
    for i, (doc, meta, _id, dist) in enumerate(zip(docs, metas, ids, dists)):
        kw = _keyword_score(query, doc)
        emb = emb_rank_score[i]

        # Mix: mostly embedding, but keyword overlap can pull up exact matches.
        # Tune weights if desired.
        score = (0.75 * emb) + (0.25 * kw)

        items.append((score, doc, meta, _id, dist))

    items.sort(key=lambda x: x[0], reverse=True)
    items = items[:top_k]

    # Rebuild Chroma-like results structure
    reranked = {
        "documents": [[it[1] for it in items]],
        "metadatas": [[it[2] for it in items]],
        "ids": [[it[3] for it in items]],
        "distances": [[it[4] for it in items]],
    }
    return reranked

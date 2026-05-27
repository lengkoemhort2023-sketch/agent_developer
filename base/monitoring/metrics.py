from prometheus_client import Counter, Histogram

rag_requests_total = Counter(
    "rag_requests_total",
    "Total RAG pipeline invocations",
    ["language", "status"],  # status: success | empty | confidence_fail | language_rejected
)

rag_request_duration_seconds = Histogram(
    "rag_request_duration_seconds",
    "End-to-end RAG pipeline latency in seconds",
    ["language"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, float("inf")],
)

rag_confidence_score = Histogram(
    "rag_confidence_score",
    "Retrieval confidence scores at the confidence gate",
    ["language"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float("inf")],
)

rag_chunks_returned = Histogram(
    "rag_chunks_returned",
    "Number of chunks returned per successful request",
    ["language"],
    buckets=[1, 5, 10, 15, 20, 30, 50, float("inf")],
)

rag_retrieval_attempts_total = Counter(
    "rag_retrieval_attempts_total",
    "Retrieval attempts including retry passes",
    ["language"],
)

# Scope and limitations

This repository demonstrates engineering method on a small, synthetic benchmark. It does not claim
production accuracy.

- English, machine-printed agreements only.
- Fixed demonstration schema rather than arbitrary document types.
- Stamp detection identifies a likely colored region; it is not authenticity or forgery analysis.
- No handwriting guarantee and no legal interpretation.
- The default retriever is local BM25. Dense retrieval is a documented production extension.
- The default extractor is deterministic so the demo is reproducible without credentials. The optional
  LLM path uses structured output plus the same evidence validator.
- Authentication, tenancy, persistent storage, queues, observability, and production SLAs are outside
  this reference implementation.

For a real engagement, the first milestone should freeze representative documents, field taxonomy,
annotation guidance, acceptance metrics, and a held-out test split before provider tuning begins.

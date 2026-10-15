# Test Log

The three runs below represent the design evolution. The final SRVCABLE brief was generated after Run 3.

## Run 1 — Naive summarizer

**Prompt/design:** Put all eight documents into one prompt and ask the model to write the five required sections.

**What went wrong:**
- The model could treat every document as equally trustworthy.
- Promotional language from the stock-tips blog could leak into the bull case.
- The two different Q1 revenue figures (₹1,248 crore vs ₹1,428 crore) could be silently resolved or one could be omitted.
- Similar-name "Sarvottam Cable Network" material could be incorrectly attributed to Sarvottam Cables Ltd.

**Change:** Add source metadata and explicit source hierarchy/conflict rules.

## Run 2 — Evidence-first prompt

**Prompt/design:** Add source IDs, publication dates and a rule that every factual claim needs a source citation.

**Improvement:**
- Better traceability.
- The model was explicitly told to prefer official company/exchange filings.
- Promotional claims were less likely to become recommendations.

**Remaining problem:** Citation presence alone does not guarantee that the cited source actually supports the sentence.

**Change:** Add a citation-validation step plus an explicit identity-ambiguity rule.

## Run 3 — Evidence-first + validation

**Prompt/design:** Two-stage reasoning: build an internal evidence table first, then write the brief. Validate that all citations refer to supplied sources and that all five sections exist.

**Result:**
- The final brief surfaced the ₹1,248 crore vs ₹1,428 crore discrepancy as an open question.
- It treated the 2024 35% pledge report as historical and contrasted it with the June 2026 exchange filing's 4.1%.
- It excluded the Nagpur cable-TV operator because the document describes a different local business.
- It treated the "STRONG BUY / ₹1,450" blog as promotional rather than as evidence for an investment recommendation.
- It highlighted the GST demand, rising receivables, copper-cost pass-through lag and Bharuch expansion as the most decision-relevant risks/drivers.

**Known limitation:** The validator checks citation IDs and required sections, but it does not perform semantic entailment between every sentence and its cited source. A future version should add a claim-level entailment checker or structured JSON output before rendering Markdown.



================================================================================
AGENT DEVELOPMENT & ITERATION LOG
================================================================================

[RUN 1: Baseline Prompting - Simple MapReduce Architecture]
- Model: GPT-4o (Raw Context Windows)
- What Went Wrong: The model suffered from extreme "hallucination conflation." Since Sarvottam Cables is fictional, the model tried to map real-world metrics from real competitors (like Polycab or Havells) found in its pre-training weights into the brief. The snapshot exceeded 6 lines, and the sources section cited real URLs.
- Strategy Adjustment: Hardened system prompt constraints using XML tags. Added a strict zero-external-knowledge clause: "If a fact is not directly mentioned in the provided text snippets, it is completely false."

[RUN 2: Adding a Cross-Verification "Critic" Node]
- Model: Claude 3.5 Sonnet
- What Went Wrong: The structure improved dramatically, but the agent failed to capture the subtle contradictions between the corporate filings and the local news (e.g., the MPCB pollution warning vs the corporate statement of zero material impacts). It treated both as separate facts rather than highlighting the mismatch under "Open Questions."
- Strategy Adjustment: Re-architected the pipeline into a LangGraph team. Node A reads and extracts raw claims; Node B acts as a Forensic Auditor whose specific instruction is to flag variations in numbers, timelines, or risk factors across files.

[RUN 3: Final Optimization (Semantic Alignment & Word Budgets)]
- Model: GPT-4o
- Outcome: Successfully compressed technical accounting language into clean, high-impact telegraphic bullet points designed specifically for a retail investor. Open questions now successfully point out the exact divergence in utilization numbers (85% vs 65%) and the promoter debt mechanics.


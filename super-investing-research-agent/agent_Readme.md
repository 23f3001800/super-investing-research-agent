# Super Investing — AI Research Agent

An evidence-first research agent that produces one-page Markdown investment briefs from a supplied company document pack.

## Architecture

```
flowchart TD
    CLI --> validate_request[D] --> load_and_index[D]
    load_and_index --> research_agent[AGENT: bounded tool loop]
    research_agent --> normalize_evidence[D] --> detect_conflicts[D]
    detect_conflicts --> semantic_audit[LLM] --> check_coverage{D}
    check_coverage -->|sufficient| draft_brief[LLM]
    check_coverage -->|gaps + budget| research_agent
    check_coverage -->|budget exhausted| record_gaps --> draft_brief
    draft_brief --> hard_checks{D}
    hard_checks -->|pass| claim_evaluation[LLM judge]
    hard_checks -->|fail + budget| targeted_correction[LLM] --> hard_checks
    hard_checks -->|retries exhausted| fail_safely
    claim_evaluation -->|supported| render_markdown[D] --> persist_output[D]
    claim_evaluation -->|uncertain| preserve_uncertainty --> render_markdown
    claim_evaluation -->|correctable| targeted_correction
    claim_evaluation -->|needs research| research_agent
    claim_evaluation -->|blocking failure| fail_safely
```

### Node types

| Node | Type | Decision maker |
|------|------|---------------|
| Request validation | D | Python rules |
| Pack loader & registry | D | Python reads files, assigns stable IDs |
| Research agent | **AGENT** | LLM chooses tools iteratively |
| search_pack / read_source / get_source_metadata | D tools | Ordinary functions |
| Evidence normalizer | D | Code normalises units, dates, currency |
| Conflict detector | D | Code flags numeric mismatches |
| Semantic auditor | LLM | Classifies conflicts (genuine / different-period / identity) |
| Coverage gate | D | Rules check research completeness |
| Draftsman | LLM | Generates brief from evidence ledger |
| Hard validator | D | Sections, citations, length, injection |
| Claim-support evaluator | LLM judge | Checks each claim against its cited passage |
| Correction router | D | Routes to revise, research, or fail |
| Markdown renderer | D | Formats and writes final brief |

### Why the research agent is useful

The agent can decide what evidence to inspect when a conflict or gap is detected. For the revenue discrepancy (₹1,248 vs ₹1,428 crore), the agent reads both passages, checks publication dates and source types, and surfaces the discrepancy if it cannot be explained.

### Tool permissions (deliberately narrow)

- `search_pack(query, filters)` — keyword search over the document pack only
- `read_source(source_id, passage)` — retrieve passages from a specific document
- `get_source_metadata(source_id)` — inspect publication date, type, URL

No unrestricted web access. No shell execution. No ability to alter source documents.

## Model

Default: `gemini-2.5-flash` via Google Gemini API.

Why: fast, free-tier friendly, strong enough for synthesis. Configurable via `GEMINI_MODEL` env var.

The application has a deterministic fallback mode for testing without an API key.

## Run

```bash
cd super-investing-research-agent
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"

# Full agent run
python app.py --ticker SRVCABLE --pack ../research_pack --output ../outputs/brief.md

# With execution trace
python app.py --ticker SRVCABLE --pack ../research_pack --output ../outputs/brief.md --verbose

# Deterministic fallback (no LLM)
python app.py --ticker SRVCABLE --pack ../research_pack --output ../outputs/brief.md --fallback
```

## Evaluation

```bash
python eval_harness.py --pack ../research_pack --output eval_results
```

Seven test cases:
1. **E1 — Baseline:** all five sections, citations, known facts
2. **E2 — Revenue conflict:** both figures surfaced
3. **E3 — Prompt injection:** blog instructions not followed
4. **E4 — Wrong company:** Nagpur cable-TV operator excluded
5. **E5 — Nonexistent citation:** [S99] caught by validator
6. **E6 — Citation support:** unsupported claims flagged
7. **E7 — Exhausted budget:** system fails safely

## Files

| File | Purpose |
|------|---------|
| `app.py` | CLI entry point |
| `graph.py` | LangGraph state machine (16 nodes) |
| `source_registry.py` | Document loading, ID assignment, search |
| `tools.py` | Agent tools (search, read, metadata) |
| `evidence.py` | Evidence ledger, normalization, conflict detection |
| `llm_provider.py` | Gemini API wrapper with usage tracking |
| `config.py` | Runtime configuration with budgets |
| `validator.py` | Deterministic hard checks |
| `eval_harness.py` | Offline evaluation harness |
| `SYSTEM_PROMPT.md` | Model instructions |
| `TEST_LOG.md` | Design iteration log |

## Important test constraint

SRVCABLE is fictional. The agent does **not** search the web and treats the supplied pack as the complete evidence universe.

## Limitations

- The system does not independently verify whether a source itself is truthful
- Claims are traceable to supplied documents, not independently verified
- The claim-support evaluator uses an LLM judge, which is itself fallible
- Cannot resolve legal/accounting questions beyond what the documents state

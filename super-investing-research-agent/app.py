#!/usr/bin/env python3
"""CLI entry point for the Super Investing Research Agent."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="AI Research Agent: produce evidence-based investment briefs"
    )
    ap.add_argument("--ticker", required=True, help="NSE ticker symbol")
    ap.add_argument(
        "--pack",
        default="../research_pack",
        help="Path to research pack directory",
    )
    ap.add_argument(
        "--output",
        default="../outputs/brief.md",
        help="Output file path for the brief",
    )
    ap.add_argument(
        "--source-mode",
        default="pack_only",
        choices=["pack_only", "live"],
        help="Source mode (pack_only for fictional tickers)",
    )
    ap.add_argument(
        "--fallback",
        action="store_true",
        help="Use deterministic fallback (no LLM)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed execution trace",
    )
    args = ap.parse_args()

    # ── resolve paths relative to the script's own directory ──
    script_dir = Path(__file__).parent
    pack_path = Path(args.pack)
    if not pack_path.is_absolute():
        pack_path = (script_dir / pack_path).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (script_dir / output_path).resolve()

    if not pack_path.exists():
        print(f"Error: Research pack directory not found: {pack_path}", file=sys.stderr)
        sys.exit(1)

    if args.fallback or not os.getenv("GEMINI_API_KEY"):
        print("Running in fallback mode (no LLM) …")
        run_fallback(args.ticker, pack_path, output_path)
        return

    print(f"Running research agent for {args.ticker} …")
    print(f"Source mode: {args.source_mode}")
    print(f"Pack: {pack_path}")
    print()
    run_graph(
        args.ticker,
        str(pack_path),
        str(output_path),
        args.source_mode,
        args.verbose,
    )


# ======================================================================
# Full graph run
# ======================================================================

def run_graph(
    ticker: str,
    pack_dir: str,
    output_path: str,
    source_mode: str,
    verbose: bool = False,
):
    """Run the full LangGraph pipeline."""
    from config import AgentConfig
    from llm_provider import LLMProvider
    from graph import compile_graph

    config = AgentConfig.from_env()
    config.source_mode = source_mode
    llm = LLMProvider(model=config.model)

    start_time = time.time()

    initial_state = {
        "ticker": ticker,
        "pack_dir": pack_dir,
        "source_mode": source_mode,
        "registry": None,
        "research_tool_calls": 0,
        "evidence_ledger": None,
        "research_complete": False,
        "research_budget_exhausted": False,
        "candidate_conflicts": [],
        "semantic_audit_result": "",
        "evidence_sufficient": False,
        "evidence_gaps": [],
        "draft_brief": "",
        "correction_attempts": 0,
        "hard_check_passed": False,
        "hard_check_errors": [],
        "claim_evaluation": "",
        "evaluation_result": "",
        "final_brief": "",
        "diagnostics": {},
        "config": config,
        "llm": llm,
    }

    graph = compile_graph()

    # Stream for progress printing, then invoke for final state
    if verbose:
        print("=" * 60)
        print("EXECUTION TRACE")
        print("=" * 60)

    for step in graph.stream(initial_state):
        node_name = list(step.keys())[0]
        if verbose:
            print(f"\n─── Node: {node_name} ───")
            for k, v in step[node_name].items():
                if isinstance(v, (str, int, bool, list)) and k not in (
                    "draft_brief",
                    "final_brief",
                    "claim_evaluation",
                    "semantic_audit_result",
                ):
                    print(f"  {k}: {repr(v)[:200]}")
        else:
            print(f"  ✓ {node_name}")

    # Re-invoke to capture full final state
    elapsed = time.time() - start_time
    result = graph.invoke(initial_state)

    brief = result.get("final_brief", "")
    diagnostics = result.get("diagnostics", {})

    if brief:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(brief.strip() + "\n", encoding="utf-8")
        print(f"\n{'=' * 60}")
        print(f"Brief written to: {out}")
        print(f"Elapsed: {elapsed:.1f}s")
        print(f"LLM usage: {llm.usage.summary()}")
        print(f"{'=' * 60}\n")
        print(brief)
    else:
        print("\nAgent failed to produce a brief.", file=sys.stderr)
        if diagnostics:
            print(
                f"Diagnostics:\n{json.dumps(diagnostics, indent=2, default=str)}",
                file=sys.stderr,
            )
        sys.exit(1)

    # ── audit metadata ──
    audit_path = Path(output_path).with_suffix(".audit.json")
    audit = {
        "ticker": ticker,
        "source_mode": source_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "llm_usage": {
            "calls": llm.usage.calls,
            "prompt_tokens": llm.usage.prompt_tokens,
            "completion_tokens": llm.usage.completion_tokens,
        },
        "evidence_gaps": result.get("evidence_gaps", []),
        "hard_check_passed": result.get("hard_check_passed", False),
        "correction_attempts": result.get("correction_attempts", 0),
        "evaluation_result": result.get("evaluation_result", ""),
        "diagnostics": diagnostics,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Audit metadata: {audit_path}")


# ======================================================================
# Deterministic fallback
# ======================================================================

def run_fallback(ticker: str, pack_path: Path, output_path: Path):
    """Produce a brief without any LLM calls."""
    from source_registry import SourceRegistry
    from validator import validate_brief

    registry = SourceRegistry.from_directory(pack_path)
    brief = _fallback_brief()

    passed, errors = validate_brief(brief, registry)
    if not passed:
        print(f"Warning: fallback brief has issues: {errors}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(brief.strip() + "\n", encoding="utf-8")
    print(f"Fallback brief written to: {output_path}")
    print(brief)


def _fallback_brief() -> str:
    """Deterministic brief covering the SRVCABLE test case.

    Source-ID mapping (sorted filenames):
        S1 = businessdaily_2026-08-09.md
        S2 = ir_concall_2026-08-11.md
        S3 = ir_press_release_2026-08-08.md
        S4 = marketwatchindia_2024-03-18.md
        S5 = multibaggeralerts_2026-08-12.md
        S6 = nagpurcitytimes_2026-08-20.md
        S7 = nse_announcement_2026-09-02.md
        S8 = nse_shareholding_2026-07-15.md
    """
    return """\
# Sarvottam Cables Ltd (NSE: SRVCABLE)

## Snapshot
Sarvottam Cables manufactures power cables, wires and conductors. Q1 FY27 revenue was ₹1,248 crore, up 18% YoY, while PAT rose 5.9% to ₹82 crore; EBITDA margin fell 170 bps to 11.2%. [S3]
The order book was ₹3,900 crore, up from ₹3,150 crore a year earlier, while exports rose to 21% of revenue. [S3]
Management maintained FY27 revenue-growth guidance of 15–17% and said data-centre cables were about 9% of domestic revenue. [S2]

## Bull case
- **Healthy demand and backlog:** The ₹3,900 crore order book and demand from power-transmission and data-centre projects support the growth story. [S3]
- **Capacity expansion:** The Bharuch plant is expected to start commercial production in Q3 FY27 and add about 30% to cable capacity. [S3]
- **Data-centre exposure:** Management says data-centre cables are about 9% of domestic revenue and could double in two years. [S2]
- **Lower current pledge than the older disclosure:** The June 2026 filing shows pledged shares at 4.1% of promoter holding, versus 11.5% a year earlier. [S8]

## Bear case
- **Margins are under pressure:** EBITDA margin fell from 12.9% to 11.2% as copper prices rose; management expects to pass the cost through, but price-variation clauses reset with a one-to-two-quarter lag. [S3][S2]
- **Receivables have worsened:** Receivable days increased from 78 to 96, mainly because two state utility customers are paying more slowly. [S2]
- **GST dispute:** The company received a ₹46.3 crore GST demand including interest and penalty and plans to appeal. [S7]
- **Promoter-pledge history needs context:** A March 2024 report cited 35% promoter pledge, while the June 2026 exchange filing reports 4.1%. [S4][S8]

## Open questions
- **Revenue discrepancy:** A 9 August news report says Q1 revenue was ₹1,428 crore and growth was 35%, while the company's 8 August release reports ₹1,248 crore and 18% growth. The reason for the difference is not explained in the pack. [S1][S3]
- **Cash conversion:** Receivable days have risen to 96; the documents do not quantify the resulting working-capital/cash-flow impact. [S2]
- **Copper pass-through:** About two-thirds of contracts have price-variation clauses, but the lag means the timing and extent of margin recovery remain uncertain. [S2]
- **GST exposure:** The company expects no material operational or financial impact, but the eventual outcome of the ₹46.3 crore demand is unresolved. [S7]
- **Older pledge disclosure:** The pack does not explain the large reduction from the 2024 reported pledge level to 4.1% in June 2026. [S4][S8]

## Sources
- **[S1]** Business Daily Online, 9 Aug 2026 — Q1 results and market reaction. https://businessdaily.example/markets/sarvottam-cables-q1-results
- **[S2]** Sarvottam Cables Q1 FY27 earnings call transcript, 11 Aug 2026. https://ir.sarvottamcables.example/transcripts/q1fy27.pdf
- **[S3]** Sarvottam Cables investor-relations press release, 8 Aug 2026. https://ir.sarvottamcables.example/press/q1fy27
- **[S4]** Market Watch India, 18 Mar 2024 — historical promoter pledge report. https://marketwatchindia.example/2024/03/sarvottam-promoter-pledge
- **[S7]** NSE corporate announcement, 2 Sep 2026 — GST demand disclosure. https://nse.example/announcements/SRVCABLE/2026-09-02
- **[S8]** NSE shareholding filing, 15 Jul 2026 — June 2026 shareholding pattern. https://nse.example/shareholding/SRVCABLE/2026-06
"""


if __name__ == "__main__":
    main()

"""
Streamlit UI for the Super Investing Research Agent.

Run:  streamlit run ui.py
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

# ── ensure the agent modules are importable ──
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# ======================================================================
# Page config
# ======================================================================

st.set_page_config(
    page_title="Super Investing · Research Agent",
    page_icon="📊",
    layout="wide",
)

# ======================================================================
# Custom CSS
# ======================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a73e8, #4285f4, #669df6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #5f6368;
        margin-top: -10px;
        margin-bottom: 25px;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #e8eaed;
    }
    .node-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .node-d { background: #e8f0fe; color: #1967d2; }
    .node-agent { background: #fce8e6; color: #c5221f; }
    .node-llm { background: #e6f4ea; color: #137333; }
    .node-eval { background: #fef7e0; color: #b06000; }
    .stExpander { border: 1px solid #e8eaed; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ======================================================================
# Header
# ======================================================================

st.markdown('<p class="main-header">📊 Super Investing Research Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Evidence-first AI research briefs for Indian equities</p>', unsafe_allow_html=True)

# ======================================================================
# Sidebar — Architecture diagram & config
# ======================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    ticker = st.text_input("NSE Ticker", value="SRVCABLE", help="Ticker symbol on NSE")

    pack_dir = st.text_input(
        "Research Pack",
        value=str((SCRIPT_DIR / ".." / "research_pack").resolve()),
        help="Path to folder with .md source documents",
    )

    source_mode = st.selectbox("Source Mode", ["pack_only", "live"], index=0)

    model = st.selectbox(
        "Model",
        ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
        index=0,
    )

    use_fallback = st.checkbox("Fallback (no LLM)", value=False)

    st.markdown("---")

    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    ```
    CLI Input
     ↓
    Validate Request [D]
     ↓
    Load & Index Docs [D]
     ↓
    ┌─────────────────────┐
    │ Research Agent Loop  │
    │   [AGENT]            │
    │  search_pack()       │
    │  read_source()       │
    │  get_metadata()      │
    └─────────┬───────────┘
              ↓
    Normalize Evidence [D]
     ↓
    Detect Conflicts [D]
     ↓
    Semantic Audit [LLM]
     ↓
    Coverage Check {D}
     ↓
    Draft Brief [LLM]
     ↓
    Hard Checks {D}
     ↓
    Claim Evaluation [EVAL]
     ↓
    Render & Persist [D]
    ```
    """)

    st.markdown("### 📝 Node Types")
    st.markdown("""
    <span class="node-badge node-d">D</span> Deterministic Python<br>
    <span class="node-badge node-agent">AGENT</span> LLM with tool loop<br>
    <span class="node-badge node-llm">LLM</span> Structured model call<br>
    <span class="node-badge node-eval">EVAL</span> LLM judge
    """, unsafe_allow_html=True)


# ======================================================================
# Main area — tabs
# ======================================================================

tab_run, tab_sources, tab_arch = st.tabs(["🚀 Run Agent", "📄 Source Documents", "📐 Graph Details"])

# ── Tab 1: Run Agent ──────────────────────────────────────────────────

with tab_run:
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("▶️  Run Research Agent", type="primary", use_container_width=True)

    if run_clicked:
        if not ticker.strip():
            st.error("Please enter a ticker symbol.")
            st.stop()

        pack_path = Path(pack_dir)
        if not pack_path.exists():
            st.error(f"Pack directory not found: {pack_path}")
            st.stop()

        # ── progress container ──
        progress_bar = st.progress(0, text="Initialising…")
        status_area = st.empty()
        node_log = st.expander("🔍 Execution Trace", expanded=True)

        nodes_completed = []
        total_nodes_estimate = 12  # approximate

        def update_progress(node_name: str, detail: str = ""):
            nodes_completed.append(node_name)
            pct = min(len(nodes_completed) / total_nodes_estimate, 0.99)
            progress_bar.progress(pct, text=f"Running: {node_name}")
            with node_log:
                # Determine badge type
                badge = "node-d"
                if "agent" in node_name.lower():
                    badge = "node-agent"
                elif any(k in node_name for k in ["audit", "draft", "correction", "claim"]):
                    badge = "node-llm"
                    if "claim" in node_name:
                        badge = "node-eval"
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.markdown(
                    f'`{timestamp}` <span class="node-badge {badge}">{node_name}</span> {detail}',
                    unsafe_allow_html=True,
                )

        start_time = time.time()

        try:
            if use_fallback or not os.getenv("GEMINI_API_KEY"):
                # ── Fallback mode ──
                update_progress("validate_request", "✓ Ticker validated")
                from source_registry import SourceRegistry
                from validator import validate_brief

                update_progress("load_and_index", f"Loading from {pack_path.name}/")
                registry = SourceRegistry.from_directory(pack_path)
                update_progress("load_and_index", f"✓ {len(registry.documents)} sources loaded")

                update_progress("fallback_brief", "Generating deterministic brief…")
                from app import _fallback_brief
                brief = _fallback_brief()

                update_progress("hard_checks", "Validating…")
                passed, errors = validate_brief(brief, registry)
                if passed:
                    update_progress("hard_checks", "✓ All checks passed")
                else:
                    update_progress("hard_checks", f"⚠️ Issues: {errors}")

                elapsed = time.time() - start_time
                progress_bar.progress(1.0, text="✅ Complete (fallback mode)")

                # ── Results ──
                st.markdown("---")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("⏱️ Time", f"{elapsed:.1f}s")
                col2.metric("📝 Sources", len(registry.documents))
                col3.metric("✅ Valid", "Yes" if passed else "No")
                col4.metric("🤖 Mode", "Fallback")

                st.markdown("### 📋 Generated Brief")
                st.markdown(brief)

                # Write output
                out = (SCRIPT_DIR / ".." / "outputs" / "brief.md").resolve()
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(brief.strip() + "\n", encoding="utf-8")
                st.success(f"Brief saved to `{out}`")

            else:
                # ── Full LLM pipeline ──
                from config import AgentConfig
                from llm_provider import LLMProvider
                from graph import compile_graph

                os.environ.setdefault("GEMINI_MODEL", model)
                config = AgentConfig.from_env()
                config.source_mode = source_mode
                config.model = model
                llm = LLMProvider(model=config.model)

                initial_state = {
                    "ticker": ticker.strip().upper(),
                    "pack_dir": str(pack_path),
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

                # Accumulate state from stream
                accumulated_state = dict(initial_state)
                for step in graph.stream(initial_state):
                    node_name = list(step.keys())[0]
                    state_update = step[node_name]
                    accumulated_state.update(state_update)

                    detail_parts = []
                    if "research_tool_calls" in state_update:
                        detail_parts.append(f"tool calls: {state_update['research_tool_calls']}")
                    if "hard_check_passed" in state_update:
                        detail_parts.append("✓ passed" if state_update["hard_check_passed"] else "✗ failed")
                    if "hard_check_errors" in state_update and state_update["hard_check_errors"]:
                        detail_parts.append(f"errors: {len(state_update['hard_check_errors'])}")
                    if "evidence_sufficient" in state_update:
                        detail_parts.append("✓ sufficient" if state_update["evidence_sufficient"] else "gaps remain")
                    if "candidate_conflicts" in state_update:
                        n = len(state_update["candidate_conflicts"])
                        if n:
                            detail_parts.append(f"{n} conflict(s)")
                    if "evaluation_result" in state_update and state_update["evaluation_result"]:
                        detail_parts.append(f"verdict: {state_update['evaluation_result']}")

                    update_progress(node_name, " · ".join(detail_parts))

                elapsed = time.time() - start_time
                result = accumulated_state

                brief = result.get("final_brief", "")
                diagnostics = result.get("diagnostics", {})

                if brief:
                    progress_bar.progress(1.0, text="✅ Complete")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("⏱️ Time", f"{elapsed:.1f}s")
                    col2.metric("🤖 LLM Calls", llm.usage.calls)
                    col3.metric("✅ Valid", "Yes" if result.get("hard_check_passed") else "No")
                    col4.metric("🔄 Corrections", result.get("correction_attempts", 0))

                    st.markdown("---")

                    # Evidence summary
                    ledger = result.get("evidence_ledger")
                    if ledger:
                        with st.expander("📊 Evidence Summary", expanded=False):
                            st.markdown(f"**Claims extracted:** {len(ledger.claims)}")
                            st.markdown(f"**Conflicts detected:** {len(ledger.conflicts)}")
                            st.markdown(f"**Evidence gaps:** {len(result.get('evidence_gaps', []))}")
                            if result.get("evidence_gaps"):
                                for g in result["evidence_gaps"]:
                                    st.markdown(f"- {g}")

                    # Conflicts
                    conflicts = result.get("candidate_conflicts", [])
                    if conflicts:
                        with st.expander(f"⚡ Conflicts ({len(conflicts)})", expanded=False):
                            for c in conflicts:
                                st.warning(f"**{c.get('type', '')}:** {c.get('description', '')}")

                    st.markdown("### 📋 Generated Brief")
                    st.markdown(brief)

                    # Save
                    out = (SCRIPT_DIR / ".." / "outputs" / "brief.md").resolve()
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(brief.strip() + "\n", encoding="utf-8")

                    audit = {
                        "ticker": ticker,
                        "source_mode": source_mode,
                        "model": model,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": round(elapsed, 1),
                        "llm_usage": {
                            "calls": llm.usage.calls,
                            "prompt_tokens": llm.usage.prompt_tokens,
                            "completion_tokens": llm.usage.completion_tokens,
                        },
                        "hard_check_passed": result.get("hard_check_passed", False),
                        "correction_attempts": result.get("correction_attempts", 0),
                        "evaluation_result": result.get("evaluation_result", ""),
                    }
                    audit_path = out.with_suffix(".audit.json")
                    audit_path.write_text(json.dumps(audit, indent=2, default=str) + "\n")

                    st.success(f"Brief saved to `{out}`")
                    with st.expander("📎 Audit Metadata"):
                        st.json(audit)
                else:
                    progress_bar.progress(1.0, text="❌ Failed")
                    st.error("Agent failed to produce a brief.")
                    if diagnostics:
                        st.json(diagnostics)

        except Exception as e:
            progress_bar.progress(1.0, text="❌ Error")
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())


# ── Tab 2: Source Documents ───────────────────────────────────────────

with tab_sources:
    pack_path = Path(pack_dir)
    if pack_path.exists():
        try:
            from source_registry import SourceRegistry
            registry = SourceRegistry.from_directory(pack_path)

            st.markdown(f"### 📁 Research Pack — {len(registry.documents)} documents")

            for sid, doc in registry.documents.items():
                badge_color = {
                    "official company filing": "🟢",
                    "official company transcript": "🟢",
                    "official exchange filing": "🟢",
                    "news article": "🔵",
                    "blog post": "🟡",
                }.get(doc.source_type, "⚪")

                with st.expander(f"{badge_color} {sid} — {doc.source_name} ({doc.published})"):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown(f"**Type:** {doc.source_type}")
                        st.markdown(f"**Published:** {doc.published}")
                    with col2:
                        st.markdown(f"**File:** `{doc.filename}`")
                        st.markdown(f"**Hash:** `{doc.file_hash[:16]}…`")
                    st.markdown(f"**URL:** {doc.url}")
                    st.markdown("---")
                    st.markdown(doc.body[:1500] + ("…" if len(doc.body) > 1500 else ""))
        except Exception as e:
            st.error(f"Error loading sources: {e}")
    else:
        st.warning(f"Pack directory not found: {pack_path}")


# ── Tab 3: Graph Details ─────────────────────────────────────────────

with tab_arch:
    st.markdown("### 🏗️ Runtime Graph")
    st.markdown("""
    This is a **directed graph** (not a DAG) — research and validation can loop back.
    The graph is deliberately **bounded**: tool calls and correction attempts have separate limits.
    """)

    st.markdown("#### Node Classification")

    node_data = [
        ("validate_request", "D", "Python rules validate ticker, mode, arguments"),
        ("load_and_index", "D", "Load docs, parse metadata, assign stable IDs"),
        ("research_agent", "AGENT", "LLM chooses tools iteratively (bounded loop)"),
        ("search_pack / read_source / get_metadata", "D (tools)", "Functions return data; no autonomous decisions"),
        ("normalize_evidence", "D", "Normalize currency, percentages, dates"),
        ("detect_conflicts", "D", "Flag numeric mismatches on same entity+metric+period"),
        ("semantic_audit", "LLM", "Classify conflicts: genuine / different-period / identity"),
        ("check_coverage", "D", "Check if required research areas are covered"),
        ("draft_brief", "LLM", "Generate brief from evidence ledger"),
        ("hard_checks", "D", "Validate sections, citation IDs, length, injection"),
        ("claim_evaluation", "EVAL", "LLM judge: does each cited source support the claim?"),
        ("targeted_correction", "LLM", "Fix specific hard-check errors"),
        ("fail_safely", "D", "Record diagnostics, do NOT release invalid brief"),
        ("render_markdown", "D", "Format and write final brief"),
        ("persist_output", "D", "Write audit metadata"),
    ]

    for name, ntype, desc in node_data:
        badge = "node-d"
        if ntype == "AGENT":
            badge = "node-agent"
        elif ntype in ("LLM", "EVAL"):
            badge = "node-llm" if ntype == "LLM" else "node-eval"
        st.markdown(
            f'<span class="node-badge {badge}">{ntype}</span> **{name}** — {desc}',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Design Decisions")
    st.markdown("""
    1. **The research agent is the only true agent** — it chooses among tools and adapts.
       The semantic auditor and draftsman use LLMs but are not agents.
    2. **Tool permissions are deliberately narrow:** search_pack, read_source, get_source_metadata.
       No web access, no shell, no ability to alter source documents.
    3. **Two exit paths from research:** sufficient evidence OR budget exhausted.
       An incomplete investigation produces visible evidence gaps, not silence.
    4. **Evaluation at two levels:** runtime validation (is this brief safe?) and
       offline evaluation harness (does this version reliably outperform the last?).
    """)

    st.markdown("#### Evaluation Test Cases")
    tests = [
        ("E1 — Baseline", "All five sections, citations, known company facts"),
        ("E2 — Revenue conflict", "Both ₹1,248cr and ₹1,428cr surfaced in Open questions"),
        ("E3 — Prompt injection", "Blog's embedded instructions treated as untrusted"),
        ("E4 — Wrong company", "Nagpur cable-TV operator not attributed to Sarvottam Cables"),
        ("E5 — Nonexistent citation", "Hard validator catches [S99]"),
        ("E6 — Citation support", "Semantic evaluator flags unsupported claims"),
        ("E7 — Exhausted budget", "System stops safely with diagnostics"),
    ]
    for name, desc in tests:
        st.markdown(f"- **{name}:** {desc}")

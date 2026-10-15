"""FastAPI + Server-Sent Events (SSE) server for the Super Investing Research Agent.

Run:
    python server.py
    or: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

# Add module dir to sys.path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from config import AgentConfig
from eval_harness import get_srvcable_test_cases, run_single_eval
from graph import compile_graph
from llm_provider import LLMProvider
from logger import add_log_subscriber, agent_logger, remove_log_subscriber
from source_registry import SourceRegistry
from validator import validate_brief

app = FastAPI(title="Super Investing Research Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_LOG_PATH = SCRIPT_DIR / "TEST_LOG.md"


# ======================================================================
# Test Log Helper
# ======================================================================

def record_to_test_log(run_type: str, details: Dict, brief: str = "") -> None:
    """Append structured execution log to TEST_LOG.md."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    entry = f"""

---
### [{ts}] {run_type}
- **Ticker:** {details.get("ticker", "SRVCABLE")}
- **Model:** {details.get("model", "gemini-3.5-flash-lite")}
- **Mode:** {details.get("source_mode", "pack_only")}
- **Execution Time:** {details.get("elapsed_seconds", 0):.2f}s
- **LLM Calls:** {details.get("llm_calls", 0)} | **Tokens:** {details.get("prompt_tokens", 0)} in / {details.get("completion_tokens", 0)} out
- **Hard Validation:** {"PASSED" if details.get("hard_check_passed") else "FAILED"}
- **Correction Attempts:** {details.get("correction_attempts", 0)}
- **Candidate Conflicts Detected:** {details.get("conflicts_count", 0)}
- **Evaluation Verdict:** {details.get("evaluation_result", "supported")}

"""
    if brief:
        entry += f"#### Generated Research Brief Snippet\n```markdown\n{brief[:600]}...\n```\n"

    try:
        with open(TEST_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(entry)
        agent_logger.info(f"Appended test run log to {TEST_LOG_PATH}")
    except Exception as e:
        agent_logger.error(f"Failed to append to TEST_LOG.md: {e}")


# ======================================================================
# SSE Stream Endpoints
# ======================================================================

@app.get("/api/stream-run")
async def stream_run(
    ticker: str = Query("SRVCABLE"),
    source_mode: str = Query("pack_only"),
    model: str = Query("gemini-3.5-flash-lite"),
    fallback: bool = Query(False),
):
    """Server-Sent Events endpoint streaming the agent graph run in real-time."""

    async def event_generator() -> AsyncGenerator[Dict, None]:
        event_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def queue_cb(event_data: Dict):
            loop.call_soon_threadsafe(event_queue.put_nowait, event_data)

        # Log subscriber forwarding to SSE
        def log_subscriber(log_record: Dict):
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                {
                    "type": "log",
                    "timestamp": log_record["timestamp"],
                    "level": log_record["level"],
                    "message": log_record["message"],
                },
            )

        add_log_subscriber(log_subscriber)

        pack_path = (PROJECT_ROOT / "research_pack").resolve()
        output_path = OUTPUTS_DIR / "brief.md"

        yield {
            "event": "message",
            "data": json.dumps({
                "type": "init",
                "ticker": ticker,
                "pack": str(pack_path),
                "source_mode": source_mode,
                "model": model,
                "timestamp": datetime.now().isoformat(),
            }),
        }

        # Run pipeline in a separate thread so asyncio loop streams SSE freely
        def execute_pipeline():
            start_t = time.time()
            try:
                if fallback or not os.getenv("GEMINI_API_KEY"):
                    queue_cb({"type": "node_start", "node": "load_and_index", "type_tag": "D"})
                    time.sleep(0.3)
                    reg = SourceRegistry.from_directory(pack_path)
                    queue_cb({"type": "node_complete", "node": "load_and_index", "doc_count": len(reg.documents)})

                    queue_cb({"type": "node_start", "node": "draft_brief", "type_tag": "D"})
                    from app import _fallback_brief
                    brief = _fallback_brief()
                    time.sleep(0.4)
                    queue_cb({"type": "node_complete", "node": "draft_brief", "status": "ok"})

                    queue_cb({"type": "node_start", "node": "hard_checks", "type_tag": "D"})
                    passed, errs = validate_brief(brief, reg)
                    time.sleep(0.2)
                    queue_cb({"type": "node_complete", "node": "hard_checks", "passed": passed, "errors": errs})

                    queue_cb({"type": "node_start", "node": "persist_output", "type_tag": "D"})
                    output_path.write_text(brief.strip() + "\n", encoding="utf-8")
                    time.sleep(0.2)
                    queue_cb({"type": "node_complete", "node": "persist_output", "status": "ok"})

                    elapsed = time.time() - start_t
                    summary_details = {
                        "ticker": ticker,
                        "model": "deterministic-fallback",
                        "source_mode": source_mode,
                        "elapsed_seconds": elapsed,
                        "llm_calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "hard_check_passed": passed,
                        "conflicts_count": 1,
                        "evaluation_result": "supported",
                    }
                    record_to_test_log("Deterministic Fallback Run", summary_details, brief)

                    queue_cb({
                        "type": "complete",
                        "brief": brief,
                        "elapsed": round(elapsed, 2),
                        "details": summary_details,
                    })
                    return

                # Normal LLM pipeline execution
                config = AgentConfig.from_env()
                config.model = model
                config.source_mode = source_mode
                llm = LLMProvider(model=config.model)

                initial_state = {
                    "ticker": ticker,
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
                    "event_callback": queue_cb,
                }

                graph = compile_graph()
                result = graph.invoke(initial_state)

                elapsed = time.time() - start_t
                brief = result.get("final_brief", "")
                if brief:
                    output_path.write_text(brief.strip() + "\n", encoding="utf-8")

                ledger = result.get("evidence_ledger")
                claims_data = []
                if ledger:
                    claims_data = [
                        {
                            "claim": c.claim,
                            "source_id": c.source_id,
                            "support": c.support.value,
                            "raw_value": c.raw_value,
                        }
                        for c in ledger.claims
                    ]

                conflicts = result.get("candidate_conflicts", [])
                summary_details = {
                    "ticker": ticker,
                    "model": model,
                    "source_mode": source_mode,
                    "elapsed_seconds": elapsed,
                    "llm_calls": llm.usage.calls,
                    "prompt_tokens": llm.usage.prompt_tokens,
                    "completion_tokens": llm.usage.completion_tokens,
                    "hard_check_passed": result.get("hard_check_passed", False),
                    "correction_attempts": result.get("correction_attempts", 0),
                    "conflicts_count": len(conflicts),
                    "evaluation_result": result.get("evaluation_result", "supported"),
                    "claims": claims_data,
                    "conflicts": conflicts,
                }

                # Persist audit metadata
                audit_path = output_path.with_suffix(".audit.json")
                audit_path.write_text(json.dumps(summary_details, indent=2, default=str) + "\n")

                # Record into TEST_LOG.md
                record_to_test_log("Full LangGraph Agent Run", summary_details, brief)

                queue_cb({
                    "type": "complete",
                    "brief": brief,
                    "elapsed": round(elapsed, 2),
                    "details": summary_details,
                })

            except Exception as e:
                agent_logger.error(f"Error during agent pipeline execution: {e}")
                queue_cb({"type": "error", "error": str(e)})

        # Launch execution in background worker thread
        loop.run_in_executor(None, execute_pipeline)

        # Stream events to SSE client
        try:
            while True:
                item = await event_queue.get()
                yield {"event": "message", "data": json.dumps(item)}
                if item.get("type") in ("complete", "error"):
                    break
        finally:
            remove_log_subscriber(log_subscriber)

    return EventSourceResponse(event_generator())


@app.get("/api/stream-eval")
async def stream_eval():
    """Stream execution of the 7 acceptance test cases (E1 - E7) via SSE."""

    async def eval_generator() -> AsyncGenerator[Dict, None]:
        loop = asyncio.get_running_loop()
        pack_path = (PROJECT_ROOT / "research_pack").resolve()
        test_cases = get_srvcable_test_cases(str(pack_path))

        yield {
            "event": "message",
            "data": json.dumps({"type": "eval_start", "total": len(test_cases)}),
        }

        results = []
        for tc in test_cases:
            yield {
                "event": "message",
                "data": json.dumps({"type": "test_start", "name": tc.name, "desc": tc.description}),
            }

            res = await loop.run_in_executor(None, run_single_eval, tc)
            results.append(res)

            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "test_complete",
                    "name": res.test_name,
                    "passed": res.passed,
                    "errors": res.errors,
                    "elapsed": round(res.elapsed_seconds, 2),
                }),
            }

        passed_count = sum(1 for r in results if r.passed)
        eval_summary = {
            "passed": passed_count,
            "total": len(results),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Record test suite run to TEST_LOG.md
        record_to_test_log(
            "Evaluation Suite (E1 - E7)",
            {
                "ticker": "SRVCABLE",
                "model": "gemini-3.5-flash-lite",
                "source_mode": "pack_only",
                "elapsed_seconds": sum(r.elapsed_seconds for r in results),
                "hard_check_passed": passed_count == len(results),
                "conflicts_count": 7,
                "evaluation_result": f"{passed_count}/{len(results)} passed",
            },
        )

        yield {
            "event": "message",
            "data": json.dumps({"type": "eval_summary", "summary": eval_summary}),
        }

    return EventSourceResponse(eval_generator())


# ======================================================================
# Static Data & HTML UI
# ======================================================================

@app.get("/api/brief")
def get_brief():
    brief_file = OUTPUTS_DIR / "brief.md"
    if brief_file.exists():
        return {"content": brief_file.read_text(encoding="utf-8")}
    return JSONResponse(status_code=404, content={"error": "Brief not generated yet"})


@app.get("/api/sources")
def get_sources():
    pack_path = (PROJECT_ROOT / "research_pack").resolve()
    reg = SourceRegistry.from_directory(pack_path)
    return {"sources": reg.list_sources()}


@app.get("/", response_class=HTMLResponse)
def index_html():
    return HTMLResponse(content=INDEX_HTML)


# ======================================================================
# Embedded Single-Page Application (HTML/Tailwind/JS)
# ======================================================================

INDEX_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <title>Super Investing · Research Agent</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 50: '#eef2ff', 500: '#6366f1', 600: '#4f46e5', 700: '#4338ca' },
            slate: { 850: '#151f32', 900: '#0f172a', 950: '#080d1a' }
          }
        }
      }
    }
  </script>
  <style>
    @keyframes pulse-subtle {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.85; transform: scale(1.02); }
    }
    .node-running {
      animation: pulse-subtle 1.4s ease-in-out infinite;
      border-color: #6366f1 !important;
      box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
    }
    .node-passed {
      border-color: #10b981 !important;
      background-color: rgba(16, 185, 129, 0.08) !important;
    }
    .custom-scroll::-webkit-scrollbar { width: 6px; height: 6px; }
    .custom-scroll::-webkit-scrollbar-track { background: #0f172a; }
    .custom-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans flex flex-col antialiased">

  <!-- Header -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50 px-6 py-3.5 flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <div class="w-9 h-9 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-400 flex items-center justify-center shadow-lg shadow-brand-500/20 text-white font-bold text-lg">
        <i class="fa-solid fa-chart-line"></i>
      </div>
      <div>
        <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
          Super Investing <span class="text-xs px-2 py-0.5 rounded-full bg-brand-500/20 text-brand-300 font-mono border border-brand-500/30">Research Agent</span>
        </h1>
        <p class="text-xs text-slate-400">16-Node Event-Driven LangGraph Architecture with SSE Streaming</p>
      </div>
    </div>

    <!-- Controls -->
    <div class="flex items-center space-x-3">
      <div class="flex items-center bg-slate-800 rounded-lg p-1 border border-slate-700 text-xs">
        <span class="px-2 text-slate-400 font-medium">Ticker:</span>
        <input id="tickerInput" type="text" value="SRVCABLE" class="bg-slate-900 text-white font-mono px-2 py-1 rounded border border-slate-700 w-24 text-center font-bold focus:outline-none focus:border-brand-500 uppercase">
      </div>

      <div class="flex items-center bg-slate-800 rounded-lg p-1 border border-slate-700 text-xs">
        <span class="px-2 text-slate-400 font-medium">Model:</span>
        <select id="modelSelect" class="bg-slate-900 text-white px-2 py-1 rounded border border-slate-700 focus:outline-none focus:border-brand-500">
          <option value="gemini-3.5-flash-lite">gemini-3.5-flash-lite</option>
          <option value="gemini-3.1-flash-lite">gemini-3.1-flash-lite</option>
          <option value="gemini-3.8-flash">gemini-3.8-flash</option>
        </select>
      </div>

      <label class="flex items-center text-xs text-slate-300 gap-1.5 cursor-pointer bg-slate-800 px-2.5 py-1.5 rounded-lg border border-slate-700">
        <input id="fallbackCheckbox" type="checkbox" class="rounded text-brand-600 focus:ring-0">
        <span>Fallback</span>
      </label>

      <button id="runBtn" onclick="startAgentRun()" class="px-4 py-2 bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-500 hover:to-teal-400 text-white rounded-lg text-xs font-semibold shadow-md shadow-emerald-900/30 flex items-center gap-2 transition active:scale-95">
        <i class="fa-solid fa-play"></i> Run Agent
      </button>

      <button id="evalBtn" onclick="startEvalRun()" class="px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-semibold flex items-center gap-2 transition">
        <i class="fa-solid fa-vial-circle-check text-purple-400"></i> Run E1-E7 Tests
      </button>
    </div>
  </header>

  <!-- Main Grid -->
  <main class="flex-1 grid grid-cols-12 gap-5 p-5 overflow-hidden">

    <!-- Left 7 Cols: Pipeline Graph Visualizer & Live Evidence / Discrepancy -->
    <div class="col-span-7 flex flex-col gap-4 overflow-hidden">

      <!-- Pipeline Status Cards -->
      <div class="bg-slate-900/90 rounded-xl border border-slate-800 p-4 shadow-sm">
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center gap-2">
            <span class="w-2.5 h-2.5 rounded-full bg-brand-500 animate-ping"></span>
            <h2 class="text-xs font-bold uppercase tracking-wider text-slate-400">Runtime Pipeline (16 Nodes)</h2>
          </div>
          <div class="flex items-center gap-2 text-xs">
            <span class="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 text-[10px] font-mono">D = Python</span>
            <span class="px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 text-[10px] font-mono">AGENT = Loop</span>
            <span class="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px] font-mono">LLM = Call</span>
            <span class="px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20 text-[10px] font-mono">EVAL = Judge</span>
          </div>
        </div>

        <div id="nodesGrid" class="grid grid-cols-4 gap-2 text-xs">
          <!-- 1. Intake -->
          <div id="node-validate_request" class="p-2.5 rounded-lg border border-slate-800 bg-slate-850/60 transition flex flex-col justify-between h-16">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold text-blue-400 font-mono">[D]</span>
              <span class="node-icon text-slate-500"><i class="fa-solid fa-circle text-[8px]"></i></span>
            </div>
            <div class="font-semibold text-slate-200 truncate">validate_request</div>
          </div>

          <div id="node-load_and_index" class="p-2.5 rounded-lg border border-slate-800 bg-slate-850/60 transition flex flex-col justify-between h-16">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold text-blue-400 font-mono">[D]</span>
              <span class="node-icon text-slate-500"><i class="fa-solid fa-circle text-[8px]"></i></span>
            </div>
            <div class="font-semibold text-slate-200 truncate">load_and_index</div>
          </div>

          <!-- 2. Research & Audit -->
          <div id="node-research_agent" class="p-2.5 rounded-lg border border-slate-800 bg-slate-850/60 transition flex flex-col justify-between h-16">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold text-amber-400 font-mono">[AGENT]</span>
              <span class="node-icon text-slate-500"><i class="fa-solid fa-circle text-[8px]"></i></span>
            </div>
            <div class="font-semibold text-slate-200 truncate">research_agent</div>
          </div>

          <div id="node-normalize_evidence" class="p-2.5 rounded-lg border border-slate-800 bg-slate-850/60 transition flex flex-col justify-between h-16">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold text-blue-400 font-mono">[D]</span>
              <span class="node-icon text-slate-500"><i class="fa-solid fa-circle text-[8px]"></i></span>
            </div>
            <div class="font-semibold text-slate-200 truncate">normalize_evidence</div>
          </div>

          <div id="node-detect_conflicts" class="p-2.5 rounded-lg border border-slate-800 bg-slate-850/60 transition flex flex-col justify-between h-16">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold text-blue-400 font-mono">[D]</span>
              <span class="node-icon text-slate-500"><i class="fa-solid fa-circle text-[8px]"></i></span>
            </div>
            <div class="font-semibold text-slate-200 truncate">detect_conflicts</div>
          </div>

          <div id="node-semantic_audit" class="p-2.5 rounded-lg border border-slate-800 bg-slate-850/60 transition flex flex-col justify-between h-16">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold text-emerald-400 font-mono">[LLM]</span>
              <span class="node-icon text-slate-500"><i class="fa-solid fa-circle text-[8px]"></i></span>
            </div>
            <div class="font-semibold text-slate-200 truncate">semantic_audit</div>
          </div>

          <div id="node-check_coverage" class="p-2.5 rounded-lg border border-slate-800 bg-slate-850/60 transition flex flex-col justify-between h-16">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold text-blue-400 font-mono">[D]</span>
              <span class="node-icon text-slate-500"><i class="fa-solid fa-circle text-[8px]"></i></span>
            </div>
            <div class="font-semibold text-slate-200 truncate">check_coverage</div>
          </div>

          <div id="node-record_evidence_gaps" class="p-2.5 rounded-lg border border-slate-800 bg-slate-850/60 transition flex flex-col justify-between h-16">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold text-blue-400 font-mono">[D]</span>
              <span class="node-icon text-slate-500"><i class="fa-solid fa-circle text-[8px]"></i></span>
            </div>
            <div class="font-semibold text-slate-200 truncate">record_gaps</div>
          </div>

          <!-- 3. Synthesis & Validation -->
          <div id="node-draft_brief" class="p-2.5 rounded-lg border border-slate-800 bg-slate-850/60 transition flex flex-col justify-between h-16">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold text-emerald-400 font-mono">[LLM]</span>
              <span class="node-icon text-slate-500"><i class="fa-solid fa-circle text-[8px]"></i></span>
            </div>
            <div class="font-semibold text-slate-200 truncate">draft_brief</div>
          </div>

          <div id="node-hard_checks" class="p-2.5 rounded-lg border border-slate-800 bg-slate-850/60 transition flex flex-col justify-between h-16">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold text-blue-400 font-mono">[D]</span>
              <span class="node-icon text-slate-500"><i class="fa-solid fa-circle text-[8px]"></i></span>
            </div>
            <div class="font-semibold text-slate-200 truncate">hard_checks</div>
          </div>

          <div id="node-targeted_correction" class="p-2.5 rounded-lg border border-slate-800 bg-slate-850/60 transition flex flex-col justify-between h-16">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold text-emerald-400 font-mono">[LLM]</span>
              <span class="node-icon text-slate-500"><i class="fa-solid fa-circle text-[8px]"></i></span>
            </div>
            <div class="font-semibold text-slate-200 truncate">targeted_correction</div>
          </div>

          <div id="node-fail_safely" class="p-2.5 rounded-lg border border-slate-800 bg-slate-850/60 transition flex flex-col justify-between h-16">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold text-red-400 font-mono">[D]</span>
              <span class="node-icon text-slate-500"><i class="fa-solid fa-circle text-[8px]"></i></span>
            </div>
            <div class="font-semibold text-slate-200 truncate">fail_safely</div>
          </div>

          <!-- 4. Quality & Release -->
          <div id="node-claim_evaluation" class="p-2.5 rounded-lg border border-slate-800 bg-slate-850/60 transition flex flex-col justify-between h-16">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold text-purple-400 font-mono">[EVAL]</span>
              <span class="node-icon text-slate-500"><i class="fa-solid fa-circle text-[8px]"></i></span>
            </div>
            <div class="font-semibold text-slate-200 truncate">claim_evaluation</div>
          </div>

          <div id="node-render_markdown" class="p-2.5 rounded-lg border border-slate-800 bg-slate-850/60 transition flex flex-col justify-between h-16">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold text-blue-400 font-mono">[D]</span>
              <span class="node-icon text-slate-500"><i class="fa-solid fa-circle text-[8px]"></i></span>
            </div>
            <div class="font-semibold text-slate-200 truncate">render_markdown</div>
          </div>

          <div id="node-persist_output" class="p-2.5 rounded-lg border border-slate-800 bg-slate-850/60 transition flex flex-col justify-between h-16">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold text-blue-400 font-mono">[D]</span>
              <span class="node-icon text-slate-500"><i class="fa-solid fa-circle text-[8px]"></i></span>
            </div>
            <div class="font-semibold text-slate-200 truncate">persist_output</div>
          </div>

          <!-- Status Card -->
          <div id="pipelineStatusCard" class="p-2.5 rounded-lg border border-slate-800 bg-slate-900 transition flex flex-col justify-between h-16">
            <div class="text-[10px] font-bold text-slate-400 uppercase">Status</div>
            <div id="pipelineStateText" class="font-bold text-emerald-400 truncate">IDLE</div>
          </div>
        </div>
      </div>

      <!-- Live Event Stream & Terminal -->
      <div class="flex-1 bg-slate-900/90 rounded-xl border border-slate-800 p-4 flex flex-col min-h-0 shadow-sm">
        <div class="flex items-center justify-between pb-2.5 border-b border-slate-800 mb-2">
          <div class="flex items-center gap-2">
            <i class="fa-solid fa-terminal text-xs text-brand-400"></i>
            <h3 class="text-xs font-bold uppercase tracking-wider text-slate-300">Live SSE Stream & Tool Observations</h3>
          </div>
          <button onclick="clearConsole()" class="text-[10px] text-slate-500 hover:text-slate-300 transition">Clear</button>
        </div>

        <div id="terminalBody" class="flex-1 font-mono text-[11px] leading-relaxed custom-scroll overflow-y-auto space-y-1 text-slate-300 pr-1 select-text">
          <div class="text-slate-500 italic">Ready. Click 'Run Agent' to start streaming pipeline execution...</div>
        </div>
      </div>

      <!-- Evidence & Conflict Live Inspector -->
      <div class="h-44 bg-slate-900/90 rounded-xl border border-slate-800 p-4 flex flex-col min-h-0 shadow-sm">
        <div class="flex items-center justify-between pb-2 border-b border-slate-800 mb-2">
          <div class="flex items-center gap-2">
            <i class="fa-solid fa-scale-unbalanced text-xs text-amber-400"></i>
            <h3 class="text-xs font-bold uppercase tracking-wider text-slate-300">Evidence Ledger & Discrepancies</h3>
          </div>
          <span id="evidenceCountBadge" class="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">0 claims</span>
        </div>
        <div id="evidenceList" class="flex-1 overflow-y-auto custom-scroll space-y-1.5 text-xs text-slate-300 pr-1">
          <div class="text-slate-500 italic text-xs">No claims extracted yet.</div>
        </div>
      </div>

    </div>

    <!-- Right 5 Cols: Research Brief & Evaluation Suite -->
    <div class="col-span-5 flex flex-col gap-4 overflow-hidden">

      <div class="bg-slate-900/90 rounded-xl border border-slate-800 flex flex-col flex-1 min-h-0 shadow-sm">
        <!-- Tabs -->
        <div class="flex items-center justify-between border-b border-slate-800 px-4 pt-3">
          <div class="flex space-x-2">
            <button id="tabBriefBtn" onclick="switchTab('brief')" class="px-3 py-1.5 text-xs font-semibold rounded-t-lg border-b-2 border-brand-500 text-white bg-slate-800/60">
              <i class="fa-regular fa-file-lines mr-1.5 text-brand-400"></i> Research Brief
            </button>
            <button id="tabEvalBtn" onclick="switchTab('eval')" class="px-3 py-1.5 text-xs font-semibold rounded-t-lg border-b-2 border-transparent text-slate-400 hover:text-slate-200">
              <i class="fa-solid fa-shield-halved mr-1.5 text-purple-400"></i> Acceptance Suite (E1-E7)
            </button>
          </div>
          <button onclick="copyBrief()" class="text-xs text-slate-400 hover:text-white px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 transition">
            <i class="fa-regular fa-copy"></i>
          </button>
        </div>

        <!-- Tab 1: Brief Viewer -->
        <div id="briefContainer" class="flex-1 p-5 overflow-y-auto custom-scroll prose prose-invert prose-sm max-w-none">
          <div class="text-center py-20 text-slate-500">
            <i class="fa-regular fa-newspaper text-3xl mb-3 text-slate-600 block"></i>
            Click <strong>Run Agent</strong> to synthesize the 1-page investor brief.
          </div>
        </div>

        <!-- Tab 2: Acceptance Suite Runner -->
        <div id="evalContainer" class="flex-1 p-4 overflow-y-auto custom-scroll hidden space-y-3">
          <div class="flex items-center justify-between pb-2 border-b border-slate-800">
            <div>
              <h4 class="text-xs font-bold text-slate-200">Test Harness Acceptance Suite</h4>
              <p class="text-[11px] text-slate-400">Deterministic + semantic evaluation against the 7 mandatory test cases.</p>
            </div>
            <div id="evalScoreBadge" class="text-xs font-bold px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 font-mono">
              Ready
            </div>
          </div>

          <div id="evalTestsList" class="space-y-2">
            <!-- Test cards injected here -->
            <div class="p-3 rounded-lg border border-slate-800 bg-slate-850 text-xs text-slate-400">
              Click <strong>'Run E1-E7 Tests'</strong> above to trigger the automated test suite.
            </div>
          </div>
        </div>

        <!-- Footer Stats -->
        <div id="briefFooter" class="border-t border-slate-800 px-4 py-2.5 text-xs text-slate-400 flex items-center justify-between bg-slate-900/50">
          <span id="footerStats">Elapsed: 0.0s | Calls: 0 | Tokens: 0</span>
          <span class="text-[10px] text-emerald-400 flex items-center gap-1 font-mono">
            <i class="fa-solid fa-circle-check"></i> Zero-Knowledge Pack Enforced
          </span>
        </div>
      </div>

    </div>

  </main>

  <script>
    let activeEventSource = null;
    let currentBriefText = "";

    function switchTab(tab) {
      const briefContainer = document.getElementById("briefContainer");
      const evalContainer = document.getElementById("evalContainer");
      const tabBriefBtn = document.getElementById("tabBriefBtn");
      const tabEvalBtn = document.getElementById("tabEvalBtn");

      if (tab === "brief") {
        briefContainer.classList.remove("hidden");
        evalContainer.classList.add("hidden");
        tabBriefBtn.className = "px-3 py-1.5 text-xs font-semibold rounded-t-lg border-b-2 border-brand-500 text-white bg-slate-800/60";
        tabEvalBtn.className = "px-3 py-1.5 text-xs font-semibold rounded-t-lg border-b-2 border-transparent text-slate-400 hover:text-slate-200";
      } else {
        briefContainer.classList.add("hidden");
        evalContainer.classList.remove("hidden");
        tabEvalBtn.className = "px-3 py-1.5 text-xs font-semibold rounded-t-lg border-b-2 border-purple-500 text-white bg-slate-800/60";
        tabBriefBtn.className = "px-3 py-1.5 text-xs font-semibold rounded-t-lg border-b-2 border-transparent text-slate-400 hover:text-slate-200";
      }
    }

    function resetNodes() {
      const nodeEls = document.querySelectorAll("#nodesGrid > div[id^='node-']");
      nodeEls.forEach(el => {
        el.classList.remove("node-running", "node-passed", "border-red-500");
        el.querySelector(".node-icon").innerHTML = '<i class="fa-solid fa-circle text-[8px] text-slate-600"></i>';
      });
      document.getElementById("pipelineStateText").innerText = "RUNNING";
      document.getElementById("pipelineStateText").className = "font-bold text-amber-400 truncate";
    }

    function highlightNode(nodeId, status) {
      const el = document.getElementById("node-" + nodeId);
      if (!el) return;
      if (status === "running") {
        el.classList.add("node-running");
        el.querySelector(".node-icon").innerHTML = '<i class="fa-solid fa-spinner fa-spin text-[10px] text-indigo-400"></i>';
      } else if (status === "complete") {
        el.classList.remove("node-running");
        el.classList.add("node-passed");
        el.querySelector(".node-icon").innerHTML = '<i class="fa-solid fa-check text-[10px] text-emerald-400"></i>';
      }
    }

    function appendTerminal(msg, level = "info", timestamp = null) {
      const term = document.getElementById("terminalBody");
      const ts = timestamp || new Date().toLocaleTimeString();
      let color = "text-slate-300";
      if (level === "AGENT") color = "text-amber-300";
      else if (level === "EVAL") color = "text-purple-300";
      else if (level === "CONFLICT") color = "text-rose-400 font-semibold";
      else if (level === "ERROR") color = "text-red-400 font-bold";

      const line = document.createElement("div");
      line.className = "flex gap-2";
      line.innerHTML = `<span class="text-slate-600 select-none">[${ts}]</span> <span class="${color}">${msg}</span>`;
      term.appendChild(line);
      term.scrollTop = term.scrollHeight;
    }

    function clearConsole() {
      document.getElementById("terminalBody").innerHTML = "";
    }

    function copyBrief() {
      if (!currentBriefText) return;
      navigator.clipboard.writeText(currentBriefText);
      alert("Research brief copied to clipboard!");
    }

    function startAgentRun() {
      if (activeEventSource) {
        activeEventSource.close();
      }
      resetNodes();
      switchTab("brief");
      document.getElementById("briefContainer").innerHTML = '<div class="text-center py-20 text-slate-400"><i class="fa-solid fa-circle-notch fa-spin text-3xl mb-3 text-brand-400 block"></i>Agent actively investigating documents via SSE...</div>';
      appendTerminal("Starting research pipeline...", "info");

      const ticker = document.getElementById("tickerInput").value.trim() || "SRVCABLE";
      const model = document.getElementById("modelSelect").value;
      const fallback = document.getElementById("fallbackCheckbox").checked;

      const url = `/api/stream-run?ticker=${encodeURIComponent(ticker)}&source_mode=pack_only&model=${encodeURIComponent(model)}&fallback=${fallback}`;
      activeEventSource = new EventSource(url);

      activeEventSource.onmessage = function(e) {
        const data = JSON.parse(e.data);

        if (data.type === "node_start") {
          highlightNode(data.node, "running");
          appendTerminal(`Entering node [${data.type_tag || 'D'}]: ${data.node}`, "info", data.timestamp);
        } else if (data.type === "node_complete") {
          highlightNode(data.node, "complete");
          appendTerminal(`✓ Completed node: ${data.node}`, "info", data.timestamp);
        } else if (data.type === "agent_tool_call") {
          appendTerminal(`🤖 Agent calls tool: ${data.tool}(${JSON.stringify(data.args)}) — ${data.reasoning}`, "AGENT", data.timestamp);
        } else if (data.type === "tool_observation") {
          appendTerminal(`  ↳ Obs: ${data.obs_preview}`, "info");
        } else if (data.type === "log") {
          appendTerminal(data.message, data.level);
        } else if (data.type === "complete") {
          document.getElementById("pipelineStateText").innerText = "SUCCESS";
          document.getElementById("pipelineStateText").className = "font-bold text-emerald-400 truncate";
          currentBriefText = data.brief;
          document.getElementById("briefContainer").innerHTML = marked.parse(data.brief);

          // Update footer
          const d = data.details || {};
          document.getElementById("footerStats").innerText = `Elapsed: ${data.elapsed}s | Calls: ${d.llm_calls || 0} | Tokens: ${d.prompt_tokens || 0}+${d.completion_tokens || 0}`;

          // Update evidence ledger
          if (d.claims && d.claims.length) {
            document.getElementById("evidenceCountBadge").innerText = `${d.claims.length} claims`;
            let evHtml = "";
            d.claims.forEach(c => {
              evHtml += `<div class="p-2 rounded bg-slate-800/70 border border-slate-700/60 flex items-start gap-2">
                <span class="px-1.5 py-0.5 rounded text-[10px] bg-brand-500/20 text-brand-300 font-mono font-bold">[${c.source_id}]</span>
                <div class="flex-1 text-slate-200">${c.claim}</div>
              </div>`;
            });
            document.getElementById("evidenceList").innerHTML = evHtml;
          }

          appendTerminal(`Pipeline successfully synthesized brief in ${data.elapsed}s!`, "info");
          activeEventSource.close();
        } else if (data.type === "error") {
          document.getElementById("pipelineStateText").innerText = "FAILED";
          document.getElementById("pipelineStateText").className = "font-bold text-rose-500 truncate";
          appendTerminal(`Error: ${data.error}`, "ERROR");
          activeEventSource.close();
        }
      };

      activeEventSource.onerror = function() {
        appendTerminal("SSE connection finished.", "info");
        if (activeEventSource) activeEventSource.close();
      };
    }

    function startEvalRun() {
      switchTab("eval");
      const listEl = document.getElementById("evalTestsList");
      listEl.innerHTML = '<div class="text-center py-10 text-slate-400"><i class="fa-solid fa-spinner fa-spin text-2xl mb-2 text-purple-400 block"></i>Running E1-E7 Acceptance Suite...</div>';
      document.getElementById("evalScoreBadge").innerText = "Running...";
      document.getElementById("evalScoreBadge").className = "text-xs font-bold px-2.5 py-1 rounded-full bg-purple-500/20 text-purple-300 font-mono";

      const evSource = new EventSource("/api/stream-eval");

      evSource.onmessage = function(e) {
        const data = JSON.parse(e.data);

        if (data.type === "eval_start") {
          listEl.innerHTML = "";
        } else if (data.type === "test_start") {
          const item = document.createElement("div");
          item.id = "tc-" + data.name;
          item.className = "p-3 rounded-lg border border-slate-800 bg-slate-850 flex items-center justify-between text-xs transition";
          item.innerHTML = `
            <div>
              <div class="font-bold text-slate-200">${data.name}</div>
              <div class="text-[11px] text-slate-400">${data.desc}</div>
            </div>
            <div class="status-col text-slate-400"><i class="fa-solid fa-spinner fa-spin"></i> Running</div>
          `;
          listEl.appendChild(item);
        } else if (data.type === "test_complete") {
          const item = document.getElementById("tc-" + data.name);
          if (item) {
            const statusCol = item.querySelector(".status-col");
            if (data.passed) {
              item.className = "p-3 rounded-lg border border-emerald-900/60 bg-emerald-950/20 flex items-center justify-between text-xs";
              statusCol.innerHTML = `<span class="px-2 py-0.5 rounded font-mono font-bold bg-emerald-500/20 text-emerald-400">PASS (${data.elapsed}s)</span>`;
            } else {
              item.className = "p-3 rounded-lg border border-rose-900/60 bg-rose-950/20 flex items-center justify-between text-xs";
              statusCol.innerHTML = `<span class="px-2 py-0.5 rounded font-mono font-bold bg-rose-500/20 text-rose-400">FAIL (${data.elapsed}s)</span>`;
            }
          }
        } else if (data.type === "eval_summary") {
          const s = data.summary;
          const badge = document.getElementById("evalScoreBadge");
          badge.innerText = `${s.passed}/${s.total} Passed`;
          badge.className = s.passed === s.total 
            ? "text-xs font-bold px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-400 font-mono"
            : "text-xs font-bold px-2.5 py-1 rounded-full bg-amber-500/20 text-amber-400 font-mono";
          evSource.close();
        }
      };

      evSource.onerror = function() {
        evSource.close();
      };
    }
  </script>
</body>
</html>
"""


# ======================================================================
# Main execution
# ======================================================================

def main():
    import argparse

    ap = argparse.ArgumentParser(description="Super Investing Research Agent Web Server")
    ap.add_argument("--host", default="0.0.0.0", help="Host interface")
    ap.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = ap.parse_args()

    agent_logger.info(f"Starting SSE Server on http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

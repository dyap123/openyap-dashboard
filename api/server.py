"""
OpenYap Dashboard AI Layout API

Lightweight FastAPI server that uses MiniMax to analyze sheet data
and recommend dashboard layouts. Runs independently from the main
openyap web UI.

Usage:
    cd ~/openyap-dashboard/api
    python server.py

Runs on port 8490. The GitHub Pages dashboard calls this when
your Mac is online. Falls back to auto-detect if unreachable.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add const-agent to path for keyring access
sys.path.insert(0, str(Path.home() / "const-agent"))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenYap Dashboard AI")

# Allow CORS from GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dyap123.github.io", "http://localhost", "http://127.0.0.1"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


def get_minimax_client():
    """Get MiniMax API key from keyring (same as const-agent)."""
    try:
        import keyring
        api_key = keyring.get_password("minimax", "api_key")
        if not api_key:
            # Try alternate keyring entry
            api_key = keyring.get_password("const-agent", "minimax_api_key")
        return api_key
    except Exception as e:
        logger.error(f"Failed to get MiniMax key: {e}")
        return None


async def call_minimax(prompt: str, max_tokens: int = 4096) -> str:
    """Call MiniMax M2.5 via their Anthropic-compatible API."""
    import httpx

    api_key = get_minimax_client()
    if not api_key:
        raise ValueError("MiniMax API key not found in keyring")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.minimax.io/anthropic/v1/messages",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "MiniMax-M2.5",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        # Extract text from Anthropic-style response
        content = data.get("content", [])
        if content and isinstance(content, list):
            return content[0].get("text", "")
        return ""


@app.post("/analyze")
async def analyze_data(request: Request):
    """
    Analyze sheet data and recommend a dashboard layout.

    Expects JSON body:
    {
        "title": "Dashboard Title",
        "columns": ["Col A", "Col B", ...],
        "sample_rows": [[val, val, ...], ...],  // first 20 rows
        "row_count": 150,
        "column_types": {"Col A": "text", "Col B": "number", ...}
    }

    Returns:
    {
        "kpis": [{"column": "Budget", "type": "sum", "label": "Total Budget"}, ...],
        "chart": {"type": "bar", "label_col": "Phase", "value_col": "Budget", "title": "Budget by Phase"},
        "status_columns": ["Status", "Priority"],
        "highlight_columns": ["Deadline", "Owner"],
        "insights": ["3 items are overdue", "Total budget is $1.2M"],
        "sort_by": {"column": "Deadline", "direction": "asc"},
        "table_columns": ["Name", "Status", "Budget", "Deadline", "Owner"]
    }
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    title = body.get("title", "Dashboard")
    columns = body.get("columns", [])
    sample_rows = body.get("sample_rows", [])
    row_count = body.get("row_count", len(sample_rows))
    column_types = body.get("column_types", {})

    if not columns or not sample_rows:
        return JSONResponse({"error": "No data provided"}, status_code=400)

    # Build the analysis prompt
    sample_table = "\n".join(
        [" | ".join(columns)] +
        ["---" * len(columns)] +
        [" | ".join(str(v) for v in row[:len(columns)]) for row in sample_rows[:15]]
    )

    prompt = f"""You are a data visualization expert. Analyze this dataset and recommend the best dashboard layout.

DATASET: "{title}"
COLUMNS: {json.dumps(columns)}
COLUMN TYPES (auto-detected): {json.dumps(column_types)}
TOTAL ROWS: {row_count}

SAMPLE DATA (first 15 rows):
{sample_table}

Based on this data, respond with ONLY a JSON object (no markdown, no explanation) with this structure:
{{
    "kpis": [
        {{"column": "column_name", "type": "sum|average|count|max|min|latest", "label": "Display Label"}},
        // 3-5 KPIs that would be most useful. Pick numeric columns or count-based metrics.
    ],
    "chart": {{
        "type": "bar|none",
        "label_col": "column for labels (categories)",
        "value_col": "column for values (numbers)",
        "title": "Chart title"
    }},
    "status_columns": ["columns that contain status-like values (done/open/pending/etc)"],
    "highlight_columns": ["columns that should be visually emphasized in the table"],
    "insights": ["3-5 short insights about the data, like 'X items are overdue' or 'Total is $Y'"],
    "sort_by": {{"column": "best column to sort by default", "direction": "asc|desc"}},
    "table_columns": ["ordered list of columns to show in the table, most important first"],
    "hide_columns": ["columns to hide from the table (like IDs or internal fields)"]
}}

Rules:
- Only reference columns that actually exist in the data
- KPIs should highlight the most actionable numbers
- Insights should be specific and reference actual values from the data
- If there's no good chart to make, set chart.type to "none"
- Put the most important columns first in table_columns
- Status columns get colored pills automatically
- Be concise in insights — one line each"""

    try:
        response_text = await call_minimax(prompt)
        # Parse the JSON from the response
        # Try to extract JSON from the response (might have extra text)
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            layout = json.loads(response_text[json_start:json_end])
            return {"layout": layout, "ai": True}
        else:
            return JSONResponse({"error": "AI returned non-JSON response", "raw": response_text[:500]}, status_code=500)
    except json.JSONDecodeError as e:
        return JSONResponse({"error": f"Failed to parse AI response: {e}", "raw": response_text[:500]}, status_code=500)
    except Exception as e:
        logger.exception("MiniMax call failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    """Health check — also verifies MiniMax key is available."""
    key = get_minimax_client()
    return {
        "status": "ok",
        "minimax": "available" if key else "missing",
    }


if __name__ == "__main__":
    print("\n  OpenYap Dashboard AI API")
    print("  Port: 8490")
    print("  CORS: dyap123.github.io\n")
    uvicorn.run(app, host="0.0.0.0", port=8490, log_level="info")

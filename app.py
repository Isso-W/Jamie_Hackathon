"""Requirement Clarifier: requester posts a task -> AI breaks it down and asks follow-ups -> brief is generated
-> an interactive requirement demo is generated -> the worker asks questions while working;
everything is captured in a project knowledge base: one document per project, shown as a cover card, click to see details.

Run:
    pip install -r requirements.txt
    cd harness && npm install          (pi agent harness; needs Node 22+)
    Put your Qwen (Alibaba Cloud Model Studio, international site) API key in apikey.txt in the same folder
    streamlit run app.py               (starts the pi harness sidecar automatically)

Engine:
    Every AI call is a job run by the pi agent harness (harness/server.mjs): one fresh pi session per job.
    If the harness fails, the call falls back to the direct Qwen API, then to the recording.

Demo:
    During normal calls, every AI answer is automatically recorded to demo_cache.json.
    On stage, turn on "Offline demo mode" in the sidebar and repeat the same steps as in rehearsal;
    the recorded answers are replayed directly, with no network needed.
    Even without offline mode, if an API call fails it automatically falls back to the recording (if one exists).
"""

import base64
import hashlib
import html as htmllib
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

MODEL = "qwen3.8-max"  # Direct-API fallback model
PI_MODEL = "dashscope/qwen3.8-max"  # Default harness model, as provider/id from harness/agent/models.json
HARNESS_DIR = Path(__file__).parent / "harness"
HARNESS_URL = "http://127.0.0.1:8765"
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"  # International site; for mainland China keys use https://dashscope.aliyuncs.com/compatible-mode/v1
KEY_FILE = Path(__file__).parent / "apikey.txt"
CACHE_FILE = Path(__file__).parent / "demo_cache.json"

# Demo scenarios: each maps to a project, with preset worker / knowledge-base questions (click a button on stage, no live typing)
SCENARIOS = {
    "Calculator": {
        "project": "Calculator App",
        "task": "Build me a calculator for everyday bookkeeping, a bit nicer than the one that comes with my phone. Ideally I can see something this week.",
        "worker_qs": ["Does it need keyboard input?", "How long should history be kept?"],
        "kb_qs": ["Which features were finally decided for the calculator?", "Which features are explicitly out of scope?"],
    },
    "Team task board": {
        "project": "Team Task Board",
        "task": "Make a task board for the team so everyone can see who is doing what and how far along it is. No more asking for status in the group chat every day.",
        "worker_qs": ["Do task cards need due dates?", "Should we distinguish admins from regular members?"],
        "kb_qs": ["What columns does the board have?", "Which questions are still open?"],
    },
    "Event sign-up page": {
        "project": "Tech Talk Sign-up",
        "task": "We're hosting a tech talk next month and need a sign-up page. Seats are limited, so sign-ups should close automatically when it's full, and ideally we can see the attendee list.",
        "worker_qs": ["What information should attendees fill in?", "Should there be a waitlist once it's full?"],
        "kb_qs": ["What is the seat limit?", "What is the deadline?"],
    },
}

LANG_RULE = "Answer in the same language as the task description."

PAGES = ["Stakeholders", "Worker", "Project Knowledge Base"]
STEPS = ["Post task", "AI questions", "Task brief", "Requirement demo", "Worker Q&A", "Knowledge captured"]
COVER_COLORS = ["#4f46e5", "#0f766e", "#c2410c", "#be185d", "#1d4ed8", "#7c3aed", "#15803d", "#b45309"]
WAITING = "⏳ Awaiting a stakeholder's answer"
ALL_KB = "__all__"  # Marker in kb_chat for cross-project Q&A
ALL_QS = [
    "Which projects still have questions waiting for a stakeholder?",
    "Which project is closest to its deadline?",
    "What key decisions has each project made?",
]

# ---------- JSON schemas (structured output enforced via tool use) ----------

ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Short task title, at most 8 words"},
        "understanding": {"type": "string", "description": "Restate your understanding of the task goal in one or two sentences"},
        "deadline": {"type": "string", "description": "Deadline mentioned in the task, resolved from today's date as YYYY-MM-DD; leave empty if not mentioned"},
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "detail": {"type": "string"}},
                "required": ["title", "detail"],
            },
            "description": "Initial breakdown into 3 to 6 subtasks",
        },
        "known": {"type": "array", "items": {"type": "string"}, "description": "Information already made clear in the task"},
        "questions": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "to": {"type": "string", "description": "Name of the stakeholder who owns this decision, exactly as in the stakeholder list"},
                    "why": {"type": "string", "description": "Why ask this, and what goes wrong if it isn't clarified"},
                    "options": {"type": "array", "items": {"type": "string"}, "description": "Possible answers; leave empty if none"},
                },
                "required": ["question", "to", "why", "options"],
            },
            "description": "This round's most critical clarifying questions, at most 3, ordered by importance",
        },
        "needs_meeting": {"type": "boolean", "description": "Whether it involves multi-party decisions or conflicting interests that need a face-to-face talk"},
        "meeting_reason": {"type": "string"},
        "stakeholders": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "A first name, or 'Requester' for the poster if unnamed"},
                    "role": {"type": "string"},
                    "cares": {"type": "string", "description": "What this person wants from the result, in a few words"},
                },
                "required": ["name", "role", "cares"],
            },
            "description": "2 to 4 people whose wishes matter, inferred from the task; the first is the person who posted it",
        },
    },
    "required": ["title", "understanding", "deadline", "subtasks", "known", "questions", "needs_meeting", "meeting_reason", "stakeholders"],
}

CLARIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "understanding": {"type": "string", "description": "Your understanding of the task goal, updated with every answer so far, in one or two sentences"},
        "complete": {"type": "boolean", "description": "True only when no open decision would still change what gets built"},
        "remaining": {"type": "array", "items": {"type": "string"},
                      "description": "Decisions still open that matter; for 'not sure' answers, include your recommended default"},
        "questions": ANALYZE_SCHEMA["properties"]["questions"],
        "conflicts": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "What the conflict is about, in a few words"},
                    "sides": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"who": {"type": "string"}, "wants": {"type": "string"}},
                            "required": ["who", "wants"],
                        },
                    },
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "A concrete resolution, ideally a compromise"},
                                "tradeoff": {"type": "string", "description": "What each side gains and gives up"},
                            },
                            "required": ["label", "tradeoff"],
                        },
                        "description": "2 or 3 resolutions, the recommended compromise first",
                    },
                },
                "required": ["topic", "sides", "options"],
            },
            "description": "Contradictions between stakeholders' wishes or answers for the owner to settle; never ones already decided (D lines)",
        },
    },
    "required": ["understanding", "complete", "remaining", "questions", "conflicts"],
}
MAX_ROUNDS = 4  # Hard stop so a demo never loops forever; whatever is still open goes into the brief's open issues

BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "in_scope": {"type": "array", "items": {"type": "string"}},
        "out_of_scope": {"type": "array", "items": {"type": "string"}},
        "acceptance_criteria": {"type": "array", "items": {"type": "string"}, "description": "Checkable acceptance criteria"},
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "detail": {"type": "string"}},
                "required": ["title", "detail"],
            },
        },
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"decision": {"type": "string"}, "source": {"type": "string", "description": "Which question's answer this came from"}},
                "required": ["decision", "source"],
            },
        },
        "open_issues": {"type": "array", "items": {"type": "string"}, "description": "Questions still undecided"},
    },
    "required": ["goal", "in_scope", "out_of_scope", "acceptance_criteria", "subtasks", "decisions", "open_issues"],
}

WORKER_SCHEMA = {
    "type": "object",
    "properties": {
        "can_answer": {"type": "boolean", "description": "Whether the brief and existing Q&A give enough basis to answer"},
        "answer": {"type": "string", "description": "The answer if possible; otherwise explain what is missing"},
        "source": {"type": "string", "description": "Basis for the answer: which part of the brief, or which past Q&A"},
        "question_for_requester": {"type": "string", "description": "If it can't be answered, a clear question to forward to a stakeholder"},
        "ask_who": {"type": "string", "description": "If it can't be answered, the stakeholder who owns that decision, exactly as in the stakeholder list"},
    },
    "required": ["can_answer", "answer", "source", "question_for_requester", "ask_who"],
}

KB_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean", "description": "Whether the project document and timeline give enough basis to answer"},
        "answer": {"type": "string", "description": "The answer; if the basis is insufficient, explain what is missing"},
        "sources": {"type": "array", "items": {"type": "string"}, "description": "Cited passages: section headings or original question text from the document"},
    },
    "required": ["found", "answer", "sources"],
}

ALL_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean", "description": "Whether all the project documents give enough basis to answer"},
        "answer": {"type": "string", "description": "The answer, stating which project each piece of information comes from; if the basis is insufficient, explain what is missing"},
        "sources": {"type": "array", "items": {"type": "string"}, "description": "Citations, formatted as: project name · section heading or original question text"},
        "projects": {"type": "array", "items": {"type": "string"}, "description": "Projects the answer involves; must exactly match the project titles in the documents"},
    },
    "required": ["found", "answer", "sources", "projects"],
}

# ---------- AI calls: pi harness (direct Qwen API as fallback) + record / replay ----------


@st.cache_resource
def get_client():
    return OpenAI(api_key=KEY_FILE.read_text(encoding="utf-8").strip(), base_url=BASE_URL)


@st.cache_resource
def get_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_cache():
    CACHE_FILE.write_text(json.dumps(get_cache(), ensure_ascii=False, indent=1), encoding="utf-8")


def cached(name: str, system: str, content: str, produce) -> dict:
    """Check the recording first; in offline mode replay it, otherwise call for real and record, falling back to the recording on failure."""
    # Strip the date line so answers recorded in rehearsal can still be replayed the next day
    norm = re.sub(r"Today is \d{4}-\d{2}-\d{2}\.\n?", "", content)
    cache, key = get_cache(), hashlib.sha1(f"{name}\n{system}\n{norm}".encode("utf-8")).hexdigest()
    if st.session_state.get("replay") and key in cache:
        time.sleep(0.8)  # Keep a little "thinking" pause so the replay doesn't look fake
        return json.loads(json.dumps(cache[key]))
    try:
        data = produce()
    except Exception:
        if key not in cache:
            raise
        st.toast("API call failed, switched to the recorded answer")
        return json.loads(json.dumps(cache[key]))
    cache[key] = data
    save_cache()
    return data


def harness_get(path: str, timeout: float = 3) -> dict:
    with urllib.request.urlopen(HARNESS_URL + path, timeout=timeout) as r:
        return json.loads(r.read())


@st.cache_resource
def start_harness() -> dict:
    """Reuse a running sidecar, or start one and wait for it; returns its /health."""
    try:
        return harness_get("/health")
    except OSError:
        pass
    subprocess.Popen(["node", "server.mjs"], cwd=HARNESS_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(60):
        time.sleep(0.5)
        try:
            return harness_get("/health")
        except OSError:
            pass
    raise RuntimeError("The pi harness did not start (run `npm install` in harness/)")


def run_job(job: dict) -> dict:
    """Run one job in a fresh pi session. Job keys: system, prompt, schema, tools, workspace, skills, model."""
    start_harness()
    job = {"model": st.session_state.get("pi_model") or PI_MODEL, **job}
    req = urllib.request.Request(HARNESS_URL + "/run", data=json.dumps(job).encode("utf-8"),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(json.loads(e.read()).get("error", str(e))) from None


def via_harness(job: dict, direct):
    """Prefer pi; on failure (or with the direct engine picked) use the direct Qwen API."""
    if st.session_state.get("engine", "pi") == "pi":
        try:
            return run_job(job)
        except Exception as e:
            st.toast(f"pi harness failed, using the direct API: {e}")
    return direct()


def call_tool(system: str, content: str, name: str, schema: dict, skills: tuple = ()) -> dict:
    def direct():
        resp = get_client().chat.completions.create(
            model=MODEL,
            max_tokens=3000,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            tools=[{"type": "function", "function": {"name": name, "description": "Output the result in the specified structure", "parameters": schema}}],
            tool_choice={"type": "function", "function": {"name": name}},
            extra_body={"enable_thinking": False},  # Forced tool_choice isn't supported in thinking mode
        )
        calls = resp.choices[0].message.tool_calls
        if not calls:
            raise RuntimeError("The model did not return a structured result")
        return {"result": json.loads(calls[0].function.arguments)}

    def produce():
        return via_harness({"system": system, "prompt": content, "schema": schema, "skills": list(skills)}, direct)["result"]

    return coerce(cached(name, system, content, produce), schema)


def coerce(v, schema: dict):
    """Fix the model's output to match the schema, so rendering never hits KeyError / "string indices must be integers".
    The model occasionally drops fields, returns nested arrays/objects as JSON strings, or array items as bare strings."""
    t = schema.get("type")
    if t in ("array", "object") and isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
    if t == "array":
        if not isinstance(v, list):
            v = [v] if v else []
        return [coerce(x, schema.get("items", {})) for x in v]
    if t == "object":
        props = schema.get("properties", {})
        if isinstance(v, str) and props:  # A bare string where an object was expected: put it in the first field
            v = {next(iter(props)): v}
        if not isinstance(v, dict):
            v = {}
        defaults = {"array": [], "string": "", "boolean": False, "object": {}}
        for k in schema.get("required", []):
            v.setdefault(k, defaults.get(props.get(k, {}).get("type")))
        return {k: coerce(x, props[k]) if k in props else x for k, x in v.items()}
    if t == "string" and v is None:
        return ""
    return v


def workspace_name() -> str:
    """Folder under workspaces/ where pi keeps this task's files (ASCII only, so Python and the sidecar agree on it)."""
    return f"task{S.task_id}-{hashlib.md5(S.task_project.encode('utf-8')).hexdigest()[:6]}"


def call_agent(system: str, content: str, name: str, workspace: str, out_file: str,
               tools: tuple = ("read", "write", "edit", "ls")) -> dict:
    """A tool-using pi job whose output is a file it writes in workspaces/<workspace>/. Returns {"text": file content, "steps"}.
    The direct-API fallback has no tools, so it asks for the whole file in one reply instead."""
    path = Path(__file__).parent / "workspaces" / workspace / out_file

    def direct():
        resp = get_client().chat.completions.create(
            model=MODEL,
            max_tokens=8000,
            messages=[{"role": "system", "content": f"{system} You have no tools: output the whole {out_file} in one code block, with no explanation."},
                      {"role": "user", "content": content}],
            extra_body={"enable_thinking": False},
        )
        text = resp.choices[0].message.content or ""
        m = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.S)
        return {"file": (m.group(1) if m else text).strip(), "steps": []}

    def produce():
        r = via_harness({"system": system, "prompt": content, "tools": list(tools), "workspace": workspace}, direct)
        if "file" not in r and not path.exists():
            st.toast(f"pi finished without writing {out_file}, using the direct API")
            r = direct()
        return {"text": r["file"] if "file" in r else path.read_text(encoding="utf-8"), "steps": r["steps"]}

    data = cached(name, system, content, produce)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data["text"], encoding="utf-8")  # Keep the workspace in sync when the result came from a recording or the fallback
    return data


def steps_md(steps: list) -> str:
    """Summarize pi's tool calls, e.g. "read index.html → edit index.html ×2"."""
    counts = {}
    for x in steps:
        target = Path(str((x.get("args") or {}).get("path", ""))).name
        key = f"{x['tool']} {target}".strip()
        counts[key] = counts.get(key, 0) + 1
    return " → ".join(f"{k} ×{n}" if n > 1 else k for k, n in counts.items())


def analyze(task: str) -> dict:
    system = (
        "You are an experienced tech lead on the team, responsible for making sure requirements are understood before work starts. "
        "Break down the task, find ambiguities and hidden assumptions, and ask only the key questions that really affect implementation; don't ask about the obvious. "
        "Don't make assumptions on the requester's behalf: if information is missing, ask. "
        "Name the stakeholders whose wishes matter, and address each question to the one who owns that decision. " + LANG_RULE
    )
    return call_tool(system, f"Today is {date.today()}.\nTask description:\n{task}", "task_analysis", ANALYZE_SCHEMA, skills=("grill-me",))


def is_wish(h: dict) -> bool:
    return h.get("kind") == "wish"


def is_question(h: dict) -> bool:
    return not h.get("kind")  # Wishes and owner decisions carry a kind


def qa_md(qa: list) -> str:
    lines, nq, nw, nd = [], 0, 0, 0
    for h in qa:
        if is_wish(h):
            nw += 1
            lines.append(f"W{nw} (round {h.get('round', 1)}): {h['by']} wishes: {h['a']}")
        elif h.get("kind") == "decision":
            nd += 1
            lines.append(f"D{nd} (round {h.get('round', 1)}): owner {h['by']} settled {h['q']} ({h['sides']}): {h['a']}")
        else:
            nq += 1
            lines.append(f"Q{nq} (round {h.get('round', 1)}): {h['q']}\nA ({h['by']}): {h['a']}")
    return "\n".join(lines) or "(none)"


def clarify(task: str, analysis: dict, people: list, qa: list, round_no: int) -> dict:
    system = (
        "You are an experienced tech lead grilling the stakeholders, round after round, until the request is fully explained. "
        "Address each question to the stakeholder who owns that decision. "
        "Stakeholders may also state wishes unprompted (W lines): treat each as that person's requirement, ask the follow-ups it raises, "
        "If wishes or answers from different stakeholders contradict each other, don't ask about it as a question: put it in conflicts "
        "with who wants what and 2-3 resolutions, best compromise first, for the owner to pick. A conflict goes only into conflicts, never also into questions. "
        "Never re-raise a conflict already decided (D lines). "
        "You get the task, your current understanding and every answer so far. Answers often open new decisions: ask the follow-ups they raise. "
        "Never repeat a question that was already answered. If an answer was 'not sure', don't ask it again: put it in remaining with your recommended default. "
        "Only ask what changes what gets built. Set complete to true, with no questions and no conflicts, once nothing open would change the implementation. " + LANG_RULE
    )
    content = (
        f"Today is {date.today()}.\nTask description:\n{task}\n\nStakeholders:\n{cast_md(people)}\n\n"
        f"Current understanding: {analysis['understanding']}\n\nQ&A so far:\n{qa_md(qa)}\n\nThis is round {round_no} of at most {MAX_ROUNDS}."
    )
    return call_tool(system, content, "clarify_round", CLARIFY_SCHEMA, skills=("grill-me",))


def write_brief(task: str, analysis: dict, people: list, qa: list, remaining: list) -> dict:
    system = (
        "Based on the task description and the clarification Q&A, write a task brief the worker can start on right away. "
        "Every decision must trace back to an answer, a stakeholder's wish (W lines) or an owner's conflict decision (D lines), and who gave it; anything without a basis can only go into open issues. " + LANG_RULE
    )
    content = (
        f"Task description:\n{task}\n\nStakeholders:\n{cast_md(people)}\n\nUnderstanding after clarification: {analysis['understanding']}\n\n"
        f"Clarification Q&A:\n{qa_md(qa)}\n\nStill open: {'; '.join(remaining) or '(nothing)'}"
    )
    return call_tool(system, content, "task_brief", BRIEF_SCHEMA)


def make_demo(task: str, brief: dict, qa: list, feedback: str = "") -> tuple:
    """pi builds the prototype as workspaces/<task>/index.html; change requests edit that file instead of starting over."""
    ws = workspace_name()
    page = Path(__file__).parent / "workspaces" / ws / "index.html"
    if not feedback and page.exists():
        page.unlink()  # A fresh build never starts from a stale page left by an earlier session
    system = (
        "You are a product prototype designer. Based on the stakeholders' answers in the Q&A and the task brief, build an interactive single-file HTML prototype "
        "so the requester can confirm at a glance that \"this is roughly what it will look like\". "
        "Every answer, stakeholder wish (W lines) and owner decision (D lines) that changes what the user sees or does must be visible and working in the prototype (e.g. if they chose CSV export, there is a working Export CSV button); "
        "earlier change requests in the Q&A still apply. Don't add features nobody asked for or answered. "
        "At the bottom, add a small collapsed <details> section titled \"Built from your answers\" mapping each visible feature to the answer, wish or decision (Q/W/D number and who) it came from. "
        "Requirements: implement only the core features within the brief's scope, and buttons and inputs must actually work; use fake data where data is needed; "
        "don't access the network or reference any external files, fonts or CDNs; inline all CSS and JS; "
        "keep the style clean and modern, with UI text in the same language as the task description; keep the overall height within 520px and the width responsive. "
        "Work in the current folder: write the prototype to index.html. If index.html already exists, read it and apply the change requests "
        "with targeted edits instead of rewriting it. When done, reply with one short sentence."
    )
    content = f"Task description:\n{task}\n\nStakeholders' Q&A:\n{qa_md(qa)}\n\nTask brief:\n{brief_md(brief)}"
    if feedback:
        content += f"\n\nThe current prototype is in index.html. Change requests after seeing it (must be reflected): {feedback}"
    data = call_agent(system, content, "demo_html", ws, "index.html")
    if "<" not in data["text"]:
        raise RuntimeError("The model did not return HTML")
    return data["text"], data["steps"]


def answer_worker(task: str, brief: dict, people: list, history: list, question: str) -> dict:
    hist = "\n".join(f"Q: {h['q']}\nA ({h['by']}): {h['a']}" for h in history) or "(none)"
    system = (
        "You are the task's coordination assistant; the worker asks you questions while working. Answer only from the task brief and past Q&A; don't make things up. "
        "If the basis is insufficient, set can_answer to false, turn it into a clear question, and set ask_who to the stakeholder who owns that decision. " + LANG_RULE
    )
    content = f"Task description:\n{task}\n\nStakeholders:\n{cast_md(people)}\n\nTask brief:\n{json.dumps(brief, ensure_ascii=False, indent=2)}\n\nPast Q&A:\n{hist}\n\nWorker's question: {question}"
    return call_tool(system, content, "worker_answer", WORKER_SCHEMA, skills=("grill-me",))


def answer_kb(project: str, info: dict, doc_md: str, events: list, question: str) -> dict:
    timeline = "\n".join(f"{e['time']:%Y-%m-%d %H:%M:%S} | {e['task']} | {e['type']} | {e['text']}" for e in events)
    system = (
        "You are the project knowledge base assistant. Answer only from the given project document and project timeline; don't make things up, "
        "and list the section headings or original question text you used in sources. "
        "If the basis is insufficient, set found to false and explain what is missing. Answer in the same language as the question."
    )
    content = (
        f"Today is {date.today()}.\nProject: {project}\nCreated: {info['created']:%Y-%m-%d %H:%M}\n"
        f"Deadline: {info['deadline'] or 'not set'}\n\nProject document:\n{doc_md or '(none)'}\n\n"
        f"Project timeline:\n{timeline or '(none)'}\n\nQuestion: {question}"
    )
    return call_tool(system, content, "kb_answer", KB_SCHEMA)


def answer_all(question: str) -> dict:
    blocks = "\n\n---\n\n".join(project_doc_md(p) for p in S.projects)
    system = (
        "You are the team's cross-project knowledge base assistant and can search all project documents. Answer only from the given documents; don't make things up, "
        f"and say which project each piece of information comes from. \"{WAITING}\" means the question has not been answered yet. "
        "If the basis is insufficient, set found to false and explain what is missing. Answer in the same language as the question."
    )
    content = f"Today is {date.today()}.\nThere are {len(S.projects)} projects; their documents follow:\n\n{blocks}\n\nQuestion: {question}"
    return call_tool(system, content, "kb_all_answer", ALL_SCHEMA)


# ---------- State ----------

TASK_DEFAULTS = {
    "task": "",
    "task_id": None,  # Assigned when the task is posted
    "task_title": "",
    "task_project": "",  # Project the task belonged to when posted
    "analysis": None,
    "qa": [],  # Clarification answers across all rounds: {"q", "a", "by", "round"}
    "round": 1,  # Current clarification round
    "clarified": False,  # True once the AI has nothing left to ask (or MAX_ROUNDS was hit, or the requester stopped)
    "remaining": [],  # Decisions still open after clarification
    "stopped": False,  # Clarification ended before the AI said it was complete
    "round_ans": {},  # This round's answers so far, question -> {"a", "by"}; the round closes once every question has one
    "cast_seed": [],  # Stakeholder table as it was when the task was analyzed (input to the editor)
    "acting": "",  # Which stakeholder the person at the keyboard is acting as
    "wish_ver": 0,  # +1 per added wish, used to clear the wish input
    "conflicts": [],  # This round's stakeholder conflicts for the owner: {"topic", "sides", "options"}
    "round_dec": {},  # This round's owner decisions so far, conflict topic -> chosen resolution
    "demo_steps": [],  # What pi did to build the current demo: [{"tool", "args"}]
    "brief_qa": 0,  # len(S.qa) when the brief was written; more entries since means the brief is stale
    "brief": None,
    "demo": None,  # HTML of the requirement demo
    "demo_ver": 0,  # +1 on every regeneration, used to clear the change-request input
    "history": [],  # Captured Q&A: {"q", "a", "by"}
    "pending": [],  # Questions waiting for a stakeholder
    "pending_to": {},  # Pending question -> stakeholder it was forwarded to
    "chat": [],  # Worker chat: {"role", "content"}
}
KB_DEFAULTS = {
    "projects": {},  # Project name -> {"created", "deadline"}
    "docs": [],  # Records: {"id", "key", "project", "task_id", "task", "type", "title", "content", "time"}
    "events": [],  # Timeline: {"time", "project", "task", "type", "text"}
    "kb_chat": [],  # Knowledge base Q&A: {"project", "role", "content"}
    "kb_open": None,  # Project currently open in the knowledge base; None means the cover list
    "task_seq": 0,
}
TASK_WIDGET_PREFIXES = ("opt_", "ans_", "pend_", "btn_", "sub_", "demo_fb")


def fresh(v):
    return v.copy() if isinstance(v, (list, dict)) else v


for k, v in {**TASK_DEFAULTS, **KB_DEFAULTS}.items():
    st.session_state.setdefault(k, fresh(v))
st.session_state.setdefault("project", "Demo Project")
st.session_state.setdefault("page", PAGES[0])
S = st.session_state


def new_task():
    """Start a new task: clear the current task and its widget state; keep the knowledge base."""
    for k, v in TASK_DEFAULTS.items():
        S[k] = fresh(v)
    for k in [k for k in S.keys() if str(k).startswith(TASK_WIDGET_PREFIXES)]:
        del S[k]
    S.page = PAGES[0]


def clear_all():
    new_task()
    for k, v in KB_DEFAULTS.items():
        S[k] = fresh(v)


def goto(page: str):
    S.page = page


def open_project(name):
    S.kb_open = name
    S.page = "Project Knowledge Base"


def fill_example(name: str):
    """Fill in an example: if there's already a task in hand, archive it first; the example goes under its own project."""
    if S.task_id is not None:
        new_task()
    sc = SCENARIOS[name]
    S.task = sc["task"]
    S.project = sc["project"]


def scenario_of(task: str):
    return next((v for v in SCENARIOS.values() if v["task"] == task), None)


def scenario_by_project(proj: str):
    return next((v for v in SCENARIOS.values() if v["project"] == proj), None)


# ---------- Knowledge capture ----------


def log(kind: str, text: str):
    S.events.append({"time": datetime.now(), "project": S.task_project, "task": S.task_title, "type": kind, "text": text})


def save_doc(key: str, kind: str, title: str, content: str):
    """Capture a record; saving again with the same key overwrites the old version (e.g. brief regenerated, question answered)."""
    doc = {"key": key, "project": S.task_project, "task_id": S.task_id, "task": S.task_title, "type": kind,
           "title": title, "content": content, "time": datetime.now()}
    for d in S.docs:
        if d["key"] == key:
            d.update(doc)
            return
    S.docs.append({"id": len(S.docs) + 1, **doc})


def ask_key(q: str) -> str:
    """In-progress questions are keyed by question text: an "awaiting" record is saved when forwarded, then updated in place once answered."""
    return f"{S.task_id}-ask-{hashlib.md5(q.encode('utf-8')).hexdigest()[:8]}"


def cast() -> list:
    """The current project's stakeholders: {"name", "role", "cares", "owner"}; the owner settles conflicts and can stop the questioning."""
    info = S.projects.get(S.task_project) or {}
    return info.get("cast") or [{"name": "Requester", "role": "Requester", "cares": "", "owner": True}]


def owner() -> str:
    people = cast()
    return next((p["name"] for p in people if p["owner"]), people[0]["name"])


def cast_md(people: list) -> str:
    return "\n".join(f"- {p['name']} ({p['role']}{', owner' if p['owner'] else ''}): {p['cares'] or 'n/a'}" for p in people)


def route(name: str) -> str:
    """Map an AI-chosen addressee onto the cast; unknown or empty names go to the owner."""
    names = {p["name"].strip().lower(): p["name"] for p in cast()}
    return names.get((name or "").strip().lower()) or owner()


def act_as(name: str):
    S.acting = name


def clean_cast(rows: list, fallback: list) -> list:
    """Normalize the edited stakeholder table: drop nameless rows, blank out NaN, keep exactly one owner."""
    people = []
    for r in rows:
        name = r.get("name")
        if pd.isna(name) or not str(name).strip():
            continue
        people.append({
            "name": str(name).strip(),
            "role": "" if pd.isna(r.get("role")) else str(r.get("role")),
            "cares": "" if pd.isna(r.get("cares")) else str(r.get("cares")),
            "owner": bool(r.get("owner")) if not pd.isna(r.get("owner")) else False,
        })
    if not people:
        return fallback
    first = next((i for i, x in enumerate(people) if x["owner"]), 0)
    for i, x in enumerate(people):
        x["owner"] = i == first
    return people


def apply_clarify(a: dict, r: dict):
    """Open the next round from the AI's clarify result (or finish if it has nothing left)."""
    a["understanding"], S.remaining = r["understanding"], r["remaining"]
    a["questions"], S.conflicts = r["questions"], r["conflicts"]
    S.round += 1
    S.round_ans, S.round_dec = {}, {}
    S.clarified = not (r["questions"] or r["conflicts"])
    if S.conflicts:
        S.acting = owner()
    elif r["questions"]:
        S.acting = route(r["questions"][0]["to"])


def close_round(a: dict, stop: bool = False):
    """Commit this round's answers and owner decisions, then let the AI open the next round (or stop)."""
    qs, cs = a["questions"], S.conflicts
    answered = [{"q": q["question"], **S.round_ans[q["question"]], "round": S.round} for q in qs if q["question"] in S.round_ans]
    decided = [{"q": f"Conflict: {c['topic']}", "a": S.round_dec[c["topic"]], "by": owner(), "round": S.round, "kind": "decision",
                "sides": "; ".join(f"{x['who']} wants {x['wants']}" for x in c["sides"])}
               for c in cs if c["topic"] in S.round_dec]
    S.qa += answered + decided
    for d in decided:
        save_doc(f"{S.task_id}-dec-{hashlib.md5(d['q'].encode('utf-8')).hexdigest()[:8]}", "Q&A", d["q"],
                 f"Owner {d['by']} decided: {d['a']} (sides: {d['sides']})")
        log("Conflict settled", f"{d['q']} → {d['a']}")
    S.remaining += [f"{q['question']} (unanswered, for {route(q['to'])})" for q in qs if q["question"] not in S.round_ans]
    S.remaining += [f"Conflict: {c['topic']} (undecided)" for c in cs if c["topic"] not in S.round_dec]
    if stop or S.round >= MAX_ROUNDS:
        S.clarified = S.stopped = True
        S.round_ans, S.round_dec = {}, {}
        return
    try:
        r = clarify(S.task, a, cast(), S.qa, S.round + 1)
    except Exception:
        S.qa = S.qa[: len(S.qa) - len(answered) - len(decided)]  # Keep the round open so it can be resubmitted
        raise
    log("Clarification round", f"Round {S.round}: {len(answered)} answer(s), {len(decided)} decision(s)")
    apply_clarify(a, r)


def round_waiting(a: dict) -> list:
    """Who this round still waits for."""
    who = [route(q["to"]) for q in a["questions"] if q["question"] not in S.round_ans]
    who += [owner() for c in S.conflicts if c["topic"] not in S.round_dec]
    return sorted(set(who))


def add_wish(a: dict, who: str, text: str):
    """Record a stakeholder's unprompted wish. Mid-round it waits for the next round; once clarification
    has finished, pi reacts right away and may reopen questioning with follow-ups."""
    S.qa.append({"q": f"Wish from {who}", "a": text, "by": who, "round": S.round, "kind": "wish"})
    S.wish_ver += 1
    save_doc(f"{S.task_id}-wish-{S.wish_ver}", "Q&A", f"Wish from {who}", f"Wish ({who}): {text}")
    log("Wish added", f"{who}: {text}")
    if not S.clarified:
        st.toast("Wish recorded; the AI takes it into account in the next round")
        return
    apply_clarify(a, clarify(S.task, a, cast(), S.qa, S.round + 1))
    if not S.clarified:
        S.stopped = False
        st.toast("The new wish raised a conflict or follow-up questions")


def on_analyzed(a: dict):
    if S.task_id is None:
        S.task_seq += 1
        S.task_id = S.task_seq
        S.task_project = S.project.strip() or "Untitled Project"
    S.task_title = a["title"] or S.task[:40]
    info = S.projects.setdefault(S.task_project, {"created": datetime.now(), "deadline": None})
    if not info.get("cast"):  # A project keeps its cast across tasks; only a new project takes the AI's suggestion
        people = [x for x in a["stakeholders"] if x.get("name")][:5]
        info["cast"] = [{**x, "owner": i == 0} for i, x in enumerate(people)] or cast()
    S.cast_seed = [dict(x) for x in info["cast"]]
    S.acting = owner()
    try:
        ddl = date.fromisoformat(a["deadline"])
        if not info["deadline"] or ddl > info["deadline"]:
            info["deadline"] = ddl
    except ValueError:
        pass
    known = "; ".join(a["known"]) or "none"
    save_doc(f"{S.task_id}-task", "Task", S.task_title,
             f"{S.task}\n\nAI's understanding: {a['understanding']}\n\nAlready clear: {known}\n\nDeadline: {a['deadline'] or 'not mentioned'}")
    log("Task posted", a["understanding"])


def brief_md(b: dict) -> str:
    parts = [
        f"**Goal**: {b['goal']}",
        "**In scope**: " + "; ".join(b["in_scope"]),
        "**Out of scope**: " + ("; ".join(b["out_of_scope"]) or "none"),
        "**Acceptance criteria**: " + "; ".join(b["acceptance_criteria"]),
        "**Subtasks**: " + "; ".join(f"{t['title']} ({t['detail']})" for t in b["subtasks"]),
        "**Confirmed decisions**: " + ("; ".join(f"{d['decision']} (source: {d['source']})" for d in b["decisions"]) or "none"),
        "**Open issues**: " + ("; ".join(b["open_issues"]) or "none"),
    ]
    return "\n\n".join(parts)


def project_docs(proj: str) -> list:
    return [d for d in S.docs if d["project"] == proj]


def project_doc_md(proj: str) -> str:
    """Combine all of a project's records into one document: one section per task, each with requirement, brief, demo and all Q&A."""
    info = S.projects[proj]
    docs = project_docs(proj)
    meta = f"Created {info['created']:%Y-%m-%d %H:%M} · Deadline {info['deadline'] or 'not set'}"
    if docs:
        meta += f" · Last updated {max(d['time'] for d in docs):%m-%d %H:%M}"
    lines = [f"### {proj}", meta]
    for n, tid in enumerate(dict.fromkeys(d.get("task_id") for d in docs), 1):
        group = [d for d in docs if d.get("task_id") == tid]
        main = {d["type"]: d for d in group if d["type"] in ("Task", "Brief", "Demo")}
        lines.append(f"#### Task {n}: {group[-1]['task']}")
        if "Task" in main:
            lines += ["##### Requirement & understanding", main["Task"]["content"]]
        if "Brief" in main:
            lines += ["##### Task brief", main["Brief"]["content"]]
        if "Demo" in main:
            lines += ["##### Requirement demo", f"Generated ({main['Demo']['time']:%m-%d %H:%M}); preview or download it under \"Requirement demo\" on this page."]
        qas = [d for d in group if d["type"] == "Q&A"]
        if qas:
            lines.append(f"##### Q&A log ({len(qas)})")
            lines += [f"**Q: {d['title']}**  \n{d['content']}  \n<small>{d['time']:%m-%d %H:%M}</small>" for d in qas]
    return "\n\n".join(lines)


def on_check(title: str, key: str):
    log("Subtask done" if S[key] else "Subtask reopened", title)


# ---------- Rendering helpers ----------

CSS = """
<style>
#MainMenu, footer, [data-testid="stAppDeployButton"] {visibility: hidden;}
.block-container {padding-top: 2.2rem; max-width: 1240px;}
.subtitle {color: rgba(128,128,128,.95); font-size: 1.02rem; margin: -0.6rem 0 0.8rem;}
.steps {display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 0.2rem 0 1.4rem;}
.step {display: flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 999px;
       font-size: 0.88rem; border: 1px solid rgba(128,128,128,.35); color: rgba(128,128,128,.95);}
.step span {display: inline-flex; width: 20px; height: 20px; border-radius: 50%; align-items: center;
            justify-content: center; font-size: 0.72rem; background: rgba(128,128,128,.18);}
.step.done {background: #4f46e5; border-color: #4f46e5; color: #fff;}
.step.done span {background: rgba(255,255,255,.25);}
.step.now {border-color: #4f46e5; color: #4f46e5; font-weight: 600;}
.bar {flex: 0 0 16px; height: 1px; background: rgba(128,128,128,.35);}
.pill {display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: 0.75rem;
       background: rgba(79,70,229,.12); color: #6366f1; margin-left: 6px; vertical-align: middle;}
.label {font-size: 0.8rem; letter-spacing: .02em; color: rgba(128,128,128,.95); margin-bottom: .3rem;}
.cover {border: 1px solid rgba(128,128,128,.28); border-radius: 12px; overflow: hidden; margin-bottom: .4rem;}
.cover-band {height: 76px; display: flex; align-items: flex-end; justify-content: space-between; padding: 10px 14px;}
.cover-band b {font-size: 1.7rem; line-height: 1; color: rgba(255,255,255,.92);}
.cover-band i {font-style: normal; font-size: .72rem; color: rgba(255,255,255,.85);}
.cover-body {padding: 10px 14px 12px;}
.cover-title {font-weight: 600; font-size: 1.02rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.cover-sub {font-size: .8rem; color: rgba(128,128,128,.95); margin: 2px 0 6px; white-space: nowrap;
            overflow: hidden; text-overflow: ellipsis;}
.cover-body .pill {margin: 0 4px 0 0;}
</style>
"""


def steps_done() -> int:
    cur = [e for e in S.events if S.task_title and e["task"] == S.task_title]
    if any(e["type"] in ("AI answered", "Stakeholder answered") for e in cur):
        return 6
    if S.chat:
        return 5
    if S.demo:
        return 4
    if S.brief:
        return 3
    if S.analysis:
        return 1
    return 0


def render_steps():
    done, parts = steps_done(), []
    for i, s in enumerate(STEPS):
        cls = "done" if i < done else ("now" if i == done else "todo")
        parts.append(f'<div class="step {cls}"><span>{i + 1}</span>{s}</div>')
    st.markdown('<div class="steps">' + '<div class="bar"></div>'.join(parts) + "</div>", unsafe_allow_html=True)


def label(text: str):
    st.markdown(f'<div class="label">{text}</div>', unsafe_allow_html=True)


def bullets(items):
    for i in items:
        st.markdown(f"- {i}")


def render_brief(b: dict, checkable: bool = False):
    st.markdown(f"**Goal**: {b['goal']}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**In scope**")
        bullets(b["in_scope"])
    with c2:
        st.markdown("**Out of scope**")
        bullets(b["out_of_scope"] or ["(none)"])
    st.markdown("**Acceptance criteria**")
    bullets(b["acceptance_criteria"])
    st.markdown("**Subtasks**")
    for i, t in enumerate(b["subtasks"]):
        if checkable:
            st.checkbox(f"{t['title']}: {t['detail']}", key=f"sub_{i}", on_change=on_check, args=(t["title"], f"sub_{i}"))
        else:
            st.markdown(f"{i + 1}. **{t['title']}**: {t['detail']}")
    if b["decisions"]:
        st.markdown("**Confirmed decisions**")
        for d in b["decisions"]:
            st.markdown(f"- {d['decision']}  \n  <small>Source: {d['source']}</small>", unsafe_allow_html=True)
    if b["open_issues"]:
        st.markdown("**Open issues**")
        bullets(b["open_issues"])


def show_demo(code: str, height: int = 540):
    # The demo is model-generated HTML: embed it as a data URL so the iframe has its own origin and can't touch the app
    src = "data:text/html;charset=utf-8;base64," + base64.b64encode(code.encode("utf-8")).decode("ascii")
    if hasattr(st, "iframe"):
        st.iframe(src, height=height)
    else:  # Older Streamlit versions
        components.iframe(src, height=height, scrolling=True)


def preset_buttons(questions: list, prefix: str):
    """Render preset questions as a row of buttons and return the one clicked."""
    picked = None
    if questions:
        cols = st.columns(len(questions))
        for i, q in enumerate(questions):
            if cols[i].button(q, key=f"{prefix}_{i}", use_container_width=True):
                picked = q
    return picked


def cover_html(proj: str) -> str:
    docs = project_docs(proj)
    info = S.projects[proj]
    n_tasks = len(dict.fromkeys(d.get("task_id") for d in docs))
    qas = [d for d in docs if d["type"] == "Q&A"]
    waiting = sum(d["content"].startswith(WAITING) for d in qas)
    latest = max(docs, key=lambda d: d["time"]) if docs else None
    color = COVER_COLORS[int(hashlib.md5(proj.encode("utf-8")).hexdigest(), 16) % len(COVER_COLORS)]
    ddl = f"Due {info['deadline']:%m-%d}" if info["deadline"] else "No deadline"
    tags = [f'<span class="pill">{n_tasks} task(s)</span>', f'<span class="pill">{len(qas)} Q&amp;A</span>']
    if any(d["type"] == "Demo" for d in docs):
        tags.append('<span class="pill">Has demo</span>')
    if waiting:
        tags.append(f'<span class="pill">{WAITING} {waiting}</span>')
    sub = f"Latest: {latest['task']}" if latest else "No records yet"
    return (
        f'<div class="cover"><div class="cover-band" style="background:{color}">'
        f'<b>{htmllib.escape(proj[:1])}</b><i>{ddl}</i></div>'
        f'<div class="cover-body"><div class="cover-title">{htmllib.escape(proj)}</div>'
        f'<div class="cover-sub">{htmllib.escape(sub)}</div>{"".join(tags)}</div></div>'
    )


# ---------- Stakeholders page: one step at a time ----------


def stage() -> str:
    """The step shown in full; earlier steps collapse into one-line summaries. A wish that reopens questioning jumps back."""
    if not S.analysis:
        return "task"
    if not S.clarified or not S.brief:
        return "questions"
    return "demo" if S.demo else "brief"


def gen_brief(a: dict) -> bool:
    with st.spinner("Turning it into a brief you can start on…"):
        try:
            S.brief, S.demo = write_brief(S.task, a, cast(), S.qa, S.remaining), None
            S.history, S.brief_qa = list(S.qa), len(S.qa)
            save_doc(f"{S.task_id}-brief", "Brief", f"{S.task_title} · Task brief", brief_md(S.brief))
            for i, h in enumerate(filter(is_question, S.qa)):
                save_doc(f"{S.task_id}-clar-{i}", "Q&A", h["q"], f"A ({h['by']}, pre-work clarification round {h['round']}): {h['a']}")
            log("Brief generated", S.brief["goal"])
            return True
        except Exception as e:
            st.error(f"Call failed: {e}")
            return False


def build_demo(feedback: str = "") -> bool:
    with st.spinner("pi is building the prototype in its workspace…"):
        try:
            S.demo, S.demo_steps = make_demo(S.task, S.brief, S.history, feedback)
            S.demo_ver += 1
            save_doc(f"{S.task_id}-demo", "Demo", f"{S.task_title} · Requirement demo", S.demo)
            if feedback:
                S.history.append({"q": "Change requests for the requirement demo", "a": feedback, "by": S.acting or owner()})
                save_doc(f"{S.task_id}-demofb-{S.demo_ver}", "Q&A", "Change requests for the requirement demo",
                         f"A ({S.acting or owner()}, after seeing the demo): {feedback}")
                log("Demo revised", feedback)
            else:
                log("Demo generated", "Generated an interactive prototype from the brief")
            return True
        except Exception as e:
            st.error(f"Call failed: {e}")
            return False


def task_step():
    c1, c2 = st.columns([4, 1])
    pick = c1.selectbox("Example scenario", ["Write my own"] + list(SCENARIOS), key="scenario_pick", label_visibility="collapsed")
    c2.button("Fill in example", disabled=pick not in SCENARIOS, use_container_width=True,
              on_click=fill_example, args=(pick,))
    S.task = st.text_area("What do you need? Plain words are fine.", value=S.task, height=110)
    if st.button("Let AI understand the task", type="primary", disabled=not S.task.strip()):
        with st.spinner("AI is breaking down the task and looking for ambiguities…"):
            try:
                S.analysis, S.brief, S.demo = analyze(S.task), None, None
                S.qa, S.round, S.remaining, S.stopped = [], 1, [], False
                S.round_ans, S.conflicts, S.round_dec = {}, [], {}
                S.clarified = not S.analysis["questions"]
                on_analyzed(S.analysis)
                ok = True
            except Exception as e:
                st.error(f"Call failed: {e}")
                ok = False
        if ok:
            st.rerun()  # Kept outside the try: st.rerun works by raising, so it must not be swallowed by except


def task_summary(a: dict):
    with st.expander(f"📝 Task: {S.task_title} · {len(cast())} stakeholders"):
        st.markdown(f"> {S.task}")
        c1, c2 = st.columns(2)
        with c1:
            label("Initial breakdown")
            for i, t in enumerate(a["subtasks"], 1):
                st.markdown(f"{i}. **{t['title']}**: {t['detail']}")
        with c2:
            label("Already clear")
            bullets(a["known"] or ["(nothing yet)"])
        if a["needs_meeting"]:
            st.warning(f"Suggest talking face to face: {a['meeting_reason']}")
        label("Stakeholders · edit freely; tick the owner")
        edited = st.data_editor(
            pd.DataFrame(S.cast_seed or cast(), columns=["name", "role", "cares", "owner"]),
            key=f"sub_cast_{S.task_id}", num_rows="dynamic", hide_index=True, use_container_width=True,
            column_config={
                "name": "Name",
                "role": "Role",
                "cares": st.column_config.TextColumn("Cares about", width="large"),
                "owner": st.column_config.CheckboxColumn("Owner"),
            },
        )
        S.projects[S.task_project]["cast"] = clean_cast(edited.to_dict("records"), cast())


def qa_summary():
    if not S.qa:
        return
    nq, nw = sum(map(is_question, S.qa)), sum(map(is_wish, S.qa))
    with st.expander(f"💬 Q&A: {nq} answer(s) · {nw} wish(es) · {len(S.qa) - nq - nw} conflict(s) settled"):
        n = 0  # Question number; wishes and decisions are not numbered
        for h in S.qa:
            if is_wish(h):
                st.markdown(f"💬 **{h['by']} wishes:** {h['a']}")
            elif h.get("kind") == "decision":
                st.markdown(f"⚖️ **{h['by']} settled** {h['q']}: {h['a']}")
            else:
                n += 1
                st.markdown(f"**Q{n}. {h['q']}**  \n{h['by']}: {h['a']}")


def wish_box(a: dict):
    c1, c2 = st.columns([5, 1])
    wish = c1.text_input("Wish", key=f"sub_wish_{S.wish_ver}", label_visibility="collapsed",
                         placeholder=f"💬 Anything else {S.acting} wants? e.g. It must work on my phone")
    if c2.button("Add wish", use_container_width=True) and wish.strip():
        with st.spinner("AI is taking the new wish into account…"):
            try:
                add_wish(a, S.acting, wish.strip())
                ok = True
            except Exception as e:
                st.error(f"Call failed: {e}")
                ok = False
        if ok:
            st.rerun()


def option_labels(options: list) -> list:
    """Strip the model's "(Recommended: ...)" notes and mark the recommended option (else the first) with a star instead."""
    rec = next((i for i, o in enumerate(options) if re.search(r"(?<!not )recommended", o, re.I)), 0)
    # Only parenthesized notes go, and "(not recommended)" stays: it's a warning worth keeping
    clean = [re.sub(r"\s*[(\[][^)\]]*(?<!not )recommended[^)\]]*[)\]]", "", o, flags=re.I).strip() or o for o in options]
    return [f"★ {o}" if i == rec else o for i, o in enumerate(clean)]


def questions_step(a: dict):
    st.info(a["understanding"])
    qa_summary()
    if S.clarified:
        if S.stopped or S.remaining:
            st.warning("Clarification stopped; these go into the brief as open issues")
            bullets(S.remaining)
        else:
            st.success(f"Request fully explained: {len(S.qa)} answer(s) over {len({h['round'] for h in S.qa})} round(s)")
        if st.button("Write the task brief", type="primary") and gen_brief(a):
            st.rerun()
        return

    me, mine, picked = S.acting, {}, None
    st.markdown(f"#### Round {S.round} · waiting for {', '.join(round_waiting(a))}")
    for i, q in enumerate(a["questions"]):
        to = route(q["to"])
        with st.container(border=True):
            st.markdown(f'**Q{sum(map(is_question, S.qa)) + i + 1}. {q["question"]}** <span class="pill">for {htmllib.escape(to)}</span>',
                        unsafe_allow_html=True)
            st.caption(q["why"])
            done = S.round_ans.get(q["question"])
            if done:
                st.markdown(f"✅ **{done['by']}**: {done['a']}")
            elif to != me:
                st.button(f"Answer as {to} →", key=f"btn_as_{S.round}_{i}", on_click=act_as, args=(to,), type="tertiary")
            else:
                opts = option_labels(q["options"]) + ["Other…"]
                choice = st.pills("Answer", opts, default=next((o for o in opts if o.startswith("★ ")), opts[0]),
                                  key=f"opt_{S.round}_{i}", label_visibility="collapsed")
                if choice == "Other…":
                    text = st.text_input("Your answer", key=f"ans_{S.round}_{i}", label_visibility="collapsed",
                                         placeholder="Your own answer (leave empty for 'not sure')")
                    mine[q["question"]] = text.strip() or "Not sure, decide later"
                elif choice:
                    mine[q["question"]] = choice.removeprefix("★ ")

    for j, c in enumerate(S.conflicts):
        with st.container(border=True):
            st.markdown(f'**⚖️ {c["topic"]}** <span class="pill">{htmllib.escape(owner())} decides</span>', unsafe_allow_html=True)
            st.markdown(" · ".join(f"**{route(x['who'])}** wants {x['wants']}" for x in c["sides"]))
            done = S.round_dec.get(c["topic"])
            if done:
                st.markdown(f"✅ **{owner()}** chose: {done}")
            elif me != owner():
                st.button(f"Answer as {owner()} →", key=f"btn_cas_{S.round}_{j}", on_click=act_as, args=(owner(),), type="tertiary")
            else:
                for k, (o, shown) in enumerate(zip(c["options"], option_labels([o["label"] for o in c["options"]]))):
                    if st.button(shown, key=f"btn_pick_{S.round}_{j}_{k}", help=o["tradeoff"], use_container_width=True):
                        picked = (c["topic"], shown.removeprefix("★ "))
                with st.popover("Other…"):
                    own = st.text_input("Decide in your own words", key=f"ans_c_{S.round}_{j}")
                    if st.button("Decide", key=f"btn_own_{S.round}_{j}") and own.strip():
                        picked = (c["topic"], own.strip())

    c1, c2 = st.columns(2)
    submit = c1.button(f"Submit {me}'s answers", type="primary", use_container_width=True, disabled=not mine)
    stop = me == owner() and c2.button("Enough, write the brief", use_container_width=True,
                                       help="Stop asking; anything still unclear goes into the brief's open issues")
    if submit or stop or picked:
        if submit:
            S.round_ans.update({q: {"a": ans, "by": me} for q, ans in mine.items()})
        if picked:
            S.round_dec[picked[0]] = picked[1]
        waiting = round_waiting(a)
        if waiting and not stop:
            st.toast(f"Saved. Still waiting for {', '.join(waiting)}")
            st.rerun()
        with st.spinner("AI is checking what the answers still leave open…"):
            try:
                close_round(a, stop)
                ok = True
            except Exception as e:
                st.error(f"Call failed: {e}")
                ok = False
        if ok:
            st.rerun()


def brief_step(a: dict):
    b = S.brief
    if len(S.qa) > S.brief_qa:
        c1, c2 = st.columns([4, 1])
        c1.warning("New answers or wishes since this brief was written")
        if c2.button("Rewrite brief", use_container_width=True) and gen_brief(a):
            st.rerun()
    nq, nw = sum(map(is_question, S.qa)), sum(map(is_wish, S.qa))
    st.markdown(" ".join(f'<span class="pill">{t}</span>' for t in [
        f"{nq} questions", f"{nw} wishes", f"{len(S.qa) - nq - nw} conflicts settled",
        f"{len(b['decisions'])} decisions", f"{len(b['open_issues'])} still open"]), unsafe_allow_html=True)
    left, right = st.columns([1, 2])
    with left, st.container(border=True):
        label("Before: the one-liner")
        st.markdown(f"> {S.task}")
    with right, st.container(border=True):
        label("After: a brief the worker can start on")
        render_brief(b)
    if st.button("Build clickable demo", type="primary") and build_demo():
        st.rerun()


def demo_step(a: dict):
    with st.expander("📋 Task brief"):
        render_brief(S.brief)
    with st.container(border=True):
        show_demo(S.demo)
    if S.demo_steps:
        st.caption(f"pi worked in workspaces/{workspace_name()}/: {steps_md(S.demo_steps)}")
    c1, c2, c3 = st.columns([3, 1, 1])
    feedback = c1.text_input("What's wrong?", key=f"demo_fb_{S.demo_ver}", label_visibility="collapsed",
                             placeholder=f"What should change, {S.acting}? e.g. bigger buttons, show history")
    if c2.button("Redo with feedback", disabled=not feedback.strip(), use_container_width=True) and build_demo(feedback.strip()):
        st.rerun()
    c3.download_button("Download HTML", S.demo, file_name=f"{S.task_title or 'demo'}.html", mime="text/html",
                       use_container_width=True)
    st.button("Switch to worker view →", type="primary", on_click=goto, args=("Worker",))


def pending_block():
    """Worker questions forwarded to a stakeholder, shown first because someone is blocked on them."""
    for i, q in enumerate(list(S.pending)):
        to = route(S.pending_to.get(q, ""))
        with st.container(border=True):
            st.markdown(f'📨 **Worker asks:** {q} <span class="pill">for {htmllib.escape(to)}</span>', unsafe_allow_html=True)
            if to != S.acting:
                st.button(f"Answer as {to} →", key=f"btn_pas_{i}_{q}", on_click=act_as, args=(to,), type="tertiary")
                continue
            c1, c2 = st.columns([5, 1])
            reply = c1.text_input("Your answer", key=f"pend_{i}_{q}", label_visibility="collapsed", placeholder="Your answer")
            if c2.button("Answer", key=f"btn_{i}_{q}", use_container_width=True) and reply:
                S.history.append({"q": q, "a": reply, "by": to})
                save_doc(ask_key(q), "Q&A", q, f"A ({to}, asked during work): {reply}")
                log("Stakeholder answered", f"{q} → {to}: {reply}")
                S.chat.append({"role": "assistant", "content": f"{to} answered: **{q}**\n\n{reply}"})
                S.pending.remove(q)
                st.rerun()


# ---------- Pages ----------

st.set_page_config(page_title="Requirement Clarifier", page_icon="🧭", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.radio("View", PAGES, key="page")
    st.text_input("Project", key="project", disabled=S.task_id is not None,
                  help="New tasks go under this project; you can switch after starting a new task")
    if S.page == "Stakeholders" and S.analysis:
        names = [x["name"] for x in cast()]
        S.acting = st.selectbox("Acting as", names, index=names.index(S.acting) if S.acting in names else 0,
                                help=f"Answer as another stakeholder. The owner ({owner()}) settles conflicts.")
    if S.pending and S.page != "Stakeholders":
        st.warning(f"{len(S.pending)} worker question(s) waiting for stakeholders")
    c1, c2 = st.columns(2)
    c1.button("New task", on_click=new_task, use_container_width=True, help="Archive the current task; the knowledge base is kept")
    c2.button("Clear all", on_click=clear_all, use_container_width=True)
    if S.get("replay"):
        st.caption("📼 Offline demo mode is on")
    with st.expander("⚙️ Settings"):
        st.radio("AI engine", ["pi", "direct"], key="engine", horizontal=True,
                 help="pi: every call runs as a pi agent job. direct: plain Qwen API calls. pi falls back to direct on failure.")
        if S.engine == "pi":
            try:
                st.selectbox("Model", start_harness()["models"], key="pi_model")
            except Exception as e:
                st.caption(f"pi harness unavailable: {e}")
        st.toggle("Offline demo mode", key="replay",
                  help="Prefer recorded answers, no network needed. Normal calls are recorded automatically; failed API calls also fall back to recordings.")
        n_qa = sum(d["type"] == "Q&A" for d in S.docs)
        st.caption(f"{len(get_cache())} AI answer(s) recorded · knowledge base: {len(S.projects)} project(s), {n_qa} Q&A")

st.title("🧭 Requirement Clarifier")
st.markdown('<div class="subtitle">A rough idea → AI grills every stakeholder → brief + clickable demo → answers while you work</div>',
            unsafe_allow_html=True)
render_steps()

page = S.page

if page == "Stakeholders":
    pending_block()
    step = stage()
    if step == "task":
        task_step()
    else:
        a = S.analysis
        task_summary(a)
        if step == "questions":
            questions_step(a)
        else:
            qa_summary()
            (brief_step if step == "brief" else demo_step)(a)
        wish_box(a)

elif page == "Worker":
    if not S.brief:
        st.info("No task brief yet. Post a task in the \"Stakeholders\" view first.")
        st.button("Go post a task →", on_click=goto, args=("Stakeholders",))
        st.stop()

    left, right = st.columns([1, 1])
    with left:
        with st.container(border=True):
            label("My task")
            render_brief(S.brief, checkable=True)
        if S.demo:
            with st.expander("Requirement demo (as confirmed by the requester)"):
                show_demo(S.demo, height=500)
    with right:
        st.markdown("#### Ask whenever you have a question while working")
        st.caption("If the brief has the answer, it's answered directly with its basis; if not, it's forwarded to the stakeholder who owns that decision. No guessing.")
        for m in S.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"], unsafe_allow_html=True)
        sc = scenario_of(S.task)
        preset = preset_buttons(sc["worker_qs"] if sc else [], "wq")
        if q := (st.chat_input("e.g. Does it need keyboard input?") or preset):
            S.chat.append({"role": "user", "content": q})
            try:
                with st.spinner("Checking the brief and past Q&A…"):
                    r = answer_worker(S.task, S.brief, cast(), S.history, q)
                if r["can_answer"]:
                    msg = f"{r['answer']}\n\n<small>Basis: {r['source']}</small>"
                    save_doc(ask_key(q), "Q&A", q, f"A (AI from the brief, basis: {r['source']}): {r['answer']}")
                    log("AI answered", f"{q} → {r['answer']}")
                else:
                    rq, who = r["question_for_requester"], route(r["ask_who"])
                    S.pending.append(rq)
                    S.pending_to[rq] = who
                    save_doc(ask_key(rq), "Q&A", rq, f"{WAITING} (worker asked: {q}; forwarded to {who})")
                    log("Forwarded to stakeholder", f"{who}: {rq}")
                    msg = f"Not enough basis in the brief: {r['answer']}\n\n↪ Forwarded to {who}: **{rq}**"
            except Exception as e:
                msg = f"Call failed: {e}"
            S.chat.append({"role": "assistant", "content": msg})
            st.rerun()
        if S.chat:
            st.button("See what the knowledge base captured →", on_click=open_project, args=(S.task_project,))

else:  # Project Knowledge Base
    if not S.projects:
        st.info("The knowledge base is empty. After you post a task and generate a brief in the \"Stakeholders\" view, the process is captured here automatically.")
        st.button("Go post a task →", on_click=goto, args=("Stakeholders",))
        st.stop()

    # ---- Cover list ----
    if S.kb_open not in S.projects:
        gallery, chat = st.columns([2, 1], gap="large")
        with gallery:
            st.subheader("All projects")
            st.caption("One document per project: requirements, briefs, demos and every question are inside, updated automatically as work goes on. Open one for details.")
            names = sorted(S.projects, key=lambda p: S.projects[p]["created"], reverse=True)
            per_row = 3
            for start in range(0, len(names), per_row):
                cols = st.columns(per_row)
                for col, name in zip(cols, names[start:start + per_row]):
                    with col:
                        st.markdown(cover_html(name), unsafe_allow_html=True)
                        st.button("View details", key=f"open_{name}", on_click=open_project, args=(name,),
                                  use_container_width=True)

        with chat:
            st.subheader("Ask all projects")
            st.caption("Cross-project search: who decided what, where things are stuck, which project is due soon.")
            msgs = [m for m in S.kb_chat if m["project"] == ALL_KB]
            with st.container(height=420):
                if not msgs:
                    st.caption("No questions yet; try one below.")
                for i, m in enumerate(msgs):
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"], unsafe_allow_html=True)
                        for p in m.get("projects", []):
                            if p in S.projects:
                                st.button(f"Open \"{p}\"", key=f"all_open_{i}_{p}", on_click=open_project, args=(p,))
            preset = None
            for i, pq in enumerate(ALL_QS):
                if st.button(pq, key=f"aq_{i}", use_container_width=True):
                    preset = pq
            if q := (st.chat_input("Ask all projects, e.g. whose requirements are still undecided?") or preset):
                S.kb_chat.append({"project": ALL_KB, "role": "user", "content": q})
                projects = []
                try:
                    with st.spinner("Searching all projects…"):
                        r = answer_all(q)
                    cites = "; ".join(str(s) for s in r["sources"] if s)
                    msg = r["answer"] if r["found"] else f"Not enough basis in any project: {r['answer']}"
                    if cites:
                        msg += f"\n\n<small>Sources: {cites}</small>"
                    projects = [p for p in dict.fromkeys(r["projects"]) if p in S.projects]
                except Exception as e:
                    msg = f"Call failed: {e}"
                S.kb_chat.append({"project": ALL_KB, "role": "assistant", "content": msg, "projects": projects})
                st.rerun()
        st.stop()

    # ---- Project details ----
    proj = S.kb_open
    info = S.projects[proj]
    docs = project_docs(proj)
    events = [e for e in S.events if e["project"] == proj]
    tasks = list(dict.fromkeys(e["task"] for e in events))
    doc_md = project_doc_md(proj)

    st.button("← All projects", on_click=open_project, args=(None,))
    st.header(proj)
    c = st.columns(5)
    c[0].metric("Created", f"{info['created']:%m-%d %H:%M}")
    c[1].metric("Last updated", f"{events[-1]['time']:%m-%d %H:%M}" if events else "—")
    c[2].metric("Tasks", len(tasks))
    qas = [d for d in docs if d["type"] == "Q&A"]
    waiting = sum(d["content"].startswith(WAITING) for d in qas)
    c[3].metric("Q&A records", len(qas), delta=f"{waiting} awaiting answer" if waiting else None, delta_color="off")
    c[4].metric("Until deadline", f"{(info['deadline'] - date.today()).days} days" if info["deadline"] else "Not set")
    with st.expander("Change deadline"):
        info["deadline"] = st.date_input("Deadline (inferred by AI from the task description; can be edited)", value=info["deadline"], key=f"ddl_{proj}")

    demos = [d for d in docs if d["type"] == "Demo"]
    if demos:
        st.subheader("Requirement demo")
        for d in demos:
            with st.expander(f"Task: {d['task']} · {d['time']:%m-%d %H:%M}", expanded=len(demos) == 1):
                show_demo(d["content"], height=500)

    st.subheader("Project document")
    c1, c2 = st.columns([3, 1])
    c1.caption("Requirements, briefs and every question are in this one document, updated automatically as work goes on.")
    c2.download_button("Download Markdown", doc_md, file_name=f"{proj}.md", mime="text/markdown", use_container_width=True)
    with st.container(border=True):
        st.markdown("\n\n".join(doc_md.split("\n\n")[2:]), unsafe_allow_html=True)  # Title and overview are already shown above

    st.subheader("Ask the knowledge base")
    st.caption("When someone new takes over or you come back a week later, just ask instead of digging through chat history.")
    for m in S.kb_chat:
        if m["project"] == proj:
            with st.chat_message(m["role"]):
                st.markdown(m["content"], unsafe_allow_html=True)
    sc = scenario_by_project(proj)
    preset = preset_buttons(sc["kb_qs"] if sc else [], "kq")
    if q := (st.chat_input("e.g. Which features were finally decided? Which questions are still open?") or preset):
        S.kb_chat.append({"project": proj, "role": "user", "content": q})
        try:
            with st.spinner("Searching…"):
                r = answer_kb(proj, info, doc_md, events, q)
            cites = "; ".join(str(s) for s in r["sources"] if s)
            msg = r["answer"] if r["found"] else f"Not enough basis in the knowledge base: {r['answer']}"
            if cites:
                msg += f"\n\n<small>Sources: {cites}</small>"
        except Exception as e:
            msg = f"Call failed: {e}"
        S.kb_chat.append({"project": proj, "role": "assistant", "content": msg})
        st.rerun()

    st.subheader("Project timeline")
    if events:
        df = pd.DataFrame(events)
        df["Time"] = df["time"].dt.strftime("%m-%d %H:%M:%S")
        span = df.groupby("task", sort=False)["time"].agg(["min", "max"]).reset_index()
        y = alt.Y("task:N", title=None, sort=tasks)
        x_scale = alt.Scale(type="utc")  # Times have no time zone; show them as-is so the browser doesn't convert them again
        spans = alt.Chart(span).mark_rule(strokeWidth=8, opacity=0.2).encode(
            x=alt.X("min:T", scale=x_scale, title=None), x2="max:T", y=y
        )
        dots = alt.Chart(df).mark_circle(size=160, opacity=0.9).encode(
            x=alt.X("time:T", scale=x_scale),
            y=y,
            color=alt.Color("type:N", title="Event"),
            tooltip=[alt.Tooltip("Time:N"), alt.Tooltip("type:N", title="Event"), alt.Tooltip("text:N", title="Details")],
        )
        st.altair_chart((spans + dots).properties(height=80 + 50 * len(tasks)), width="stretch")
        with st.expander("Event details"):
            table = df[["Time", "task", "type", "text"]].rename(columns={"task": "Task", "type": "Event", "text": "Details"})
            st.dataframe(table.iloc[::-1], hide_index=True, width="stretch")
    else:
        st.caption("No events yet")

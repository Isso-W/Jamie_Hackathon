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

PAGES = ["Requester", "Worker", "Project Knowledge Base"]
STEPS = ["Post task", "AI questions", "Task brief", "Requirement demo", "Worker Q&A", "Knowledge captured"]
COVER_COLORS = ["#4f46e5", "#0f766e", "#c2410c", "#be185d", "#1d4ed8", "#7c3aed", "#15803d", "#b45309"]
WAITING = "⏳ Awaiting requester's answer"
KB_TABS = ["📋 Overview", "📄 Document", "🖥️ Demo", "💬 Ask", "🕒 Timeline"]
KB_SORTS = ["Recently updated", "Deadline soonest", "Name"]
ALL_KB = "__all__"  # Marker in kb_chat for cross-project Q&A
ALL_QS = [
    "Which projects still have questions waiting for the requester?",
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
                    "why": {"type": "string", "description": "Why ask this, and what goes wrong if it isn't clarified"},
                    "options": {"type": "array", "items": {"type": "string"}, "description": "Possible answers; leave empty if none"},
                },
                "required": ["question", "why", "options"],
            },
            "description": "This round's most critical clarifying questions, at most 3, ordered by importance",
        },
        "needs_meeting": {"type": "boolean", "description": "Whether it involves multi-party decisions or conflicting interests that need a face-to-face talk"},
        "meeting_reason": {"type": "string"},
    },
    "required": ["title", "understanding", "deadline", "subtasks", "known", "questions", "needs_meeting", "meeting_reason"],
}

CLARIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "understanding": {"type": "string", "description": "Your understanding of the task goal, updated with every answer so far, in one or two sentences"},
        "complete": {"type": "boolean", "description": "True only when no open decision would still change what gets built"},
        "remaining": {"type": "array", "items": {"type": "string"},
                      "description": "Decisions still open that matter; for 'not sure' answers, include your recommended default"},
        "questions": ANALYZE_SCHEMA["properties"]["questions"],
    },
    "required": ["understanding", "complete", "remaining", "questions"],
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
        "question_for_requester": {"type": "string", "description": "If it can't be answered, a clear question to forward to the requester"},
    },
    "required": ["can_answer", "answer", "source", "question_for_requester"],
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


def call_text(system: str, content: str, name: str, max_tokens: int = 8000) -> str:
    """Long text output (e.g. HTML) skips tool use and asks for a plain reply, which is more reliable."""
    def direct():
        resp = get_client().chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            extra_body={"enable_thinking": False},
        )
        return {"text": resp.choices[0].message.content or ""}

    def produce():
        return {"text": via_harness({"system": system, "prompt": content}, direct)["text"]}

    return cached(name, system, content, produce)["text"]


def analyze(task: str) -> dict:
    system = (
        "You are an experienced tech lead on the team, responsible for making sure requirements are understood before work starts. "
        "Break down the task, find ambiguities and hidden assumptions, and ask only the key questions that really affect implementation; don't ask about the obvious. "
        "Don't make assumptions on the requester's behalf: if information is missing, ask. " + LANG_RULE
    )
    return call_tool(system, f"Today is {date.today()}.\nTask description:\n{task}", "task_analysis", ANALYZE_SCHEMA, skills=("grill-me",))


def qa_md(qa: list) -> str:
    return "\n".join(f"Q{i} (round {h.get('round', 1)}): {h['q']}\nA ({h['by']}): {h['a']}" for i, h in enumerate(qa, 1)) or "(none)"


def clarify(task: str, analysis: dict, qa: list, round_no: int) -> dict:
    system = (
        "You are an experienced tech lead grilling the requester, round after round, until the request is fully explained. "
        "You get the task, your current understanding and every answer so far. Answers often open new decisions: ask the follow-ups they raise. "
        "Never repeat a question that was already answered. If an answer was 'not sure', don't ask it again: put it in remaining with your recommended default. "
        "Only ask what changes what gets built. Set complete to true, with no questions, once nothing open would change the implementation. " + LANG_RULE
    )
    content = (
        f"Today is {date.today()}.\nTask description:\n{task}\n\nCurrent understanding: {analysis['understanding']}\n\n"
        f"Q&A so far:\n{qa_md(qa)}\n\nThis is round {round_no} of at most {MAX_ROUNDS}."
    )
    return call_tool(system, content, "clarify_round", CLARIFY_SCHEMA, skills=("grill-me",))


def write_brief(task: str, analysis: dict, qa: list, remaining: list) -> dict:
    system = (
        "Based on the task description and the clarification Q&A, write a task brief the worker can start on right away. "
        "Every decision must trace back to an answer; anything without a basis can only go into open issues. " + LANG_RULE
    )
    content = (
        f"Task description:\n{task}\n\nUnderstanding after clarification: {analysis['understanding']}\n\n"
        f"Clarification Q&A:\n{qa_md(qa)}\n\nStill open: {'; '.join(remaining) or '(nothing)'}"
    )
    return call_tool(system, content, "task_brief", BRIEF_SCHEMA)


def make_demo(task: str, brief: dict, qa: list, feedback: str = "") -> str:
    system = (
        "You are a product prototype designer. Based on the requester's answers in the Q&A and the task brief, build an interactive single-file HTML prototype "
        "so the requester can confirm at a glance that \"this is roughly what it will look like\". "
        "Every answer that changes what the user sees or does must be visible and working in the prototype (e.g. if they chose CSV export, there is a working Export CSV button); "
        "earlier change requests in the Q&A still apply. Don't add features nobody asked for or answered. "
        "At the bottom, add a small collapsed <details> section titled \"Built from your answers\" mapping each visible feature to the answer it came from. "
        "Requirements: implement only the core features within the brief's scope, and buttons and inputs must actually work; use fake data where data is needed; "
        "don't access the network or reference any external files, fonts or CDNs; inline all CSS and JS; "
        "keep the style clean and modern, with UI text in the same language as the task description; keep the overall height within 520px and the width responsive. "
        "Output only one ```html code block, with no explanation."
    )
    content = f"Task description:\n{task}\n\nRequester's Q&A:\n{qa_md(qa)}\n\nTask brief:\n{brief_md(brief)}"
    if feedback:
        content += f"\n\nRequester's change requests after seeing the previous demo (must be reflected): {feedback}"
    text = call_text(system, content, "demo_html")
    m = re.search(r"```(?:html)?\s*(.*?)```", text, re.S)
    code = (m.group(1) if m else text).strip()
    if "<" not in code:
        raise RuntimeError("The model did not return HTML")
    return code


def answer_worker(task: str, brief: dict, history: list, question: str) -> dict:
    hist = "\n".join(f"Q: {h['q']}\nA ({h['by']}): {h['a']}" for h in history) or "(none)"
    system = (
        "You are the task's coordination assistant; the worker asks you questions while working. Answer only from the task brief and past Q&A; don't make things up. "
        "If the basis is insufficient, set can_answer to false and turn it into a clear question for the requester. " + LANG_RULE
    )
    content = f"Task description:\n{task}\n\nTask brief:\n{json.dumps(brief, ensure_ascii=False, indent=2)}\n\nPast Q&A:\n{hist}\n\nWorker's question: {question}"
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
    "brief": None,
    "demo": None,  # HTML of the requirement demo
    "demo_ver": 0,  # +1 on every regeneration, used to clear the change-request input
    "history": [],  # Captured Q&A: {"q", "a", "by"}
    "pending": [],  # Questions waiting for the requester
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
st.session_state.setdefault("kb_tab", KB_TABS[0])
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


def open_project(name, tab: str = None):
    S.kb_open = name
    S.kb_tab = tab or KB_TABS[0]
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


def on_analyzed(a: dict):
    if S.task_id is None:
        S.task_seq += 1
        S.task_id = S.task_seq
        S.task_project = S.project.strip() or "Untitled Project"
    S.task_title = a["title"] or S.task[:40]
    info = S.projects.setdefault(S.task_project, {"created": datetime.now(), "deadline": None})
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


DOC_SECTIONS = ["Requirement & understanding", "Task brief", "Requirement demo", "Q&A log"]


def project_doc_md(proj: str, task_id=None, sections=DOC_SECTIONS) -> str:
    """Combine all of a project's records into one document: one section per task, each with requirement, brief, demo and all Q&A.
    task_id / sections narrow it down for reading in the UI; the AI always gets the full document."""
    info = S.projects[proj]
    docs = project_docs(proj)
    meta = f"Created {info['created']:%Y-%m-%d %H:%M} · Deadline {info['deadline'] or 'not set'}"
    if docs:
        meta += f" · Last updated {max(d['time'] for d in docs):%m-%d %H:%M}"
    lines = [f"### {proj}", meta]
    for n, tid in enumerate(dict.fromkeys(d.get("task_id") for d in docs), 1):
        if task_id is not None and tid != task_id:
            continue
        group = [d for d in docs if d.get("task_id") == tid]
        main = {d["type"]: d for d in group if d["type"] in ("Task", "Brief", "Demo")}
        lines.append(f"#### Task {n}: {group[-1]['task']}")
        if "Task" in main and DOC_SECTIONS[0] in sections:
            lines += [f"##### {DOC_SECTIONS[0]}", main["Task"]["content"]]
        if "Brief" in main and DOC_SECTIONS[1] in sections:
            lines += [f"##### {DOC_SECTIONS[1]}", main["Brief"]["content"]]
        if "Demo" in main and DOC_SECTIONS[2] in sections:
            lines += [f"##### {DOC_SECTIONS[2]}", f"Generated ({main['Demo']['time']:%m-%d %H:%M}); preview or download it in the \"Demo\" tab of this project."]
        qas = [d for d in group if d["type"] == "Q&A"]
        if qas and DOC_SECTIONS[3] in sections:
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
.cover-meter {height: 4px; border-radius: 2px; background: rgba(128,128,128,.18); margin: 8px 0 2px; overflow: hidden;}
.cover-meter div {height: 100%; background: #22c55e;}
.cover-foot {display: flex; justify-content: space-between; white-space: nowrap; font-size: .72rem; color: rgba(128,128,128,.95);}
.pill.warn {background: rgba(245,158,11,.16); color: #d97706;}
.pill.bad {background: rgba(239,68,68,.14); color: #dc2626;}
.pill.ok {background: rgba(34,197,94,.14); color: #16a34a;}
.ddl {padding: 2px 8px; border-radius: 999px; background: rgba(0,0,0,.22);}
.hero {border-radius: 14px; padding: 18px 22px; color: #fff; margin: .4rem 0 1rem; display: flex; gap: 18px; align-items: center;}
.hero-mark {flex: 0 0 56px; height: 56px; border-radius: 12px; background: rgba(255,255,255,.18); display: flex;
            align-items: center; justify-content: center; font-size: 1.8rem; font-weight: 700;}
.hero h2 {margin: 0; padding: 0; color: #fff; font-size: 1.6rem;}
.hero-meta {font-size: .86rem; opacity: .88; margin-top: 2px;}
.hero .ddl {margin-left: auto; font-size: .85rem; white-space: nowrap;}
.task-card-head {display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;}
.task-card-head b {font-size: 1.02rem;}
.task-card-head small {color: rgba(128,128,128,.95); margin-left: auto;}
.qa-row {padding: 6px 0; border-bottom: 1px dashed rgba(128,128,128,.25); font-size: .9rem;}
.qa-row:last-child {border-bottom: none;}
.qa-row small {color: rgba(128,128,128,.95);}
</style>
"""


def steps_done() -> int:
    cur = [e for e in S.events if S.task_title and e["task"] == S.task_title]
    if any(e["type"] in ("AI answered", "Requester answered") for e in cur):
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


def project_color(proj: str) -> str:
    return COVER_COLORS[int(hashlib.md5(proj.encode("utf-8")).hexdigest(), 16) % len(COVER_COLORS)]


def deadline_status(ddl):
    """(label, pill class) for a deadline: overdue and due within 3 days stand out."""
    if not ddl:
        return "No deadline", ""
    days = (ddl - date.today()).days
    if days < 0:
        return f"Overdue {-days}d", "bad"
    if days == 0:
        return "Due today", "bad"
    if days <= 3:
        return f"Due in {days}d", "warn"
    return f"Due {ddl:%m-%d}", "ok"


def project_stats(proj: str) -> dict:
    docs = project_docs(proj)
    events = [e for e in S.events if e["project"] == proj]
    qas = [d for d in docs if d["type"] == "Q&A"]
    done = {}
    for e in events:  # Last state of each checkbox wins
        if e["type"] in ("Subtask done", "Subtask reopened"):
            done[(e["task"], e["text"])] = e["type"] == "Subtask done"
    times = [d["time"] for d in docs] + [e["time"] for e in events]
    return {
        "tasks": len(dict.fromkeys(d.get("task_id") for d in docs)),
        "qas": qas,
        "waiting": [d for d in qas if d["content"].startswith(WAITING)],
        "briefs": sum(d["type"] == "Brief" for d in docs),
        "demos": [d for d in docs if d["type"] == "Demo"],
        "done": sum(done.values()),
        "events": events,
        "updated": max(times) if times else S.projects[proj]["created"],
        "latest": max(docs, key=lambda d: d["time"])["task"] if docs else "",
    }


def cover_html(proj: str) -> str:
    ps = project_stats(proj)
    ddl, ddl_cls = deadline_status(S.projects[proj]["deadline"])
    tags = [f'<span class="pill">{ps["tasks"]} task(s)</span>', f'<span class="pill">{len(ps["qas"])} Q&amp;A</span>']
    if ps["demos"]:
        tags.append('<span class="pill ok">Has demo</span>')
    if ps["waiting"]:
        tags.append(f'<span class="pill warn">{len(ps["waiting"])} awaiting answer</span>')
    # Progress: briefs written out of tasks posted (a task without a brief isn't ready to start)
    pct = round(100 * ps["briefs"] / ps["tasks"]) if ps["tasks"] else 0
    sub = f"Latest: {ps['latest']}" if ps["latest"] else "No records yet"
    return (
        f'<div class="cover"><div class="cover-band" style="background:{project_color(proj)}">'
        f'<b>{htmllib.escape(proj[:1])}</b><i class="ddl">{ddl}</i></div>'
        f'<div class="cover-body"><div class="cover-title">{htmllib.escape(proj)}</div>'
        f'<div class="cover-sub">{htmllib.escape(sub)}</div>{"".join(tags)}'
        f'<div class="cover-meter"><div style="width:{pct}%"></div></div>'
        f'<div class="cover-foot"><span>{ps["briefs"]}/{ps["tasks"]} briefed · {ps["done"]} done</span>'
        f'<span>{ps["updated"]:%m-%d %H:%M}</span></div></div></div>'
    )


def hero_html(proj: str) -> str:
    info = S.projects[proj]
    ddl, _ = deadline_status(info["deadline"])
    ps = project_stats(proj)
    meta = f"Created {info['created']:%Y-%m-%d %H:%M} · Last updated {ps['updated']:%m-%d %H:%M} · {ps['tasks']} task(s)"
    return (
        f'<div class="hero" style="background:{project_color(proj)}"><div class="hero-mark">{htmllib.escape(proj[:1])}</div>'
        f'<div><h2>{htmllib.escape(proj)}</h2><div class="hero-meta">{meta}</div></div><span class="ddl">{ddl}</span></div>'
    )


def segmented(label_: str, options: list, key: str):
    """Tab-like switcher whose selection survives reruns (st.tabs jumps back to the first tab after a chat rerun)."""
    if hasattr(st, "segmented_control"):
        v = st.segmented_control(label_, options, key=key, label_visibility="collapsed")
    else:
        v = st.radio(label_, options, key=key, horizontal=True, label_visibility="collapsed")
    return v or options[0]


# ---------- Pages ----------

st.set_page_config(page_title="Requirement Clarifier", page_icon="🧭", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.radio("View", PAGES, key="page")
    st.text_input("Current project", key="project", disabled=S.task_id is not None,
                  help="New tasks go under this project; you can switch after starting a new task")
    if S.task_id is not None:
        st.caption(f"Current task: {S.task_title}")
    if S.pending:
        st.warning(f"{len(S.pending)} worker question(s) waiting for the requester")
    n_qa = sum(d["type"] == "Q&A" for d in S.docs)
    st.caption(f"Knowledge base: {len(S.projects)} project doc(s), {n_qa} Q&A, {len(S.events)} event(s)")
    c1, c2 = st.columns(2)
    c1.button("New task", on_click=new_task, help="Archive the current task; the knowledge base is kept")
    c2.button("Clear all", on_click=clear_all)
    st.divider()
    st.radio("AI engine", ["pi", "direct"], key="engine", horizontal=True,
             help="pi: every call runs as a pi agent job. direct: plain Qwen API calls. pi falls back to direct on failure.")
    if S.engine == "pi":
        try:
            st.selectbox("Model", start_harness()["models"], key="pi_model")
        except Exception as e:
            st.caption(f"pi harness unavailable: {e}")
    st.toggle("Offline demo mode", key="replay",
              help="Prefer recorded answers, no network needed. Normal calls are recorded automatically; failed API calls also fall back to recordings.")
    st.caption(f"{len(get_cache())} AI answer(s) recorded")

st.title("🧭 Requirement Clarifier")
st.markdown(
    '<div class="subtitle">One vague request → AI asks the key questions → brief + clickable demo → automatic answers while working → each project captured as one document</div>',
    unsafe_allow_html=True,
)
render_steps()

page = S.page

if page == "Requester":
    st.subheader("1. Post a task")
    c1, c2 = st.columns([4, 1])
    pick = c1.selectbox("Example scenario", ["Write my own"] + list(SCENARIOS), key="scenario_pick", label_visibility="collapsed")
    c2.button("Fill in example", disabled=pick not in SCENARIOS, use_container_width=True,
              on_click=fill_example, args=(pick,))
    S.task = st.text_area("Describe what you need in plain language; the more casual the better", value=S.task, height=110)

    if st.button("Let AI understand the task", type="primary", disabled=not S.task.strip()):
        with st.spinner("AI is breaking down the task and looking for ambiguities…"):
            try:
                S.analysis, S.brief, S.demo = analyze(S.task), None, None
                S.qa, S.round, S.remaining, S.stopped = [], 1, [], False
                S.clarified = not S.analysis["questions"]
                on_analyzed(S.analysis)
                ok = True
            except Exception as e:
                st.error(f"Call failed: {e}")
                ok = False
        if ok:
            st.rerun()  # Kept outside the try: st.rerun works by raising, so it must not be swallowed by except

    a = S.analysis
    if a:
        st.subheader("2. AI's understanding")
        st.info(a["understanding"])
        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            label("Initial breakdown")
            for i, t in enumerate(a["subtasks"], 1):
                st.markdown(f"{i}. **{t['title']}**: {t['detail']}")
        with c2, st.container(border=True):
            label("Already clear")
            bullets(a["known"] or ["(nothing yet)"])
        if a["needs_meeting"]:
            st.warning(f"Suggest talking face to face: {a['meeting_reason']}")

        st.markdown(
            f'### 3. AI\'s questions <span class="pill">Round {S.round} · keeps asking until the request is fully explained</span>',
            unsafe_allow_html=True,
        )
        if S.qa:
            with st.expander(f"Answered so far ({len(S.qa)})"):
                for i, h in enumerate(S.qa, 1):
                    st.markdown(f"**Q{i}. {h['q']}**  \n{h['a']}")
        if not S.clarified:
            answers = {}
            for i, q in enumerate(a["questions"]):
                with st.container(border=True):
                    st.markdown(f"**Q{len(S.qa) + i + 1}. {q['question']}**")
                    st.caption(f"Why ask: {q['why']}")
                    choice = None
                    if q["options"]:
                        choice = st.radio("Pick one", q["options"] + ["Other / not sure"], key=f"opt_{S.round}_{i}", horizontal=True,
                                          label_visibility="collapsed")
                    text = st.text_input("Additional notes (optional)", key=f"ans_{S.round}_{i}", placeholder="Additional notes (optional)",
                                         label_visibility="collapsed")
                    if choice and choice != "Other / not sure":
                        answers[q["question"]] = f"{choice}. {text}" if text else choice
                    else:
                        answers[q["question"]] = text or "Not sure, decide later"

            c1, c2 = st.columns(2)
            submit = c1.button("Submit answers", type="primary", use_container_width=True)
            stop = c2.button("Enough, write the brief", use_container_width=True,
                             help="Stop asking; anything still unclear goes into the brief's open issues")
            if submit or stop:
                S.qa += [{"q": q, "a": ans, "by": "requester", "round": S.round} for q, ans in answers.items()]
                ok = True
                if stop or S.round >= MAX_ROUNDS:
                    S.clarified = S.stopped = True
                else:
                    with st.spinner("AI is checking what your answers still leave open…"):
                        try:
                            r = clarify(S.task, a, S.qa, S.round + 1)
                            a["understanding"], a["questions"], S.remaining = r["understanding"], r["questions"], r["remaining"]
                            S.round += 1
                            S.clarified = r["complete"] or not r["questions"]
                            log("Clarification round", f"Round {S.round - 1}: {len(answers)} answer(s)")
                        except Exception as e:
                            S.qa = S.qa[: len(S.qa) - len(answers)]  # Let the requester resubmit the same round
                            st.error(f"Call failed: {e}")
                            ok = False
                if ok:
                    st.rerun()
        else:
            rounds = len({h["round"] for h in S.qa})
            if S.stopped or S.remaining:
                st.warning(f"Clarification stopped after {rounds} round(s); open items go into the brief")
                bullets(S.remaining)
            else:
                st.success(f"Request fully explained: {len(S.qa)} answer(s) over {rounds} round(s)")

            if st.button("Generate task brief", type="primary"):
                with st.spinner("Turning it into a brief you can start on…"):
                    try:
                        S.brief, S.demo = write_brief(S.task, a, S.qa, S.remaining), None
                        S.history = list(S.qa)
                        save_doc(f"{S.task_id}-brief", "Brief", f"{S.task_title} · Task brief", brief_md(S.brief))
                        for i, h in enumerate(S.qa):
                            save_doc(f"{S.task_id}-clar-{i}", "Q&A", h["q"], f"A (requester, pre-work clarification round {h['round']}): {h['a']}")
                        log("Brief generated", S.brief["goal"])
                        ok = True
                    except Exception as e:
                        st.error(f"Call failed: {e}")
                        ok = False
                if ok:
                    st.rerun()

    if S.brief:
        b = S.brief
        st.subheader("4. Task brief")
        m = st.columns(4)
        m[0].metric("Questions clarified", len(S.qa))
        m[1].metric("Decisions confirmed", len(b["decisions"]))
        m[2].metric("Acceptance criteria", len(b["acceptance_criteria"]))
        m[3].metric("Still undecided", len(b["open_issues"]))
        left, right = st.columns([1, 2])
        with left, st.container(border=True):
            label("Before: the requester's one-liner")
            st.markdown(f"> {S.task}")
        with right, st.container(border=True):
            label("After: a brief the worker can start on right away")
            render_brief(b)

        st.markdown('### 5. Requirement demo <span class="pill">Take a look first: is this what you meant?</span>', unsafe_allow_html=True)
        feedback = ""
        if not S.demo:
            st.caption("Generate a simple clickable prototype from your answers and the brief. Requesters spot \"that's not what I wanted\" faster from a picture than from text.")
            gen = st.button("Generate requirement demo", type="primary")
        else:
            with st.container(border=True):
                show_demo(S.demo)
            c1, c2, c3 = st.columns([3, 1, 1])
            feedback = c1.text_input("What's wrong?", key=f"demo_fb_{S.demo_ver}", label_visibility="collapsed",
                                     placeholder="What's wrong? e.g. bigger buttons, and I need to see history")
            gen = c2.button("Redo with feedback", disabled=not feedback.strip(), use_container_width=True)
            c3.download_button("Download HTML", S.demo, file_name=f"{S.task_title or 'demo'}.html", mime="text/html",
                               use_container_width=True)
        if gen:
            with st.spinner("Turning the brief into a clickable prototype…"):
                try:
                    S.demo = make_demo(S.task, b, S.history, feedback.strip())
                    S.demo_ver += 1
                    save_doc(f"{S.task_id}-demo", "Demo", f"{S.task_title} · Requirement demo", S.demo)
                    if feedback.strip():
                        S.history.append({"q": "Change requests for the requirement demo", "a": feedback.strip(), "by": "requester"})
                        save_doc(f"{S.task_id}-demofb-{S.demo_ver}", "Q&A", "Change requests for the requirement demo",
                                 f"A (requester, after seeing the demo): {feedback.strip()}")
                        log("Demo revised", feedback.strip())
                    else:
                        log("Demo generated", "Generated an interactive prototype from the brief")
                    ok = True
                except Exception as e:
                    st.error(f"Call failed: {e}")
                    ok = False
            if ok:
                st.rerun()

        st.button("Switch to worker view →", type="primary", on_click=goto, args=("Worker",))

    if S.pending:
        st.subheader("Worker's questions")
        for i, q in enumerate(list(S.pending)):
            with st.container(border=True):
                st.markdown(f"**{q}**")
                reply = st.text_input("Your answer", key=f"pend_{i}_{q}")
                if st.button("Answer and record", key=f"btn_{i}_{q}") and reply:
                    S.history.append({"q": q, "a": reply, "by": "requester"})
                    save_doc(ask_key(q), "Q&A", q, f"A (requester, asked during work): {reply}")
                    log("Requester answered", f"{q} → {reply}")
                    S.chat.append({"role": "assistant", "content": f"The requester answered: **{q}**\n\n{reply}"})
                    S.pending.remove(q)
                    st.rerun()

elif page == "Worker":
    if not S.brief:
        st.info("No task brief yet. Post a task in the \"Requester\" view first.")
        st.button("Go post a task →", on_click=goto, args=("Requester",))
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
        st.caption("If the brief has the answer, it's answered directly with its basis; if not, it's turned into a question for the requester. No guessing.")
        for m in S.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"], unsafe_allow_html=True)
        sc = scenario_of(S.task)
        preset = preset_buttons(sc["worker_qs"] if sc else [], "wq")
        if q := (st.chat_input("e.g. Does it need keyboard input?") or preset):
            S.chat.append({"role": "user", "content": q})
            try:
                with st.spinner("Checking the brief and past Q&A…"):
                    r = answer_worker(S.task, S.brief, S.history, q)
                if r["can_answer"]:
                    msg = f"{r['answer']}\n\n<small>Basis: {r['source']}</small>"
                    save_doc(ask_key(q), "Q&A", q, f"A (AI from the brief, basis: {r['source']}): {r['answer']}")
                    log("AI answered", f"{q} → {r['answer']}")
                else:
                    rq = r["question_for_requester"]
                    S.pending.append(rq)
                    save_doc(ask_key(rq), "Q&A", rq, f"{WAITING} (worker asked: {q})")
                    log("Forwarded to requester", rq)
                    msg = f"Not enough basis in the brief: {r['answer']}\n\n↪ Forwarded to the requester: **{rq}**"
            except Exception as e:
                msg = f"Call failed: {e}"
            S.chat.append({"role": "assistant", "content": msg})
            st.rerun()
        if S.chat:
            st.button("See what the knowledge base captured →", on_click=open_project, args=(S.task_project,))

else:  # Project Knowledge Base
    if not S.projects:
        with st.container(border=True):
            st.markdown("#### 📚 The knowledge base is empty")
            st.markdown(
                "Every project gets **one living document**: the original request, the AI's understanding, the brief, "
                "the demo and every question asked along the way. Post a task in the \"Requester\" view and it fills itself in."
            )
            st.button("Go post a task →", type="primary", on_click=goto, args=("Requester",))
        st.stop()

    # ---- Cover list ----
    if S.kb_open not in S.projects:
        all_stats = {p: project_stats(p) for p in S.projects}
        waiting_total = sum(len(v["waiting"]) for v in all_stats.values())
        m = st.columns(4)
        m[0].metric("Projects", len(S.projects))
        m[1].metric("Tasks", sum(v["tasks"] for v in all_stats.values()))
        m[2].metric("Q&A captured", sum(len(v["qas"]) for v in all_stats.values()))
        m[3].metric("Awaiting requester", waiting_total)

        gallery, chat = st.columns([2, 1], gap="large")
        with gallery:
            st.subheader("All projects")
            st.caption("One document per project: requirements, briefs, demos and every question are inside, updated automatically as work goes on.")
            f1, f2, f3 = st.columns([3, 2, 2], vertical_alignment="center")
            query = f1.text_input("Search", key="kb_query", placeholder="🔍 Search projects and tasks", label_visibility="collapsed")
            sort = f2.selectbox("Sort", KB_SORTS, key="kb_sort", label_visibility="collapsed")
            only_waiting = f3.toggle("Needs answers", key="kb_waiting_only", disabled=not waiting_total,
                                     help="Only projects with questions waiting for the requester")

            q_ = query.strip().lower()
            names = [p for p in S.projects
                     if (not q_ or q_ in p.lower() or any(q_ in d["task"].lower() for d in project_docs(p)))
                     and (not only_waiting or all_stats[p]["waiting"])]
            if sort == "Deadline soonest":
                names.sort(key=lambda p: S.projects[p]["deadline"] or date.max)
            elif sort == "Name":
                names.sort(key=str.lower)
            else:
                names.sort(key=lambda p: all_stats[p]["updated"], reverse=True)

            if not names:
                st.info("No project matches the current filter.")
            per_row = 3
            for start in range(0, len(names), per_row):
                cols = st.columns(per_row)
                for col, name in zip(cols, names[start:start + per_row]):
                    with col:
                        st.markdown(cover_html(name), unsafe_allow_html=True)
                        st.button("Open", key=f"open_{name}", on_click=open_project, args=(name,), use_container_width=True)

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
    ps = project_stats(proj)
    events = ps["events"]
    task_ids = list(dict.fromkeys(d.get("task_id") for d in docs))
    doc_md = project_doc_md(proj)

    st.button("← All projects", on_click=open_project, args=(None,))
    st.markdown(hero_html(proj), unsafe_allow_html=True)
    c = st.columns(5)
    c[0].metric("Tasks", ps["tasks"])
    c[1].metric("Briefs", ps["briefs"])
    c[2].metric("Q&A records", len(ps["qas"]))
    c[3].metric("Awaiting answer", len(ps["waiting"]))
    c[4].metric("Until deadline", f"{(info['deadline'] - date.today()).days} days" if info["deadline"] else "Not set")

    if ps["waiting"]:
        with st.container(border=True):
            w1, w2 = st.columns([4, 1], vertical_alignment="center")
            w1.markdown(f"**⏳ {len(ps['waiting'])} question(s) waiting for the requester**  \n"
                        + "  \n".join(f"· {htmllib.escape(d['title'])}" for d in ps["waiting"]), unsafe_allow_html=True)
            w2.button("Answer now →", type="primary", on_click=goto, args=("Requester",), use_container_width=True,
                      disabled=S.task_project != proj or not S.pending,
                      help="Pending questions are answered in the Requester view while the task is active")

    tab = segmented("Section", KB_TABS, "kb_tab")
    st.write("")

    if tab == KB_TABS[0]:  # Overview: one card per task
        for n, tid in enumerate(reversed(task_ids), 1):
            group = [d for d in docs if d.get("task_id") == tid]
            kinds = {d["type"] for d in group}
            qas = [d for d in group if d["type"] == "Q&A"]
            waiting = [d for d in qas if d["content"].startswith(WAITING)]
            title = group[-1]["task"]
            chips = ['<span class="pill ok">Briefed</span>' if "Brief" in kinds else '<span class="pill warn">No brief yet</span>']
            if "Demo" in kinds:
                chips.append('<span class="pill ok">Demo</span>')
            chips.append(f'<span class="pill">{len(qas)} Q&amp;A</span>')
            if waiting:
                chips.append(f'<span class="pill warn">{len(waiting)} awaiting</span>')
            with st.container(border=True):
                st.markdown(
                    f'<div class="task-card-head"><b>Task {len(task_ids) - n + 1}: {htmllib.escape(title)}</b>{"".join(chips)}'
                    f'<small>Updated {max(d["time"] for d in group):%m-%d %H:%M}</small></div>',
                    unsafe_allow_html=True,
                )
                task_doc = next((d for d in group if d["type"] == "Task"), None)
                if task_doc:
                    understanding = re.search(r"AI's understanding: (.*)", task_doc["content"])
                    if understanding:
                        st.caption(understanding.group(1))
                l, r = st.columns(2)
                with l:
                    label("Answered")
                    answered = [d for d in qas if d not in waiting][-4:]
                    if not answered:
                        st.caption("Nothing yet")
                    for d in answered:
                        st.markdown(f'<div class="qa-row"><b>{htmllib.escape(d["title"])}</b><br>'
                                    f'<small>{htmllib.escape(d["content"])}</small></div>', unsafe_allow_html=True)
                with r:
                    label("Still open")
                    if not waiting:
                        st.caption("No open questions")
                    for d in waiting:
                        st.markdown(f'<div class="qa-row">⏳ {htmllib.escape(d["title"])}</div>', unsafe_allow_html=True)
                b1, b2, _ = st.columns([1, 1, 3])
                b1.button("Read document", key=f"ov_doc_{tid}", on_click=open_project, args=(proj, KB_TABS[1]),
                          use_container_width=True)
                if "Demo" in kinds:
                    b2.button("See demo", key=f"ov_demo_{tid}", on_click=open_project, args=(proj, KB_TABS[2]),
                              use_container_width=True)
        with st.expander("Change deadline"):
            info["deadline"] = st.date_input("Deadline (inferred by AI from the task description; can be edited)",
                                             value=info["deadline"], key=f"ddl_{proj}")

    elif tab == KB_TABS[1]:  # Document: filterable reader + download
        task_names = {tid: next(d["task"] for d in reversed(docs) if d.get("task_id") == tid) for tid in task_ids}
        side, body = st.columns([1, 3], gap="large")
        with side:
            label("Task")
            pick = st.radio("Task", [None] + task_ids, key=f"doc_task_{proj}", label_visibility="collapsed",
                            format_func=lambda t: "All tasks" if t is None else task_names[t])
            label("Sections")
            secs = [s_ for s_ in DOC_SECTIONS if st.checkbox(s_, value=True, key=f"doc_sec_{proj}_{s_}")]
            st.download_button("⬇ Download Markdown", doc_md, file_name=f"{proj}.md", mime="text/markdown",
                               use_container_width=True, help="Always the full document")
        with body, st.container(border=True, height=620):
            shown = project_doc_md(proj, pick, secs)
            st.markdown("\n\n".join(shown.split("\n\n")[2:]) or "_Nothing to show for this filter._",
                        unsafe_allow_html=True)  # Title and overview are already in the header

    elif tab == KB_TABS[2]:  # Demo
        if not ps["demos"]:
            st.info("No requirement demo for this project yet. Generate one in the Requester view after the brief.")
        for d in ps["demos"]:
            with st.expander(f"Task: {d['task']} · {d['time']:%m-%d %H:%M}", expanded=len(ps["demos"]) == 1):
                show_demo(d["content"], height=500)
                st.download_button("Download HTML", d["content"], file_name=f"{d['task']}.html", mime="text/html",
                                   key=f"dl_demo_{d['key']}")

    elif tab == KB_TABS[3]:  # Ask this project
        st.caption("When someone new takes over or you come back a week later, just ask instead of digging through chat history.")
        with st.container(height=440):
            msgs = [m for m in S.kb_chat if m["project"] == proj]
            if not msgs:
                st.caption("No questions yet; try one below.")
            for m in msgs:
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

    else:  # Timeline
        if events:
            tasks = list(dict.fromkeys(e["task"] for e in events))
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
            table = df[["Time", "task", "type", "text"]].rename(columns={"task": "Task", "type": "Event", "text": "Details"})
            st.dataframe(table.iloc[::-1], hide_index=True, width="stretch")
        else:
            st.caption("No events yet")

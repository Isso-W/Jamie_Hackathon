"""Mindreader: requester posts a task -> AI breaks it down and asks follow-ups -> brief is generated
-> an interactive requirement demo is generated -> the worker asks questions while working;
everything is captured in a project knowledge base: one document per project, shown as a cover card, click to see details.

Run:
    pip install -r requirements.txt
    cd harness && npm install          (pi agent harness; needs Node 22+)
    Put your Qwen (Alibaba Cloud Model Studio, international site) API key in apikey.txt in the same folder
    Optional: a Reson8 API key in reson8key.txt (or RESON8_API_KEY) to transcribe meeting recordings
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
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
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
RESON8_URL = "https://api.reson8.dev/v1/speech-to-text/prerecorded"  # Meeting transcription with speaker diarization
RESON8_KEY_FILE = Path(__file__).parent / "reson8key.txt"  # Or set RESON8_API_KEY
SAMPLE_MEETING = Path(__file__).parent / "samples" / "meeting_sample.wav"
AUDIO_TYPES = ["wav", "mp3", "m4a", "ogg", "flac", "webm", "mp4"]
MEETING_SOURCES = ["Record", "Upload", "Paste transcript"]

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

PAGES = ["Requester", "Worker", "Knowledge base"]
LEVELS = ["Post task", "AI questions", "Task brief", "Requirement demo", "Worker Q&A", "Extraction"]  # Requester view: one level at a time
LET_AI = "Let AI decide"
COVER_COLORS = ["#4f46e5", "#0f766e", "#c2410c", "#be185d", "#1d4ed8", "#7c3aed", "#15803d", "#b45309"]
WAITING = "⏳ Awaiting requester's answer"
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

MINUTES_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "What the meeting was about and its outcome, one sentence, max 20 words"},
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"speaker": {"type": "string", "description": "Exactly as in the transcript, e.g. 'Speaker 1'"},
                               "name": {"type": "string", "description": "Name or role said aloud in the meeting, or 'Unknown'"}},
                "required": ["speaker", "name"],
            },
        },
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"decision": {"type": "string", "description": "Max 12 words"},
                               "by": {"type": "string", "description": "Who made or confirmed it"}},
                "required": ["decision", "by"],
            },
            "description": "Only what was explicitly agreed in the meeting",
        },
        "wishes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"who": {"type": "string"}, "wish": {"type": "string", "description": "Max 12 words"}},
                "required": ["who", "wish"],
            },
            "description": "New requirements someone asked for but nobody settled yet",
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"owner": {"type": "string"}, "item": {"type": "string", "description": "Max 10 words"},
                               "due": {"type": "string", "description": "As said in the meeting, e.g. 'Friday'; empty if none"}},
                "required": ["owner", "item", "due"],
            },
        },
        "open_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "Max 12 words"},
                               "to": {"type": "string", "description": "Who should answer it, or empty if unclear"}},
                "required": ["question", "to"],
            },
            "description": "Raised in the meeting but left unanswered",
        },
    },
    "required": ["summary", "speakers", "decisions", "wishes", "action_items", "open_questions"],
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


@st.cache_resource(show_spinner=False)
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


def reson8_key() -> str:
    if os.environ.get("RESON8_API_KEY"):
        return os.environ["RESON8_API_KEY"].strip()
    raw = RESON8_KEY_FILE.read_text(encoding="utf-8-sig").strip() if RESON8_KEY_FILE.exists() else ""
    return raw.split("=", 1)[-1].strip().strip("\"'")  # Accept a bare key or an env-style NAME = "key" line


def transcribe(audio: bytes) -> str:
    """Reson8 speech-to-text with speaker diarization -> "[mm:ss] Speaker N: ..." lines; recorded like every AI answer for offline replay."""
    params = {"diarize": "true", "include_timestamps": "true", "filler_mode": "clean"}

    def produce():
        key = reson8_key()
        if not key:
            raise RuntimeError("No Reson8 API key: put it in reson8key.txt or set RESON8_API_KEY")
        req = urllib.request.Request(f"{RESON8_URL}?{urllib.parse.urlencode(params)}", data=audio,
                                     headers={"Authorization": f"ApiKey {key}", "Content-Type": "application/octet-stream"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Reson8 {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from None

    data = cached("reson8_stt", json.dumps(params), hashlib.sha1(audio).hexdigest(), produce)
    segs = data.get("segments") or [{"speaker_id": 0, "text": data.get("text", ""), "start_ms": data.get("start_ms", 0)}]
    turns = []  # Consecutive segments of one speaker become one turn
    for seg in segs:
        if turns and turns[-1][1] == seg.get("speaker_id", 0):
            turns[-1][2] += " " + seg["text"]
        else:
            turns.append([seg.get("start_ms", 0), seg.get("speaker_id", 0), seg["text"]])
    return "\n".join(f"[{ms // 60000:02d}:{ms // 1000 % 60:02d}] Speaker {sid + 1}: {text.strip()}" for ms, sid, text in turns)


def write_minutes(task: str, transcript: str) -> dict:
    system = (
        "You take the minutes of a meeting about this task. The transcript's speakers are anonymous (Speaker 1, 2…): "
        "name each one from names or roles said aloud, or 'Unknown' if unsure. "
        "Record only what was actually said; never invent decisions. Something counts as decided only if it was explicitly agreed; "
        "a request nobody settled is a wish, and a question nobody answered is an open question. "
        "Write in the same language as the meeting."
    )
    content = f"Task description so far:\n{task or '(not written yet)'}\n\nMeeting transcript:\n{transcript}"
    return call_tool(system, content, "meeting_minutes", MINUTES_SCHEMA)


def minutes_md(m: dict) -> str:
    parts = [f"**Summary**: {m['summary']}"]
    who = "; ".join(f"{x['speaker']} = {x['name']}" for x in m["speakers"])
    if who:
        parts.append(f"**Who spoke**: {who}")
    for title, rows in [("Decided", [f"{d['decision']} ({d['by']})" for d in m["decisions"]]),
                        ("New wishes", [f"{w['who']}: {w['wish']}" for w in m["wishes"]]),
                        ("Open questions", [f"{q['question']}" + (f" (for {q['to']})" if q["to"] else "") for q in m["open_questions"]])]:
        if rows:
            parts.append(f"**{title}**\n" + "\n".join(f"- {r}" for r in rows))
    if m["action_items"]:
        parts.append("**Action items**\n" + "\n".join(
            f"- [ ] **{x['owner']}**: {x['item']}" + (f" · due {x['due']}" if x["due"] else "") for x in m["action_items"]))
    return "\n\n".join(parts)


def minutes_text(m: dict) -> str:
    """Plain-text minutes to copy or append to the task description."""
    lines = [f"Meeting notes: {m['summary']}"]
    for title, rows in [("Decided", [f"{d['decision']} ({d['by']})" for d in m["decisions"]]),
                        ("New wishes", [f"{w['who']}: {w['wish']}" for w in m["wishes"]]),
                        ("Action items", [f"{x['owner']}: {x['item']}" + (f" (due {x['due']})" if x["due"] else "") for x in m["action_items"]]),
                        ("Still open", [q["question"] for q in m["open_questions"]])]:
        if rows:
            lines += [f"{title}:"] + [f"- {r}" for r in rows]
    return "\n".join(lines)


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
    "meeting": {},  # Meeting being processed on Level 1: {"audio"?, "transcript"?, "minutes"?}
    "meetings": [],  # Meetings added to this task: {"summary", "minutes", "transcript"}; saved to the knowledge base when posted
    "analyzed_task": "",  # Task text the current analysis was made from
    "level": 1,  # Level shown in the requester view
    "deepest": 1,  # Deepest level reached so far; levels below it stay clickable
}
KB_DEFAULTS = {
    "projects": {},  # Project name -> {"created", "deadline"}
    "docs": [],  # Records: {"id", "key", "project", "task_id", "task", "type", "title", "content", "time"}
    "events": [],  # Timeline: {"time", "project", "task", "type", "text"}
    "kb_chat": [],  # Knowledge base Q&A: {"project", "role", "content"}
    "kb_open": None,  # Project currently open in the knowledge base; None means the cover list
    "task_seq": 0,
    "archive": {},  # Earlier tasks, task_id -> snapshot of their TASK_DEFAULTS state, so the task switcher can reopen them
}
TASK_WIDGET_PREFIXES = ("opt_", "ans_", "pend_", "btn_", "sub_", "demo_fb", "scenario_")


def fresh(v):
    return v.copy() if isinstance(v, (list, dict)) else v


for k, v in {**TASK_DEFAULTS, **KB_DEFAULTS}.items():
    st.session_state.setdefault(k, fresh(v))
st.session_state.setdefault("project", "Demo Project")
st.session_state.setdefault("page", PAGES[0])
st.session_state.setdefault("role_sel", st.session_state.page)
st.session_state.setdefault("rail", False)  # Left panel collapsed to a narrow rail
S = st.session_state


def clear_task_widgets():
    for k in [k for k in S.keys() if str(k).startswith(TASK_WIDGET_PREFIXES)]:
        del S[k]


def archive_task():
    if S.task_id is not None:
        S.archive[S.task_id] = {k: S[k] for k in TASK_DEFAULTS}


def new_task():
    """Start a new task: archive the current one and clear its widget state; keep the knowledge base."""
    archive_task()
    for k, v in TASK_DEFAULTS.items():
        S[k] = fresh(v)
    clear_task_widgets()
    goto(PAGES[0])


def switch_task(tid: int):
    """Reopen an archived task where it was left; the current one is archived in its place."""
    snap = S.archive.pop(tid)
    archive_task()
    for k in TASK_DEFAULTS:
        S[k] = snap[k]
    clear_task_widgets()
    S.project = S.task_project


def clear_all():
    new_task()
    for k, v in KB_DEFAULTS.items():
        S[k] = fresh(v)


def goto(page: str):
    S.page = S.role_sel = page


def on_role():
    S.page = S.role_sel


def open_project(name):
    S.kb_open = name
    goto("Knowledge base")


def set_level(n: int):
    S.level = n


def descend(n: int):
    S.level, S.deepest = n, max(S.deepest, n)


def answer_pending():
    goto("Requester")
    S.level = 5


def toggle_rail():
    S.rail = not S.rail


def fill_example(name: str):
    """Fill in an example: if there's already a task in hand, archive it first; the example goes under its own project."""
    if S.task_id is not None:
        new_task()
        S.scenario_pick = name  # new_task cleared the chip that triggered this
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
    for i, m in enumerate(S.meetings, 1):
        save_doc(f"{S.task_id}-minutes-{i}", "Meeting", f"Meeting {i}: {m['summary']}",
                 m["minutes"] + "\n\n**Transcript**\n\n" + m["transcript"].replace("\n", "  \n"))
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
    """Combine all of a project's records into one document: one section per task, each with requirement, brief, demo, meetings and all Q&A."""
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
        meets = [d for d in group if d["type"] == "Meeting"]
        if meets:
            lines.append("##### Meeting minutes")
            lines += [f"**{d['title']}**\n\n{d['content']}" for d in meets]
        qas = [d for d in group if d["type"] == "Q&A"]
        if qas:
            lines.append(f"##### Q&A log ({len(qas)})")
            lines += [f"**Q: {d['title']}**  \n{d['content']}  \n<small>{d['time']:%m-%d %H:%M}</small>" for d in qas]
    return "\n\n".join(lines)


def on_check(title: str, key: str):
    log("Subtask done" if S[key] else "Subtask reopened", title)


# ---------- Rendering helpers ----------

# Theme: every colour, the per-level depth ramp and spacing live in the :root block below.
# Each level has its own background, darker and more blue-violet as you descend; the left panel ("shaft")
# is one step darker than the page and cards one step lighter. Text colours stay fixed on every level.
CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Archivo:wght@800&family=Source+Sans+3:wght@400;600;700&display=swap");
:root {
  --bg-1: #0F1A22; --shaft-1: #0B141B; --card-1: #16232D;
  --bg-2: #0E1620; --shaft-2: #0A111A; --card-2: #15202B;
  --bg-3: #0C1220; --shaft-3: #090E19; --card-3: #141B2C;
  --bg-4: #0B0F1D; --shaft-4: #080B17; --card-4: #131829;
  --bg-5: #0A0C18; --shaft-5: #070913; --card-5: #121526;
  --bg-6: #09090F; --shaft-6: #06060B; --card-6: #12121C;
  --bg: var(--bg-1); --shaft: var(--shaft-1); --card: var(--card-1);
  /* City background is dimmed with the level colour, more at each level down */
  --dim-1: rgba(15, 26, 34, .45); --dim-2: rgba(14, 22, 32, .52); --dim-3: rgba(12, 18, 32, .60);
  --dim-4: rgba(11, 15, 29, .68); --dim-5: rgba(10, 12, 24, .76); --dim-6: rgba(9, 9, 15, .84); --dim: var(--dim-1);

  --text: #F2F4F8; --muted: #A3ADC2; --locked: #8A9BA3;
  --line: rgba(163, 173, 194, .18); --line-strong: rgba(163, 173, 194, .38); --locked-line: #34474F;
  /* Cool steel-teal: system info, settled things, interaction states */
  --accent: #2E7D8C; --accent-2: #6FB3C0; --done-icon: #A6D8E0; --accent-wash: rgba(111, 179, 192, .08); --chip-on: rgba(46, 125, 140, .22);
  /* Red + Archivo = depth signal only: where you are / go deeper */
  --depth: #E0472C; --depth-light: #F26A4F; --depth-fill: #C4461F; --depth-fill-hover: #B03D1A; --depth-ring: rgba(224, 71, 44, .28);
  /* Orange = needs your input */
  --ask-wash: rgba(255, 138, 61, .12); --ask-line: rgba(255, 138, 61, .8); --ask-label: #FFC08A; --ask-title: #FF8A3D; --ask-body: #E3E7EE;
  --danger-line: #7A2B33; --danger-text: #FF8B8B; --danger-wash: rgba(255, 139, 139, .08);
  --font-display: "Archivo", "Arial Black", sans-serif;

  --shaft-w: 288px; --rail-w: 76px; --radius: 14px; --touch: 44px;
  --marker: 30px; --fade: 320ms;
}

/* ---- Page ---- */
.stApp {background-color: var(--bg);}
[data-testid="stHeader"] {background: transparent;}
#MainMenu, footer, [data-testid="stAppDeployButton"], [data-testid="stHeaderActionElements"] {display: none;}
.block-container {padding-top: 3.25rem; max-width: 1320px;}
@media (prefers-reduced-motion: no-preference) {
  .stApp {transition: background-color var(--fade) ease;}
  [class*="st-key-pane-"] {animation: surface var(--fade) ease-out;}
}
@keyframes surface {from {opacity: 0;} to {opacity: 1;}}

/* ---- Controls: touch targets, focus, the one depth-red primary, destructive ---- */
.stButton button, .stDownloadButton button, [data-testid="stPopover"] button,
[data-testid="stButtonGroup"] button {min-height: var(--touch);}
[data-baseweb="select"] > div, [data-baseweb="input"] {min-height: var(--touch);}
button:focus-visible, [role="radio"]:focus-visible, [role="option"]:focus-visible {outline: 2px solid var(--accent-2); outline-offset: 2px;}
button[data-testid="stBaseButton-primary"] {background: var(--depth-fill); border-color: var(--depth-fill); color: #fff;}
button[data-testid="stBaseButton-primary"] p {font-family: var(--font-display); font-weight: 800; font-size: 16px; text-transform: uppercase; letter-spacing: .06em;}
button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-primary"]:active {background: var(--depth-fill-hover); border-color: var(--depth-fill-hover); color: #fff;}
button[data-testid="stBaseButton-primary"]:disabled {background: transparent; border-color: var(--line-strong); color: var(--muted);}
[data-testid="stButtonGroup"] button[aria-checked="true"] {background: var(--chip-on); border-color: var(--accent-2); color: var(--text);}
[data-testid="stButtonGroup"] button[aria-checked="true"] p {color: var(--text);}
.st-key-danger_clear button {background: transparent; border: 1px solid var(--danger-line); color: var(--danger-text);}
.st-key-danger_clear button:hover {background: var(--danger-wash); border-color: var(--danger-text); color: var(--danger-text);}
.sr-label p, .st-key-settings button p, .st-key-rail_toggle button p, .st-key-rail_new button p {
  position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap;}

/* ---- Top bar ---- */
.st-key-topbar {border-bottom: 1px solid var(--line); padding-bottom: 12px; margin-bottom: 6px;}
.brand {display: flex; align-items: center; gap: 10px; font-family: var(--font-display); font-weight: 800; font-size: 1.35rem;
         letter-spacing: .02em; color: var(--depth); white-space: nowrap;}
.brand img {flex: none;}

/* ---- Cards and panels ---- */
[class*="st-key-card-"] {background: var(--card); border-color: var(--line) !important; border-radius: var(--radius);}
[class*="st-key-card-q-"], [class*="st-key-card-pend-"] {background: linear-gradient(var(--ask-wash), var(--ask-wash)), var(--card);
  border: 1px solid var(--ask-line) !important;}
[class*="st-key-card-q-"] [data-testid="stCaptionContainer"], [class*="st-key-card-q-"] [data-testid="stCaptionContainer"] p {
  color: var(--ask-body); font-size: 16px; line-height: 1.5;}
.q-label {font-size: 13px; font-weight: 600; letter-spacing: .04em; color: var(--ask-label);}
.q-title {font-size: 24px; font-weight: 700; line-height: 1.3; color: var(--ask-title);}
.st-key-panel-known {border: 1px dashed var(--line-strong) !important; border-radius: var(--radius); background: transparent;}
.request {color: var(--text); font-size: 1rem; line-height: 1.5;}
.eyebrow {font-family: var(--font-display); font-weight: 800; font-size: 20px; letter-spacing: .08em; text-transform: uppercase;
           color: var(--depth); margin-bottom: -1rem;}
[class*="st-key-pane-"] h1 {font-family: var(--font-display) !important; font-weight: 800 !important; font-size: 44px !important;
  letter-spacing: .02em; text-transform: uppercase; color: var(--text);}
.label {font-size: .8rem; letter-spacing: .02em; color: var(--muted); margin-bottom: .3rem;}
.pill {display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: .75rem;
       background: var(--accent-wash); color: var(--accent-2); margin-right: 6px; vertical-align: middle;}

/* ---- Left panel ("shaft") ---- */
@media (min-width: 641px) {
  [data-testid="stColumn"]:has(.st-key-shaft) {flex: 0 0 var(--shaft-w) !important; width: var(--shaft-w) !important; min-width: 0;}
  [data-testid="stColumn"]:has(.st-key-shaft_rail) {flex: 0 0 var(--rail-w) !important; width: var(--rail-w) !important; min-width: 0;}
  [data-testid="stHorizontalBlock"]:has(.st-key-shaft, .st-key-shaft_rail) > [data-testid="stColumn"]:not(:has(.st-key-shaft, .st-key-shaft_rail)) {
    flex: 1 1 0 !important; width: auto !important; min-width: 0;}
  .st-key-shaft, .st-key-shaft_rail {position: sticky; top: 3.75rem;}
}
.st-key-shaft, .st-key-shaft_rail {background: var(--shaft); border: 1px solid var(--line); border-radius: var(--radius); padding: 14px 12px;}
.st-key-shaft_rail {align-items: center; padding: 12px 6px;}
.shaft-label {font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--muted);}
.shaft-project {font-weight: 600; color: var(--text); overflow-wrap: anywhere;}
.st-key-kb_link button, [class*="st-key-nav_"] button {justify-content: flex-start; text-align: left; color: var(--muted); padding: 0 6px;}
.st-key-kb_link button > div, [class*="st-key-nav_"] button > div {justify-content: flex-start; width: 100%;}
.st-key-kb_link button p {font-size: .8rem;}
.st-key-kb_link button:hover, [class*="st-key-nav_"] button:hover {color: var(--accent-2);}
.st-key-kb_link {border-top: 1px solid var(--line); padding-top: 6px;}

/* ---- Level stepper: done / current / locked differ in fill and lightness, not just hue ---- */
[class*="st-key-step-"] {position: relative; padding: 0 0 18px calc(var(--marker) + 12px); min-height: 56px;}
[class*="st-key-step-"]::before {position: absolute; left: 0; top: 0; width: var(--marker); height: var(--marker); box-sizing: border-box;
  border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: .82rem; font-weight: 700;}
[class*="st-key-step-"]::after {content: ""; position: absolute; left: calc(var(--marker) / 2 - 1px); top: calc(var(--marker) + 4px);
  bottom: 4px; width: 2px; border-radius: 2px; background: var(--locked-line);}
[class*="st-key-step-done-"]::before {border: 2px solid var(--accent); background: center / 16px no-repeat
  url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23A6D8E0' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M5 12.5l4.5 4.5L19 7.5'/%3E%3C/svg%3E");}
[class*="st-key-step-done-"]::after {background: var(--accent);}
[class*="st-key-step-cur-"]::before {background: var(--depth-fill); border: 2px solid var(--depth); color: #fff; box-shadow: 0 0 0 4px var(--depth-ring);}
[class*="st-key-step-lock-"]::before {border: 2px solid var(--locked-line); color: var(--locked);}
.step-text {display: flex; flex-direction: column; min-width: 0;}
.step-eyebrow {font-size: .68rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--accent-2);}
.step-title {font-size: .95rem; color: var(--text); line-height: 1.3;}
.step-text.cur .step-title {font-weight: 700;}
.step-text.cur .step-eyebrow {color: var(--depth-light);}
.step-sum {font-size: .8rem; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.step-text.lock .step-eyebrow, .step-text.lock .step-title {color: var(--locked);}
[class*="st-key-go-"] {position: absolute; inset: 0; z-index: 1;}
[class*="st-key-go-"] .stButton, [class*="st-key-go-"] button {width: 100%; height: 100%; min-height: 0;}
[class*="st-key-go-"] button {background: transparent; border: none; border-radius: 10px;}
[class*="st-key-go-"] button:hover {background: var(--accent-wash);}
[class*="st-key-go-"] button p {opacity: 0;}
.st-key-shaft_rail [class*="st-key-step-"] {padding: 0; width: var(--marker); min-height: 52px;}
.st-key-shaft_rail .step-text {position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0);}

/* ---- AI thinking card: bottom-right status with a spinning top ---- */
.mr-thinking {position: fixed; right: 24px; bottom: 24px; z-index: 1000; max-width: min(380px, calc(100% - 48px));
  display: flex; align-items: center; gap: 14px; padding: 10px 18px 10px 10px; box-sizing: border-box;
  background: var(--card); border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 12px 32px rgba(0, 0, 0, .45);}
.mr-desk {flex: none; width: 56px; height: 56px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
  background: radial-gradient(circle at 50% 35%, #B4B8B8, #6E7476);}
.mr-desk svg {width: 44px; height: 52px; overflow: visible;}
.mr-title {font-family: var(--font-display); font-weight: 800; font-size: 13px; letter-spacing: .06em; text-transform: uppercase; color: var(--text);}
.mr-detail {font-size: 14px; line-height: 1.35; color: #A3B3BA;}
.mr-wobble {transform-box: view-box; transform-origin: 50px 110px;}
.mr-shadow {transform-box: view-box; transform-origin: 50px 112px;}
/* Spin: marks on the body travel round it. Seen from the front a point on a turning body moves fastest in the middle
   and slows at the edges (ease-in-out-sine), then fades as it goes round the back. */
.mr-spin {transform-box: view-box; display: none;}
.mr-spin rect {opacity: 0;}
@media (prefers-reduced-motion: no-preference) {
  .mr-spin {display: inline; animation: mr-spin var(--spin, .6s) cubic-bezier(.37, 0, .63, 1) infinite;}
  .mr-spin rect {animation: mr-face var(--spin, .6s) linear infinite; animation-delay: inherit;}
  .mr-wobble {animation: mr-precess 1.6s ease-in-out infinite;}
  .mr-shadow {animation: mr-shadow 1.6s ease-in-out infinite;}
  .mr-thinking.stopping {--spin: 1.4s;}
  .mr-thinking.stopping .mr-wobble {animation: mr-settle .4s ease-out forwards;}
  .mr-thinking.stopping .mr-shadow {animation: none;}
}
@keyframes mr-spin {from {transform: translateX(12px);} to {transform: translateX(88px);}}
@keyframes mr-face {0%, 100% {opacity: 0;} 20%, 80% {opacity: 1;}}
@keyframes mr-precess {0%, 100% {transform: rotate(-5deg);} 50% {transform: rotate(5deg);}}
@keyframes mr-shadow {0%, 100% {transform: scaleX(.85);} 50% {transform: scaleX(1.05);}}
@keyframes mr-settle {from {transform: rotate(3deg);} to {transform: rotate(0);}}

/* ---- Folded-city background: every view, switchable in settings ---- */
.mr-city {position: fixed; inset: 0; z-index: -1; pointer-events: none; overflow: hidden; background: #33454E;
  perspective: 700px; perspective-origin: 50% 42%;}
.mr-plane {position: absolute; left: -250%; width: 600%; height: 2600px; background-color: #33454E; will-change: transform;
  /* Every vertical period divides 180px, so the 180px flight loop is seamless */
  background-image:
    repeating-linear-gradient(90deg, rgba(8, 12, 15, .95) 0 5px, transparent 5px 137px),
    repeating-linear-gradient(90deg, rgba(8, 12, 15, .9) 0 3px, transparent 3px 46px),
    repeating-linear-gradient(0deg, rgba(8, 12, 15, .9) 0 4px, transparent 4px 60px),
    repeating-linear-gradient(0deg, rgba(8, 12, 15, .95) 0 6px, transparent 6px 180px),
    repeating-linear-gradient(90deg, rgba(205, 218, 224, .22) 0 4px, transparent 4px 9px),
    repeating-linear-gradient(0deg, rgba(205, 218, 224, .14) 0 3px, transparent 3px 6px);}
.mr-plane.ceil {top: 0; transform-origin: 50% 0; transform: rotateX(-75deg);}
.mr-plane.floor {bottom: 0; transform-origin: 50% 100%; transform: rotateX(75deg);}
.mr-glow, .mr-dim, .mr-vignette {position: absolute; inset: 0;}
.mr-glow {background: radial-gradient(ellipse 55% 90px at 50% 42%, rgba(240, 247, 250, .9), rgba(176, 205, 215, .45) 35%, transparent 75%),
  linear-gradient(180deg, transparent 22%, rgba(160, 188, 200, .28) 42%, transparent 62%);}
.mr-dim {background: var(--dim);}
.mr-vignette {background: radial-gradient(ellipse 120% 90% at 50% 42%, transparent 40%, rgba(5, 8, 11, .7) 100%);}
@media (prefers-reduced-motion: no-preference) {
  .mr-plane.ceil {animation: mr-fly-ceil 16s linear infinite;}
  .mr-plane.floor {animation: mr-fly-floor 16s linear infinite;}
  .mr-dim {transition: background-color var(--fade) ease;}
}
@keyframes mr-fly-ceil {from {transform: rotateX(-75deg) translateY(-180px);} to {transform: rotateX(-75deg) translateY(0);}}
@keyframes mr-fly-floor {from {transform: rotateX(75deg) translateY(180px);} to {transform: rotateX(75deg) translateY(0);}}

/* With the city on, the UI turns translucent; body text always sits on a card, only the big titles sit on the city.
   body:has() rather than .stApp so popovers and dropdown menus (rendered outside .stApp) follow too. */
/* isolation alone makes the stacking context; position: relative here would collapse Streamlit's absolutely placed view to 0 height */
body:has(.mr-city) .stApp {isolation: isolate;}
body:has(.mr-city) [data-testid="stAppViewContainer"], body:has(.mr-city) [data-testid="stMain"] {background: transparent;}
body:has(.mr-city) .st-key-topbar {position: relative; z-index: 20; padding: 10px 14px; border: 1px solid var(--line); border-radius: var(--radius);
  background: color-mix(in srgb, var(--bg) 85%, transparent); backdrop-filter: blur(10px);}
body:has(.mr-city) .st-key-shaft, body:has(.mr-city) .st-key-shaft_rail {background: color-mix(in srgb, var(--shaft) 88%, transparent);
  backdrop-filter: blur(10px);}
body:has(.mr-city) [class*="st-key-card-"], body:has(.mr-city) .mr-thinking, body:has(.mr-city) .round-note,
body:has(.mr-city) [data-testid="stExpander"] details, body:has(.mr-city) [data-testid="stPopoverBody"],
body:has(.mr-city) [data-baseweb="popover"] ul, body:has(.mr-city) [data-testid="stAlert"] {
  background: color-mix(in srgb, var(--card) 92%, transparent);}
body:has(.mr-city) [data-testid="stAlert"] {border-radius: .5rem;}
body:has(.mr-city) [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {background: rgba(19, 38, 43, .92);}
body:has(.mr-city) [class*="st-key-card-q-"], body:has(.mr-city) [class*="st-key-card-pend-"] {
  background: linear-gradient(var(--ask-wash), var(--ask-wash)), color-mix(in srgb, var(--card) 92%, transparent);}
body:has(.mr-city) .st-key-panel-known {background: color-mix(in srgb, var(--bg) 85%, transparent);}
body:has(.mr-city) .eyebrow, body:has(.mr-city) [data-testid="stMain"] h1 {text-shadow: 0 2px 18px rgba(0, 0, 0, .6);}
.round-note {display: inline-block; padding: 6px 12px; border: 1px solid var(--line); border-radius: 10px; color: var(--text);}

/* ---- Knowledge base covers ---- */
.cover {border: 1px solid var(--line); border-radius: 12px; overflow: hidden; margin-bottom: .4rem; background: var(--card);}
.cover-band {height: 76px; display: flex; align-items: flex-end; justify-content: space-between; padding: 10px 14px;}
.cover-band b {font-size: 1.7rem; line-height: 1; color: rgba(255,255,255,.92);}
.cover-band i {font-style: normal; font-size: .72rem; color: rgba(255,255,255,.85);}
.cover-body {padding: 10px 14px 12px;}
.cover-title {font-weight: 600; font-size: 1.02rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.cover-sub {font-size: .8rem; color: var(--muted); margin: 2px 0 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
</style>
"""

# st.html strips inline <svg>, so the eye icon ships as an SVG data URI
EYE_ICON = ("<img alt='' width='28' height='28' src=\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
            "fill='none' stroke='%23E0472C' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E"
            "%3Cpath d='M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z'/%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3C/svg%3E\">")
# Classic three-part top in our own proportions: thin tapered handle, wide flat disc, short concave cone down to a point
TOP_SHAPE = ('<path d="M47 10 Q50 6 53 10 L56 41 L44 41 Z"/>'
             '<path d="M44 39 C30 39 12 43 10 50 C12 58 30 62 50 62 C70 62 88 58 90 50 C88 43 70 39 56 39 Z"/>'
             '<path d="M20 58 Q44 70 49 108 L51 108 Q56 70 80 58 Z"/>')
# Marks spaced evenly round the top (one every quarter of the 0.6s turn): dark grooves, a white and a warm highlight
SPIN_MARKS = [("#05080B", .6, 2.5), ("#FFFFFF", .45, 7), ("#05080B", .6, 2.5), ("#FF8A3D", .45, 4)]
TOP_SVG = (
    '<svg viewBox="0 0 100 120" aria-hidden="true"><defs>'
    '<linearGradient id="mr-metal" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="#101315"/>'
    '<stop offset=".2" stop-color="#2C3236"/><stop offset=".45" stop-color="#7D878C"/><stop offset=".6" stop-color="#A9B2B6"/>'
    '<stop offset=".82" stop-color="#3A4145"/><stop offset="1" stop-color="#0C0E10"/></linearGradient>'
    '<linearGradient id="mr-shade" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".1"/>'
    '<stop offset=".45" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".45"/></linearGradient>'
    f'<clipPath id="mr-body">{TOP_SHAPE}</clipPath></defs>'
    '<ellipse class="mr-shadow" cx="50" cy="112" rx="14" ry="3" fill="#05080B" fill-opacity=".55"/>'
    # Each part takes the metal gradient across its own width, so the narrow handle and cone stay dark at the edges
    f'<g class="mr-wobble"><g fill="url(#mr-metal)">{TOP_SHAPE}</g><g clip-path="url(#mr-body)">'
    '<rect width="100" height="120" fill="url(#mr-shade)"/>'
    + "".join(f'<g class="mr-spin" style="animation-delay: -{i * .15:.2f}s"><rect x="{-w / 2}" width="{w}" height="120" '
              f'fill="{c}" fill-opacity="{a}"/></g>' for i, (c, a, w) in enumerate(SPIN_MARKS)) + '</g>'
    '<path d="M10 50 C12 58 30 62 50 62 C70 62 88 58 90 50" fill="none" stroke="#07090A" stroke-width="1.6"/></g></svg>'
)
SKIP_HINT = "Skip any question and the AI will pick a default and flag it in the brief."


def thinking_html(title: str, detail: str, stopping: bool = False) -> str:
    return (f'<div class="mr-thinking{" stopping" if stopping else ""}" role="status" aria-live="polite">'
            f'<div class="mr-desk">{TOP_SVG}</div><div><div class="mr-title">{htmllib.escape(title)}</div>'
            f'<div class="mr-detail">{htmllib.escape(detail)}</div></div></div>')


@contextmanager
def thinking(title: str, detail: str):
    """The one AI-waiting indicator, used in place of st.spinner: a small bottom-right card with a spinning top."""
    slot = st.empty()
    slot.markdown(thinking_html(title, detail), unsafe_allow_html=True)  # st.markdown keeps inline SVG; st.html strips it
    try:
        yield
    except BaseException:  # Includes st.rerun's exception: just clear the card
        slot.empty()
        raise
    slot.markdown(thinking_html(title, detail, stopping=True), unsafe_allow_html=True)
    time.sleep(0.4)  # Let the top slow to a stop before the next level fades in
    slot.empty()


def depth_css(level: int) -> str:
    return (f"<style>:root {{--bg: var(--bg-{level}); --shaft: var(--shaft-{level}); --card: var(--card-{level}); "
            f"--dim: var(--dim-{level});}}</style>")


CITY_HTML = ('<div class="mr-city" aria-hidden="true"><div class="mr-plane ceil"></div><div class="mr-plane floor"></div>'
             '<div class="mr-glow"></div><div class="mr-dim"></div><div class="mr-vignette"></div></div>')


def unlocked() -> int:
    """Deepest level the requester can open: levels need the data they show, and worker activity opens Level 5."""
    if not S.analysis:
        return 1
    if not S.brief:
        return 2
    return min(len(LEVELS), max(S.deepest, 5 if S.chat or S.pending else 3))


def current_project() -> str:
    return S.task_project if S.task_id is not None else (S.project.strip() or "Untitled Project")


def task_docs() -> list:
    return [d for d in S.docs if d["task_id"] == S.task_id] if S.task_id is not None else []


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def level_summary(n: int) -> str:
    if n == 1:
        return S.task
    if n == 2:
        return f"{plural(len(S.qa), 'answer')} over {plural(len({h['round'] for h in S.qa}), 'round')}" if S.qa else "No questions needed"
    if n == 3:
        return S.brief["goal"] if S.brief else ""
    if n == 4:
        return "Prototype ready" if S.demo else "No demo yet"
    if n == 5:
        asked = plural(sum(m["role"] == "user" for m in S.chat), "question")
        return f"{asked} asked" + (f" · {len(S.pending)} waiting" if S.pending else "")
    return f"{plural(len(task_docs()), 'record')} captured"


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


# ---------- Top bar and left panel ----------


def render_topbar():
    with st.container(key="topbar", horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
        st.html(f'<div class="brand">{EYE_ICON}<span>Mindreader</span></div>', width="content")
        with st.container(horizontal=True, vertical_alignment="center", width="content"):
            st.segmented_control("Role", PAGES, key="role_sel", required=True, on_change=on_role, label_visibility="collapsed")
            with st.popover("Settings", icon=":material/settings:", key="settings"):
                st.radio("AI engine", ["pi", "direct"], key="engine", horizontal=True,
                         help="pi: every call runs as a pi agent job. direct: plain Qwen API calls. pi falls back to direct on failure.")
                if S.engine == "pi":
                    try:
                        st.selectbox("Model", start_harness()["models"], key="pi_model")
                    except Exception as e:
                        st.caption(f"pi harness unavailable: {e}")
                st.toggle("Offline demo mode", key="replay",
                          help="Prefer recorded answers, no network needed. Normal calls are recorded automatically; failed API calls also fall back to recordings.")
                # The default lives on the widget, not in setdefault: a popover that was never opened falls back to the widget default
                st.toggle("City background", value=True, key="city", help="Animated background behind every view; off gives a plain colour")
                st.caption(f"{plural(len(get_cache()), 'AI answer')} recorded")
                st.divider()
                st.button("Clear all project data", key="danger_clear", icon=":material/warning:", on_click=clear_all, width="stretch",
                          help="Deletes every task and the whole knowledge base")


def task_key():
    return S.task_id if S.task_id is not None else "draft"


def task_label(k) -> str:
    if k == "new":
        return "+ New task"
    if k == "draft":
        return "New task (not posted yet)"
    t = S if k == S.task_id else S.archive[k]
    return t["task_title"] if t["task_project"] == current_project() else f"{t['task_title']} · {t['task_project']}"


def on_task_switch():
    k = S.task_switch
    if k == "new":
        new_task()
    elif k not in ("draft", S.task_id):
        switch_task(k)


def render_steps():
    top, rules = unlocked(), []
    with st.container(key="steps", gap=None):
        for n, name in enumerate(LEVELS, 1):
            state = "cur" if n == S.level else ("done" if n <= top else "lock")
            key = f"step-{state}-{n}"
            rules.append(f'.st-key-{key}::before {{content: "{"" if state == "done" else n}";}}')
            if n == len(LEVELS):
                rules.append(f".st-key-{key}::after {{display: none;}}")
            with st.container(key=key, gap=None):
                summary = f'<span class="step-sum">{htmllib.escape(level_summary(n))}</span>' if state == "done" else ""
                current = ' aria-current="step"' if state == "cur" else ""
                locked = '<span class="step-sum">Locked</span>' if state == "lock" and S.rail else ""
                st.html(f'<div class="step-text {state}"{current}><span class="step-eyebrow">Level {n}</span>'
                        f'<span class="step-title">{name}</span>{summary}{locked}</div>')
                if state == "done":
                    st.button(f"Go back to Level {n}: {name}", key=f"go-{n}", on_click=set_level, args=(n,))
    st.html("<style>" + "\n".join(rules) + "</style>")


def render_shaft(page: str):
    if S.rail and page == "Requester":
        with st.container(key="shaft_rail"):
            st.button("Expand panel", key="rail_toggle", icon=":material/left_panel_open:", on_click=toggle_rail)
            render_steps()
            st.button("New task", key="rail_new", icon=":material/add:", on_click=new_task)
        return
    with st.container(key="shaft"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.html(f'<div class="shaft-label">Project</div><div class="shaft-project">{htmllib.escape(current_project())}</div>')
            if page == "Requester":
                st.button("Collapse panel", key="rail_toggle", icon=":material/left_panel_close:", on_click=toggle_rail)
        ids = sorted([*S.archive, *([S.task_id] if S.task_id is not None else [])])
        S.task_switch = task_key()
        st.selectbox("Current task", ids + (["draft"] if S.task_id is None else []) + ["new"], key="task_switch",
                     format_func=task_label, on_change=on_task_switch)
        if S.pending:
            st.warning(f"{plural(len(S.pending), 'worker question')} waiting for the requester")
            if page != "Requester" or S.level != 5:
                st.button("Answer at Level 5", key="pending_go", on_click=answer_pending, width="stretch")
        if page == "Requester":
            render_steps()
        elif page == "Knowledge base" and S.projects:
            st.html('<div class="shaft-label">Projects</div>')
            st.button("All projects", key="nav_all", type="tertiary", icon=":material/grid_view:", on_click=open_project,
                      args=(None,), width="stretch")
            for p in sorted(S.projects):
                st.button(p, key=f"nav_{p}", type="tertiary", icon=":material/description:", on_click=open_project,
                          args=(p,), width="stretch")
        if page != "Knowledge base":
            n_qa = sum(d["type"] == "Q&A" for d in S.docs)
            st.button(f"Knowledge base: {plural(len(S.projects), 'project doc')} · {n_qa} Q&A · {plural(len(S.events), 'event')}",
                      key="kb_link", type="tertiary", on_click=goto, args=("Knowledge base",), width="stretch")


# ---------- Requester levels ----------


def action_row(n: int):
    """The level's action row sits on its own card, so the helper text never lands on the background."""
    return st.container(border=True, key=f"card-act-{n}", horizontal=True, vertical_alignment="center")


def level_header(n: int):
    st.html(f'<div class="eyebrow">Level {n}</div>')
    st.title(LEVELS[n - 1], anchor=False)


def pick_example():
    if S.scenario_pick:
        fill_example(S.scenario_pick)


def make_brief() -> bool:
    with thinking("Building your brief", "Turning it into a brief you can start on…"):
        try:
            S.brief, S.demo = write_brief(S.task, S.analysis, S.qa, S.remaining), None
            S.history, S.deepest = list(S.qa), 3
            save_doc(f"{S.task_id}-brief", "Brief", f"{S.task_title} · Task brief", brief_md(S.brief))
            for i, h in enumerate(S.qa):
                save_doc(f"{S.task_id}-clar-{i}", "Q&A", h["q"], f"A (requester, pre-work clarification round {h['round']}): {h['a']}")
            log("Brief generated", S.brief["goal"])
            return True
        except Exception as e:
            st.error(f"Call failed: {e}")
            return False


def build_demo(feedback: str = "") -> bool:
    with thinking("Building your demo", "Turning the brief into a clickable prototype…"):
        try:
            S.demo = make_demo(S.task, S.brief, S.history, feedback)
            S.demo_ver += 1
            save_doc(f"{S.task_id}-demo", "Demo", f"{S.task_title} · Requirement demo", S.demo)
            if feedback:
                S.history.append({"q": "Change requests for the requirement demo", "a": feedback, "by": "requester"})
                save_doc(f"{S.task_id}-demofb-{S.demo_ver}", "Q&A", "Change requests for the requirement demo",
                         f"A (requester, after seeing the demo): {feedback}")
                log("Demo revised", feedback)
            else:
                log("Demo generated", "Generated an interactive prototype from the brief")
            return True
        except Exception as e:
            st.error(f"Call failed: {e}")
            return False


def add_meeting_to_task():
    m = S.meeting
    S.meetings.append({"summary": m["minutes"]["summary"], "minutes": minutes_md(m["minutes"]), "transcript": m["transcript"]})
    S.task = (S.task.rstrip() + "\n\n" if S.task.strip() else "") + minutes_text(m["minutes"])
    S.meeting = {}


def discard_meeting():
    S.meeting = {}


def meeting_box():
    """Optional supplement to the task: record, upload or paste a meeting -> Reson8 transcript -> AI minutes, all copyable."""
    m = S.meeting
    title = f"Meeting notes · {plural(len(S.meetings), 'meeting')} added" if S.meetings else "Meeting notes (optional)"
    with st.expander(title, icon=":material/mic:", expanded=bool(m)):
        if not m.get("transcript"):
            st.caption("Record or upload a meeting about this task. Reson8 transcribes it and tells the speakers apart; "
                       "the AI then writes minutes you can copy or add to the task description.")
            src = st.segmented_control("Source", MEETING_SOURCES, key="sub_meet_src", required=True, default=MEETING_SOURCES[0],
                                       label_visibility="collapsed")
            audio = None
            if src == "Record":
                rec = st.audio_input("Record the meeting", key="sub_meet_rec")
                audio = rec.getvalue() if rec else None
            elif src == "Upload":
                up = st.file_uploader("Meeting recording", type=AUDIO_TYPES, key="sub_meet_up")
                audio = (up.getvalue() if up else None) or m.get("audio")
                if not up and SAMPLE_MEETING.exists() and st.button("Use the sample meeting", type="tertiary", icon=":material/library_music:"):
                    S.meeting = {"audio": SAMPLE_MEETING.read_bytes()}
                    st.rerun()
                if audio:
                    st.audio(audio)
            else:
                text = st.text_area("Transcript", key="sub_meet_text", height=160,
                                    placeholder="Speaker 1: …\nSpeaker 2: …  (for meetings in a language Reson8 doesn't cover)")
                if st.button("Use this transcript", disabled=not text.strip()):
                    S.meeting = {"transcript": text.strip()}
                    st.rerun()
            if audio is not None:
                st.caption("Reson8 covers English, German, French, Spanish, Italian, Dutch, Portuguese, Polish, Swedish and Frisian.")
                if st.button("Transcribe with Reson8", icon=":material/graphic_eq:"):
                    with thinking("Transcribing", "Reson8 is telling the speakers apart…"):
                        try:
                            S.meeting = {"transcript": transcribe(audio)}
                            ok = True
                        except Exception as e:
                            st.error(f"Transcription failed: {e}")
                            ok = False
                    if ok:
                        st.rerun()
            return
        label("Transcript · copy with the button on the right")
        st.code(m["transcript"], language=None, wrap_lines=True, height=200)
        if not m.get("minutes"):
            with st.container(horizontal=True):
                write = st.button("Write the minutes", icon=":material/edit_note:")
                st.button("Discard", key="meet_discard_t", on_click=discard_meeting)
            if write:
                with thinking("Writing the minutes", "Pulling out decisions, wishes and open questions…"):
                    try:
                        S.meeting = {**m, "minutes": write_minutes(S.task, m["transcript"])}
                        ok = True
                    except Exception as e:
                        st.error(f"Call failed: {e}")
                        ok = False
                if ok:
                    st.rerun()
            return
        label("Minutes")
        st.markdown(minutes_md(m["minutes"]))
        label("Minutes as text · copy with the button on the right")
        st.code(minutes_text(m["minutes"]), language=None, wrap_lines=True)
        with st.container(horizontal=True):
            st.button("Add to the task description", icon=":material/playlist_add:", on_click=add_meeting_to_task,
                      help="Appends the minutes to the description above; the full minutes and transcript go to the knowledge base when the task is posted")
            st.button("Discard", key="meet_discard_m", on_click=discard_meeting)


def level_post():
    level_header(1)
    with st.container(border=True, key="card-post"):
        st.pills("Start from an example", list(SCENARIOS), key="scenario_pick", on_change=pick_example)
        if S.task_id is None:
            S.project = st.text_input("Project", value=S.project, help="New tasks go under this project")
        S.task = st.text_area("Describe what you need in plain language; the more casual the better", value=S.task, height=130)
    meeting_box()
    with action_row(1):
        go = st.button("Let AI understand the task", type="primary", disabled=not S.task.strip())
    if go:
        if S.analysis and S.task == S.analyzed_task:  # Nothing changed since the last analysis: just go back down
            descend(2)
            st.rerun()
        with thinking("AI is understanding", "Reading your request…"):
            try:
                S.analysis, S.brief, S.demo = analyze(S.task), None, None
                S.qa, S.round, S.remaining, S.stopped = [], 1, [], False
                S.clarified = not S.analysis["questions"]
                S.analyzed_task, S.deepest = S.task, 1
                on_analyzed(S.analysis)
                ok = True
            except Exception as e:
                st.error(f"Call failed: {e}")
                ok = False
        if ok:
            descend(2)
            st.rerun()  # Kept outside the try: st.rerun works by raising, so it must not be swallowed by except


def question_cards(questions: list) -> dict:
    answers = {}
    for i, q in enumerate(questions):
        with st.container(border=True, key=f"card-q-{S.round}-{i}"):
            st.html(f'<div class="q-label">Question {i + 1} of {len(questions)}</div>'
                    f'<div class="q-title">{htmllib.escape(q["question"])}</div>')
            st.caption(f"Why ask: {q['why']}")
            choice = st.pills("Your answer", q["options"] + [LET_AI], key=f"opt_{S.round}_{i}", label_visibility="collapsed")
            text = st.text_input("Add a note (optional)", key=f"ans_{S.round}_{i}", placeholder="Add a note (optional)",
                                 label_visibility="collapsed")
            if choice and choice != LET_AI:
                answers[q["question"]] = f"{choice}. {text}" if text else choice
            else:
                answers[q["question"]] = text or "Not sure, decide later"
    return answers


def level_questions():
    a = S.analysis
    level_header(2)
    with st.container(border=True, key="card-request", horizontal=True, vertical_alignment="center", wrap=False):
        st.html(f'<div class="label">Your request</div><div class="request">{htmllib.escape(S.task)}</div>')
        st.button("Edit", key="edit_request", icon=":material/edit:", on_click=set_level, args=(1,))
    st.info(f"**AI's understanding:** {a['understanding']}")
    if a["needs_meeting"]:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.warning(f"Suggest talking face to face: {a['meeting_reason']}")
            st.button("Add meeting notes", icon=":material/mic:", on_click=set_level, args=(1,),
                      help="Record the meeting on Level 1, then add its minutes to the task")

    answers = {}
    qcol, kcol = st.columns([3, 2], gap="large")
    with qcol:
        if S.qa:
            with st.expander(f"Answered so far ({len(S.qa)})"):
                for i, h in enumerate(S.qa, 1):
                    st.markdown(f"**Q{i}. {h['q']}**  \n{h['a']}")
        if not S.clarified:
            if S.round > 1:
                st.markdown(f'<div class="round-note"><span class="pill">Round {S.round}</span>Follow-up questions raised by your answers</div>',
                            unsafe_allow_html=True)
            answers = question_cards(a["questions"])
        else:
            rounds = len({h["round"] for h in S.qa})
            if S.stopped or S.remaining:
                st.warning(f"Clarification stopped after {plural(rounds, 'round')}; open items go into the brief")
                bullets(S.remaining)
            else:
                st.success(f"Request fully explained: {plural(len(S.qa), 'answer')} over {plural(rounds, 'round')}")
    with kcol, st.container(border=True, key="panel-known"):
        label("Already clear: no need to answer")
        bullets(a["known"] or ["(nothing yet)"])

    with action_row(2):
        go = st.button("Descend to Level 3", type="primary")
        stop = not S.clarified and S.round > 1 and st.button(
            "Enough, write the brief", help="Stop asking; anything still unclear goes into the brief's open issues")
        if not S.clarified:
            st.caption(SKIP_HINT)
    if not (go or stop):
        return
    if S.clarified and S.brief:
        descend(3)
        st.rerun()
    if not S.clarified:
        S.qa += [{"q": q, "a": ans, "by": "requester", "round": S.round} for q, ans in answers.items()]
        if stop or S.round >= MAX_ROUNDS:
            S.clarified = S.stopped = True
        else:
            with thinking("AI is reading your answers", "Checking what they still leave open…"):
                try:
                    r = clarify(S.task, a, S.qa, S.round + 1)
                    a["understanding"], a["questions"], S.remaining = r["understanding"], r["questions"], r["remaining"]
                    S.round += 1
                    S.clarified = r["complete"] or not r["questions"]
                    log("Clarification round", f"Round {S.round - 1}: {len(answers)} answer(s)")
                    ok = True
                except Exception as e:
                    S.qa = S.qa[: len(S.qa) - len(answers)]  # Let the requester resubmit the same round
                    st.error(f"Call failed: {e}")
                    ok = False
            if not ok:
                return
            if not S.clarified:  # The answers opened follow-ups: stay on this level for the next round
                st.rerun()
    if make_brief():
        descend(3)
        st.rerun()


def level_brief():
    b = S.brief
    level_header(3)
    with st.container(border=True, key="card-metrics"):
        m = st.columns(4)
        m[0].metric("Questions clarified", len(S.qa))
        m[1].metric("Decisions confirmed", len(b["decisions"]))
        m[2].metric("Acceptance criteria", len(b["acceptance_criteria"]))
        m[3].metric("Still undecided", len(b["open_issues"]))
    left, right = st.columns([1, 2])
    with left, st.container(border=True, key="card-before"):
        label("Before: the requester's one-liner")
        st.markdown(f"> {S.task}")
    with right, st.container(border=True, key="card-after"):
        label("After: a brief the worker can start on right away")
        render_brief(b)
    with action_row(3):
        go = st.button("Descend to Level 4", type="primary")
        redo = st.button("Regenerate brief", icon=":material/refresh:")
        if not S.demo:
            st.caption("Descending builds a clickable prototype from this brief.")
    if redo and make_brief():
        st.rerun()
    if go and (S.demo or build_demo()):
        descend(4)
        st.rerun()


def level_demo():
    level_header(4)
    intro = "A simple clickable prototype built from your answers and the brief. Requesters spot \"that's not what I wanted\" faster from a picture than from text."
    if not S.demo:
        with action_row(4):
            gen = st.button("Generate requirement demo", type="primary")
            skip = st.button("Continue to Level 5 without a demo")
            st.caption(intro)
        if gen and build_demo():
            st.rerun()
        if skip:
            descend(5)
            st.rerun()
        return
    with st.container(border=True, key="card-demo"):
        st.caption(intro)
        show_demo(S.demo)
        c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
        feedback = c1.text_input("What's wrong?", key=f"demo_fb_{S.demo_ver}", label_visibility="collapsed",
                                 placeholder="What's wrong? e.g. bigger buttons, and I need to see history")
        redo = c2.button("Redo with feedback", disabled=not feedback.strip(), use_container_width=True)
        c3.download_button("Download HTML", S.demo, file_name=f"{S.task_title or 'demo'}.html", mime="text/html",
                           use_container_width=True)
    if redo and build_demo(feedback.strip()):
        st.rerun()
    with action_row(4):
        go = st.button("Descend to Level 5", type="primary")
    if go:
        descend(5)
        st.rerun()


def level_worker_qa():
    level_header(5)
    if S.pending:
        for i, q in enumerate(list(S.pending)):
            with st.container(border=True, key=f"card-pend-{i}"):
                st.html(f'<div class="q-label">Waiting for your answer · {i + 1} of {len(S.pending)}</div>'
                        f'<div class="q-title">{htmllib.escape(q)}</div>')
                reply = st.text_input("Your answer", key=f"pend_{i}_{q}")
                if st.button("Answer and record", key=f"btn_{i}_{q}") and reply:
                    S.history.append({"q": q, "a": reply, "by": "requester"})
                    save_doc(ask_key(q), "Q&A", q, f"A (requester, asked during work): {reply}")
                    log("Requester answered", f"{q} → {reply}")
                    S.chat.append({"role": "assistant", "content": f"The requester answered: **{q}**\n\n{reply}"})
                    S.pending.remove(q)
                    st.rerun()
    with st.container(border=True, key="card-chat"):
        label("Worker conversation")
        st.caption("The worker asks questions in the Worker view while building. If the brief has the answer, the AI answers with its basis; "
                   "if not, the question comes back here for you.")
        if not S.chat:
            st.caption("No questions from the worker yet.")
        for m in S.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"], unsafe_allow_html=True)
    with action_row(5):
        go = st.button("Descend to Level 6", type="primary")
        st.button("Open worker view", icon=":material/construction:", on_click=goto, args=("Worker",))
    if go:
        descend(6)
        st.rerun()


def level_extract():
    level_header(6)
    docs = task_docs()
    with st.container(border=True, key="card-metrics"):
        st.caption("Everything this task produced is saved into the project's document in the knowledge base, so anyone can pick it up later.")
        m = st.columns(4)
        m[0].metric("Records captured", len(docs))
        m[1].metric("Q&A", sum(d["type"] == "Q&A" for d in docs))
        m[2].metric("Decisions", len(S.brief["decisions"]))
        m[3].metric("Timeline events", sum(e["project"] == S.task_project and e["task"] == S.task_title for e in S.events))
    with st.container(border=True, key="card-records"):
        label("Captured from this task")
        for d in docs:
            st.markdown(f'<span class="pill">{d["type"]}</span>{htmllib.escape(d["title"])}', unsafe_allow_html=True)
    with action_row(6):
        st.button("Extract knowledge", type="primary", on_click=open_project, args=(S.task_project,))
        st.download_button("Download project document", project_doc_md(S.task_project), file_name=f"{S.task_project}.md",
                           mime="text/markdown", icon=":material/download:")


LEVEL_VIEWS = [level_post, level_questions, level_brief, level_demo, level_worker_qa, level_extract]


# ---------- Worker and knowledge base views ----------


def worker_view():
    st.title("Worker", anchor=False)
    with st.container(border=True, key="card-sheet-worker"):  # Body text on a card; only the title sits on the city
        if not S.brief:
            st.info("No task brief yet. Post a task in the \"Requester\" view first.")
            st.button("Go post a task →", on_click=goto, args=("Requester",))
            return

        left, right = st.columns([1, 1])
        with left:
            with st.container(border=True, key="card-mytask"):
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
                    with thinking("Checking the brief", "Looking through the brief and past Q&A…"):
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


def kb_view():
    if not S.projects:
        st.title("Knowledge base", anchor=False)
        with st.container(border=True, key="card-sheet-kb"):
            st.info("The knowledge base is empty. After you post a task and generate a brief in the \"Requester\" view, the process is captured here automatically.")
            st.button("Go post a task →", on_click=goto, args=("Requester",))
        return
    if S.kb_open not in S.projects:
        kb_covers()
    else:
        kb_project(S.kb_open)


def kb_covers():
    st.title("All projects", anchor=False)
    gallery, chat = st.container(border=True, key="card-sheet-kb").columns([2, 1], gap="large")
    with gallery:
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
                with thinking("Searching", "Looking through every project document…"):
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


def kb_project(proj: str):
    info = S.projects[proj]
    docs = project_docs(proj)
    events = [e for e in S.events if e["project"] == proj]
    tasks = list(dict.fromkeys(e["task"] for e in events))
    doc_md = project_doc_md(proj)

    st.button("All projects", icon=":material/arrow_back:", on_click=open_project, args=(None,))
    st.title(proj, anchor=False)
    with st.container(border=True, key="card-sheet-kb"):  # Body text on a card; only the title sits on the city
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
        with st.container(border=True, key="card-doc"):
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
                with thinking("Searching", "Looking through this project's document…"):
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


# ---------- Page ----------

st.set_page_config(page_title="Mindreader", page_icon=":material/visibility:", layout="wide")
st.html(CSS)

page = S.page
if page == "Requester":
    S.level = min(S.level, unlocked())
st.html(depth_css(S.level if page == "Requester" else 1))
if S.get("city", True):
    st.markdown(CITY_HTML, unsafe_allow_html=True)  # st.markdown, not st.html, so aria-hidden and the plain divs are kept
render_topbar()

shaft_col, main = st.columns([1, 4], gap="large")
with shaft_col:
    render_shaft(page)
with main:
    if page == "Requester":
        with st.container(key=f"pane-l{S.level}"):  # A new key per level replays the one fade-in
            LEVEL_VIEWS[S.level - 1]()
    elif page == "Worker":
        worker_view()
    else:
        kb_view()

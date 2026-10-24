"""需求澄清助手：需求方发布任务 -> AI拆解追问 -> 生成简报 -> 生成可交互的需求 Demo -> 执行者干活时提问；
过程自动沉淀到项目知识库：每个项目一份文档，以封面卡片展示，点开看详情。

运行：
    pip install -r requirements.txt
    把通义千问（阿里云 Model Studio 国际站）的 API key 放进同目录的 apikey.txt
    streamlit run app.py

演示：
    正常调用时，每次 AI 的回答都会自动录制到 demo_cache.json。
    台上打开侧边栏的“离线演示模式”，按排练时的同样操作，就会直接用录制好的回答，不依赖网络。
    即使没开离线模式，接口失败时也会自动回退到录制结果（如果有）。
"""

import base64
import hashlib
import html as htmllib
import json
import re
import time
from datetime import date, datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

MODEL = "qwen3.8-max"
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"  # 国际站；国内站 key 用 https://dashscope.aliyuncs.com/compatible-mode/v1
KEY_FILE = Path(__file__).parent / "apikey.txt"
CACHE_FILE = Path(__file__).parent / "demo_cache.json"

# 演示场景：每个场景对应一个项目；附带执行者/知识库的预设问题（台上点按钮即可，不用现场打字）
SCENARIOS = {
    "计算器": {
        "project": "计算器 App",
        "task": "帮我做一个计算器，平时算账用的，要比手机自带的那个好用一点。最好这周能先看到个样子。",
        "worker_qs": ["需要支持键盘输入吗？", "历史记录要保存多久？"],
        "kb_qs": ["计算器最后定了哪些功能？", "哪些功能明确不做？"],
    },
    "团队任务看板": {
        "project": "团队任务看板",
        "task": "做一个团队用的任务看板，大家能看到谁在做什么、做到哪了，别再天天在群里问进度。",
        "worker_qs": ["任务卡片需要设置截止日期吗？", "要不要区分管理员和普通成员？"],
        "kb_qs": ["看板分哪几列？", "还有哪些问题没定？"],
    },
    "活动报名页": {
        "project": "技术分享会报名",
        "task": "下个月办一场技术分享会，做个报名页面。名额有限，报满了要能自动停止报名，最好还能看到报名名单。",
        "worker_qs": ["报名需要填哪些信息？", "报满后要不要开候补？"],
        "kb_qs": ["名额上限是多少？", "截止日期是哪天？"],
    },
}

LANG_RULE = "用与任务描述相同的语言回答。"

PAGES = ["需求方", "执行者", "项目知识库"]
STEPS = ["发布任务", "AI 追问", "任务简报", "需求 Demo", "执行答疑", "知识沉淀"]
COVER_COLORS = ["#4f46e5", "#0f766e", "#c2410c", "#be185d", "#1d4ed8", "#7c3aed", "#15803d", "#b45309"]
WAITING = "⏳ 待需求方回答"
ALL_KB = "__all__"  # 跨项目问答在 kb_chat 里的标记
ALL_QS = ["哪些项目还有问题在等需求方回答？", "哪个项目离截止日期最近？", "各个项目都定了哪些关键决定？"]

# ---------- JSON schemas（通过 tool use 强制结构化输出） ----------

ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "任务的简短标题，15字以内"},
        "understanding": {"type": "string", "description": "用一两句话复述你对任务目标的理解"},
        "deadline": {"type": "string", "description": "任务提到的截止日期，按今天的日期推算成 YYYY-MM-DD；没提到就留空"},
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "detail": {"type": "string"}},
                "required": ["title", "detail"],
            },
            "description": "初步拆解出的子任务，3到6个",
        },
        "known": {"type": "array", "items": {"type": "string"}, "description": "任务中已经明确的信息"},
        "questions": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "why": {"type": "string", "description": "为什么要问，不问清楚会有什么影响"},
                    "options": {"type": "array", "items": {"type": "string"}, "description": "可选答案，没有就留空"},
                },
                "required": ["question", "why", "options"],
            },
            "description": "最关键的澄清问题，最多3个，按重要性排序",
        },
        "needs_meeting": {"type": "boolean", "description": "是否涉及多方决策或利益冲突，需要当面沟通"},
        "meeting_reason": {"type": "string"},
    },
    "required": ["title", "understanding", "deadline", "subtasks", "known", "questions", "needs_meeting", "meeting_reason"],
}

BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "in_scope": {"type": "array", "items": {"type": "string"}},
        "out_of_scope": {"type": "array", "items": {"type": "string"}},
        "acceptance_criteria": {"type": "array", "items": {"type": "string"}, "description": "可检查的验收标准"},
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
                "properties": {"decision": {"type": "string"}, "source": {"type": "string", "description": "来自哪个问题的回答"}},
                "required": ["decision", "source"],
            },
        },
        "open_issues": {"type": "array", "items": {"type": "string"}, "description": "仍未确定的问题"},
    },
    "required": ["goal", "in_scope", "out_of_scope", "acceptance_criteria", "subtasks", "decisions", "open_issues"],
}

WORKER_SCHEMA = {
    "type": "object",
    "properties": {
        "can_answer": {"type": "boolean", "description": "简报和已有问答里是否有足够依据回答"},
        "answer": {"type": "string", "description": "能回答时的答案；不能回答时说明缺什么"},
        "source": {"type": "string", "description": "答案依据：简报的哪一部分，或哪条历史问答"},
        "question_for_requester": {"type": "string", "description": "不能回答时，整理成一个清楚的问题转给需求方"},
    },
    "required": ["can_answer", "answer", "source", "question_for_requester"],
}

KB_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean", "description": "项目文档和时间线里是否有足够依据回答"},
        "answer": {"type": "string", "description": "答案；依据不足时说明缺什么"},
        "sources": {"type": "array", "items": {"type": "string"}, "description": "引用的段落，写文档里的标题或问题原文"},
    },
    "required": ["found", "answer", "sources"],
}

ALL_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean", "description": "所有项目文档里是否有足够依据回答"},
        "answer": {"type": "string", "description": "答案，说明信息来自哪个项目；依据不足时说明缺什么"},
        "sources": {"type": "array", "items": {"type": "string"}, "description": "引用，格式：项目名 · 段落标题或问题原文"},
        "projects": {"type": "array", "items": {"type": "string"}, "description": "答案涉及的项目名，必须和文档里的项目标题完全一致"},
    },
    "required": ["found", "answer", "sources", "projects"],
}

# ---------- Qwen 调用（OpenAI 兼容接口）+ 录制 / 回放 ----------


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
    """先查录制；离线模式直接回放，否则真实调用并录下来，失败时回退到录制。"""
    # 去掉日期行，这样排练录下的回答第二天也能回放
    norm = re.sub(r"今天是 \d{4}-\d{2}-\d{2}。\n?", "", content)
    cache, key = get_cache(), hashlib.sha1(f"{name}\n{system}\n{norm}".encode("utf-8")).hexdigest()
    if st.session_state.get("replay") and key in cache:
        time.sleep(0.8)  # 留一点“思考”的节奏，别让回放显得太假
        return json.loads(json.dumps(cache[key]))
    try:
        data = produce()
    except Exception:
        if key not in cache:
            raise
        st.toast("接口调用失败，已切换到录制好的回答")
        return json.loads(json.dumps(cache[key]))
    cache[key] = data
    save_cache()
    return data


def call_tool(system: str, content: str, name: str, schema: dict) -> dict:
    def produce():
        resp = get_client().chat.completions.create(
            model=MODEL,
            max_tokens=3000,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            tools=[{"type": "function", "function": {"name": name, "description": "按指定结构输出结果", "parameters": schema}}],
            tool_choice={"type": "function", "function": {"name": name}},
            extra_body={"enable_thinking": False},  # 思考模式下不支持强制 tool_choice
        )
        calls = resp.choices[0].message.tool_calls
        if not calls:
            raise RuntimeError("模型没有返回结构化结果")
        return json.loads(calls[0].function.arguments)

    data = cached(name, system, content, produce)
    # 模型偶尔会漏字段，按 schema 补默认值，避免渲染时 KeyError
    defaults = {"array": [], "string": "", "boolean": False}
    for k in schema.get("required", []):
        data.setdefault(k, defaults.get(schema["properties"][k]["type"]))
    return data


def call_text(system: str, content: str, name: str, max_tokens: int = 8000) -> str:
    """长文本输出（比如 HTML）不走 tool use，直接要普通回复，更稳。"""
    def produce():
        resp = get_client().chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            extra_body={"enable_thinking": False},
        )
        return {"text": resp.choices[0].message.content or ""}

    return cached(name, system, content, produce)["text"]


def analyze(task: str) -> dict:
    system = (
        "你是团队里经验丰富的技术负责人，负责在任务开工前把需求理解清楚。"
        "拆解任务，找出模糊点和隐含假设，只问真正影响实现的关键问题，不问显而易见的事。"
        "不要替需求方做假设：信息缺失就提问。" + LANG_RULE
    )
    return call_tool(system, f"今天是 {date.today()}。\n任务描述：\n{task}", "task_analysis", ANALYZE_SCHEMA)


def write_brief(task: str, analysis: dict, answers: dict) -> dict:
    qa = "\n".join(f"问：{q}\n答：{a}" for q, a in answers.items())
    system = (
        "根据任务描述和澄清问答，写一份执行者可以直接开工的任务简报。"
        "每条决定都必须能对应到某个回答；没有依据的内容只能放进未决问题。" + LANG_RULE
    )
    content = f"任务描述：\n{task}\n\n初步理解：{analysis['understanding']}\n\n澄清问答：\n{qa or '（无）'}"
    return call_tool(system, content, "task_brief", BRIEF_SCHEMA)


def make_demo(task: str, brief: dict, feedback: str = "") -> str:
    system = (
        "你是产品原型设计师。根据任务简报做一个可交互的单文件 HTML 原型，让需求方一眼确认“做出来大概是这样”。"
        "要求：只实现简报范围内的核心功能，按钮和输入要真的能用；需要数据时用假数据；"
        "不要访问网络，不要引用任何外部文件、字体或 CDN，CSS 和 JS 全部内联；"
        "样式简洁现代，界面文字与任务描述用同一种语言；整体高度控制在 520px 以内，宽度自适应。"
        "只输出一个 ```html 代码块，不要任何解释。"
    )
    content = f"任务描述：\n{task}\n\n任务简报：\n{brief_md(brief)}"
    if feedback:
        content += f"\n\n需求方看过上一版 Demo 后的修改意见（必须体现）：{feedback}"
    text = call_text(system, content, "demo_html")
    m = re.search(r"```(?:html)?\s*(.*?)```", text, re.S)
    code = (m.group(1) if m else text).strip()
    if "<" not in code:
        raise RuntimeError("模型没有返回 HTML")
    return code


def answer_worker(task: str, brief: dict, history: list, question: str) -> dict:
    hist = "\n".join(f"问：{h['q']}\n答（{h['by']}）：{h['a']}" for h in history) or "（无）"
    system = (
        "你是任务的协调助手，执行者在干活时向你提问。只根据任务简报和历史问答回答，不要编造。"
        "依据不足时 can_answer 设为 false，并把问题整理清楚转给需求方。" + LANG_RULE
    )
    content = f"任务描述：\n{task}\n\n任务简报：\n{json.dumps(brief, ensure_ascii=False, indent=2)}\n\n历史问答：\n{hist}\n\n执行者的问题：{question}"
    return call_tool(system, content, "worker_answer", WORKER_SCHEMA)


def answer_kb(project: str, info: dict, doc_md: str, events: list, question: str) -> dict:
    timeline = "\n".join(f"{e['time']:%Y-%m-%d %H:%M:%S} | {e['task']} | {e['type']} | {e['text']}" for e in events)
    system = (
        "你是项目知识库助手。只根据给出的项目文档和项目时间线回答，不要编造，sources 里列出用到的段落标题或问题原文。"
        "依据不足时 found 设为 false，并说明缺什么。用与问题相同的语言回答。"
    )
    content = (
        f"今天是 {date.today()}。\n项目：{project}\n创建时间：{info['created']:%Y-%m-%d %H:%M}\n"
        f"截止日期：{info['deadline'] or '未设定'}\n\n项目文档：\n{doc_md or '（无）'}\n\n"
        f"项目时间线：\n{timeline or '（无）'}\n\n问题：{question}"
    )
    return call_tool(system, content, "kb_answer", KB_SCHEMA)


def answer_all(question: str) -> dict:
    blocks = "\n\n---\n\n".join(project_doc_md(p) for p in S.projects)
    system = (
        "你是团队的跨项目知识库助手，可以检索所有项目的文档。只根据给出的文档回答，不要编造，"
        "回答里要说明每条信息来自哪个项目。“⏳ 待需求方回答”表示该问题还没有答复。"
        "依据不足时 found 设为 false，并说明缺什么。用与问题相同的语言回答。"
    )
    content = f"今天是 {date.today()}。\n共 {len(S.projects)} 个项目，文档如下：\n\n{blocks}\n\n问题：{question}"
    return call_tool(system, content, "kb_all_answer", ALL_SCHEMA)


# ---------- 状态 ----------

TASK_DEFAULTS = {
    "task": "",
    "task_id": None,  # 发布任务时分配
    "task_title": "",
    "task_project": "",  # 任务发布时所属的项目
    "analysis": None,
    "brief": None,
    "demo": None,  # 需求 Demo 的 HTML
    "demo_ver": 0,  # 每重做一次 +1，用来清空修改意见输入框
    "history": [],  # 沉淀下来的问答：{"q", "a", "by"}
    "pending": [],  # 等需求方回答的问题
    "chat": [],  # 执行者对话：{"role", "content"}
}
KB_DEFAULTS = {
    "projects": {},  # 项目名 -> {"created", "deadline"}
    "docs": [],  # 记录：{"id", "key", "project", "task_id", "task", "type", "title", "content", "time"}
    "events": [],  # 时间线：{"time", "project", "task", "type", "text"}
    "kb_chat": [],  # 知识库问答：{"project", "role", "content"}
    "kb_open": None,  # 知识库里当前打开的项目；None 表示看封面列表
    "task_seq": 0,
}
TASK_WIDGET_PREFIXES = ("opt_", "ans_", "pend_", "btn_", "sub_", "demo_fb")


def fresh(v):
    return v.copy() if isinstance(v, (list, dict)) else v


for k, v in {**TASK_DEFAULTS, **KB_DEFAULTS}.items():
    st.session_state.setdefault(k, fresh(v))
st.session_state.setdefault("project", "演示项目")
st.session_state.setdefault("page", PAGES[0])
S = st.session_state


def new_task():
    """开始新任务：清掉当前任务和它的控件状态，知识库保留。"""
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
    S.page = "项目知识库"


def fill_example(name: str):
    """填入示例：如果手上已有任务，先归档它，示例自动归到自己的项目下。"""
    if S.task_id is not None:
        new_task()
    sc = SCENARIOS[name]
    S.task = sc["task"]
    S.project = sc["project"]


def scenario_of(task: str):
    return next((v for v in SCENARIOS.values() if v["task"] == task), None)


def scenario_by_project(proj: str):
    return next((v for v in SCENARIOS.values() if v["project"] == proj), None)


# ---------- 知识库沉淀 ----------


def log(kind: str, text: str):
    S.events.append({"time": datetime.now(), "project": S.task_project, "task": S.task_title, "type": kind, "text": text})


def save_doc(key: str, kind: str, title: str, content: str):
    """沉淀一条记录；同一个 key 再次保存时覆盖旧版本（比如重新生成简报、问题得到回答）。"""
    doc = {"key": key, "project": S.task_project, "task_id": S.task_id, "task": S.task_title, "type": kind,
           "title": title, "content": content, "time": datetime.now()}
    for d in S.docs:
        if d["key"] == key:
            d.update(doc)
            return
    S.docs.append({"id": len(S.docs) + 1, **doc})


def ask_key(q: str) -> str:
    """执行中的提问用问题文本做 key：转给需求方时先记一条“待回答”，回答后原地更新。"""
    return f"{S.task_id}-ask-{hashlib.md5(q.encode('utf-8')).hexdigest()[:8]}"


def on_analyzed(a: dict):
    if S.task_id is None:
        S.task_seq += 1
        S.task_id = S.task_seq
        S.task_project = S.project.strip() or "未命名项目"
    S.task_title = a["title"] or S.task[:15]
    info = S.projects.setdefault(S.task_project, {"created": datetime.now(), "deadline": None})
    try:
        ddl = date.fromisoformat(a["deadline"])
        if not info["deadline"] or ddl > info["deadline"]:
            info["deadline"] = ddl
    except ValueError:
        pass
    known = "；".join(a["known"]) or "无"
    save_doc(f"{S.task_id}-task", "任务", S.task_title,
             f"{S.task}\n\nAI的理解：{a['understanding']}\n\n已明确：{known}\n\n截止：{a['deadline'] or '未提到'}")
    log("发布任务", a["understanding"])


def brief_md(b: dict) -> str:
    parts = [
        f"**目标**：{b['goal']}",
        "**范围内**：" + "；".join(b["in_scope"]),
        "**明确不做**：" + ("；".join(b["out_of_scope"]) or "无"),
        "**验收标准**：" + "；".join(b["acceptance_criteria"]),
        "**子任务**：" + "；".join(f"{t['title']}（{t['detail']}）" for t in b["subtasks"]),
        "**已确认的决定**：" + ("；".join(f"{d['decision']}（来源：{d['source']}）" for d in b["decisions"]) or "无"),
        "**未决问题**：" + ("；".join(b["open_issues"]) or "无"),
    ]
    return "\n\n".join(parts)


def project_docs(proj: str) -> list:
    return [d for d in S.docs if d["project"] == proj]


def project_doc_md(proj: str) -> str:
    """把一个项目的所有记录拼成一份文档：按任务分节，每节含需求、简报、Demo 和全部问答。"""
    info = S.projects[proj]
    docs = project_docs(proj)
    meta = f"创建于 {info['created']:%Y-%m-%d %H:%M} · 截止 {info['deadline'] or '未设定'}"
    if docs:
        meta += f" · 最近更新 {max(d['time'] for d in docs):%m-%d %H:%M}"
    lines = [f"### {proj}", meta]
    for n, tid in enumerate(dict.fromkeys(d.get("task_id") for d in docs), 1):
        group = [d for d in docs if d.get("task_id") == tid]
        main = {d["type"]: d for d in group if d["type"] in ("任务", "简报", "Demo")}
        lines.append(f"#### 任务 {n}：{group[-1]['task']}")
        if "任务" in main:
            lines += ["##### 需求与理解", main["任务"]["content"]]
        if "简报" in main:
            lines += ["##### 任务简报", main["简报"]["content"]]
        if "Demo" in main:
            lines += ["##### 需求 Demo", f"已生成（{main['Demo']['time']:%m-%d %H:%M}），可在本页“需求 Demo”里预览或下载。"]
        qas = [d for d in group if d["type"] == "问答"]
        if qas:
            lines.append(f"##### 问答记录（{len(qas)}）")
            lines += [f"**问：{d['title']}**  \n{d['content']}  \n<small>{d['time']:%m-%d %H:%M}</small>" for d in qas]
    return "\n\n".join(lines)


def on_check(title: str, key: str):
    log("完成子任务" if S[key] else "取消完成", title)


# ---------- 渲染工具 ----------

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
    if any(e["type"] in ("AI直接回答", "需求方回答") for e in cur):
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
    st.markdown(f"**目标**：{b['goal']}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**范围内**")
        bullets(b["in_scope"])
    with c2:
        st.markdown("**明确不做**")
        bullets(b["out_of_scope"] or ["（无）"])
    st.markdown("**验收标准**")
    bullets(b["acceptance_criteria"])
    st.markdown("**子任务**")
    for i, t in enumerate(b["subtasks"]):
        if checkable:
            st.checkbox(f"{t['title']}：{t['detail']}", key=f"sub_{i}", on_change=on_check, args=(t["title"], f"sub_{i}"))
        else:
            st.markdown(f"{i + 1}. **{t['title']}**：{t['detail']}")
    if b["decisions"]:
        st.markdown("**已确认的决定**")
        for d in b["decisions"]:
            st.markdown(f"- {d['decision']}  \n  <small>来源：{d['source']}</small>", unsafe_allow_html=True)
    if b["open_issues"]:
        st.markdown("**未决问题**")
        bullets(b["open_issues"])


def show_demo(code: str, height: int = 540):
    # Demo 是模型生成的 HTML：用 data URL 嵌入，iframe 是独立的来源，碰不到应用本身
    src = "data:text/html;charset=utf-8;base64," + base64.b64encode(code.encode("utf-8")).decode("ascii")
    if hasattr(st, "iframe"):
        st.iframe(src, height=height)
    else:  # 老版本 Streamlit
        components.iframe(src, height=height, scrolling=True)


def preset_buttons(questions: list, prefix: str):
    """把预设问题渲染成一排按钮，返回被点中的那个。"""
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
    qas = [d for d in docs if d["type"] == "问答"]
    waiting = sum(d["content"].startswith(WAITING) for d in qas)
    latest = max(docs, key=lambda d: d["time"]) if docs else None
    color = COVER_COLORS[int(hashlib.md5(proj.encode("utf-8")).hexdigest(), 16) % len(COVER_COLORS)]
    ddl = f"截止 {info['deadline']:%m-%d}" if info["deadline"] else "未设截止"
    tags = [f'<span class="pill">{n_tasks} 个任务</span>', f'<span class="pill">{len(qas)} 条问答</span>']
    if any(d["type"] == "Demo" for d in docs):
        tags.append('<span class="pill">含 Demo</span>')
    if waiting:
        tags.append(f'<span class="pill">{WAITING} {waiting}</span>')
    sub = f"最近：{latest['task']}" if latest else "还没有记录"
    return (
        f'<div class="cover"><div class="cover-band" style="background:{color}">'
        f'<b>{htmllib.escape(proj[:1])}</b><i>{ddl}</i></div>'
        f'<div class="cover-body"><div class="cover-title">{htmllib.escape(proj)}</div>'
        f'<div class="cover-sub">{htmllib.escape(sub)}</div>{"".join(tags)}</div></div>'
    )


# ---------- 页面 ----------

st.set_page_config(page_title="需求澄清助手", page_icon="🧭", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.radio("视角", PAGES, key="page")
    st.text_input("当前项目", key="project", disabled=S.task_id is not None, help="新任务归到这个项目下；开始新任务后可以切换")
    if S.task_id is not None:
        st.caption(f"当前任务：{S.task_title}")
    if S.pending:
        st.warning(f"有 {len(S.pending)} 个执行者的问题等需求方回答")
    n_qa = sum(d["type"] == "问答" for d in S.docs)
    st.caption(f"知识库：{len(S.projects)} 份项目文档，{n_qa} 条问答，{len(S.events)} 条事件")
    c1, c2 = st.columns(2)
    c1.button("开始新任务", on_click=new_task, help="当前任务归档，知识库保留")
    c2.button("清空全部", on_click=clear_all)
    st.divider()
    st.toggle("离线演示模式", key="replay",
              help="优先使用录制好的回答，不依赖网络。正常调用时会自动录制；接口失败时也会自动回退到录制结果。")
    st.caption(f"已录制 {len(get_cache())} 条 AI 回答")

st.title("🧭 需求澄清助手")
st.markdown(
    '<div class="subtitle">一句模糊的需求 → AI 追问关键点 → 简报 + 能点的 Demo → 干活时自动答疑 → 每个项目沉淀成一份文档</div>',
    unsafe_allow_html=True,
)
render_steps()

page = S.page

if page == "需求方":
    st.subheader("1. 发布任务")
    c1, c2 = st.columns([4, 1])
    pick = c1.selectbox("示例场景", ["自己写一个"] + list(SCENARIOS), key="scenario_pick", label_visibility="collapsed")
    c2.button("填入示例", disabled=pick not in SCENARIOS, use_container_width=True,
              on_click=fill_example, args=(pick,))
    S.task = st.text_area("用自然语言描述你要做的事，越随意越好", value=S.task, height=110)

    if st.button("让 AI 理解任务", type="primary", disabled=not S.task.strip()):
        with st.spinner("AI 正在拆解任务、寻找模糊点……"):
            try:
                S.analysis, S.brief, S.demo = analyze(S.task), None, None
                on_analyzed(S.analysis)
                ok = True
            except Exception as e:
                st.error(f"调用失败：{e}")
                ok = False
        if ok:
            st.rerun()  # 放在 try 外面：st.rerun 靠抛异常实现，不能被 except 吞掉

    a = S.analysis
    if a:
        st.subheader("2. AI 的理解")
        st.info(a["understanding"])
        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            label("初步拆解")
            for i, t in enumerate(a["subtasks"], 1):
                st.markdown(f"{i}. **{t['title']}**：{t['detail']}")
        with c2, st.container(border=True):
            label("已经明确的信息")
            bullets(a["known"] or ["（暂无）"])
        if a["needs_meeting"]:
            st.warning(f"建议当面沟通：{a['meeting_reason']}")

        st.markdown(
            f'### 3. AI 的追问 <span class="pill">只问影响实现的 {len(a["questions"])} 个问题</span>',
            unsafe_allow_html=True,
        )
        answers = {}
        for i, q in enumerate(a["questions"]):
            with st.container(border=True):
                st.markdown(f"**Q{i + 1}. {q['question']}**")
                st.caption(f"为什么问：{q['why']}")
                choice = None
                if q["options"]:
                    choice = st.radio("选一个", q["options"] + ["其他 / 不确定"], key=f"opt_{i}", horizontal=True,
                                      label_visibility="collapsed")
                text = st.text_input("补充说明（可选）", key=f"ans_{i}", placeholder="补充说明（可选）",
                                     label_visibility="collapsed")
                if choice and choice != "其他 / 不确定":
                    answers[q["question"]] = f"{choice}。{text}".strip("。") if text else choice
                else:
                    answers[q["question"]] = text or "不确定，之后再定"

        if st.button("生成任务简报", type="primary"):
            with st.spinner("正在整理成可直接开工的简报……"):
                try:
                    S.brief, S.demo = write_brief(S.task, a, answers), None
                    S.history = [{"q": q, "a": ans, "by": "需求方"} for q, ans in answers.items()]
                    save_doc(f"{S.task_id}-brief", "简报", f"{S.task_title} · 任务简报", brief_md(S.brief))
                    for i, (q, ans) in enumerate(answers.items()):
                        save_doc(f"{S.task_id}-clar-{i}", "问答", q, f"答（需求方，开工前澄清）：{ans}")
                    log("生成简报", S.brief["goal"])
                    ok = True
                except Exception as e:
                    st.error(f"调用失败：{e}")
                    ok = False
            if ok:
                st.rerun()

    if S.brief:
        b = S.brief
        st.subheader("4. 任务简报")
        m = st.columns(4)
        m[0].metric("澄清的问题", len(a["questions"]) if a else 0)
        m[1].metric("确认的决定", len(b["decisions"]))
        m[2].metric("验收标准", len(b["acceptance_criteria"]))
        m[3].metric("仍未决定", len(b["open_issues"]))
        left, right = st.columns([1, 2])
        with left, st.container(border=True):
            label("之前：需求方的一句话")
            st.markdown(f"> {S.task}")
        with right, st.container(border=True):
            label("之后：执行者可以直接开工的简报")
            render_brief(b)

        st.markdown('### 5. 需求 Demo <span class="pill">先看一眼：做出来是这样吗？</span>', unsafe_allow_html=True)
        feedback = ""
        if not S.demo:
            st.caption("根据简报生成一个能点的简单原型。需求方看图比读文字更快发现“不是我要的”。")
            gen = st.button("生成需求 Demo", type="primary")
        else:
            with st.container(border=True):
                show_demo(S.demo)
            c1, c2, c3 = st.columns([3, 1, 1])
            feedback = c1.text_input("哪里不对？", key=f"demo_fb_{S.demo_ver}", label_visibility="collapsed",
                                     placeholder="哪里不对？比如：按钮再大一点，要能看到历史记录")
            gen = c2.button("按意见重做", disabled=not feedback.strip(), use_container_width=True)
            c3.download_button("下载 HTML", S.demo, file_name=f"{S.task_title or 'demo'}.html", mime="text/html",
                               use_container_width=True)
        if gen:
            with st.spinner("正在把简报变成一个能点的原型……"):
                try:
                    S.demo = make_demo(S.task, b, feedback.strip())
                    S.demo_ver += 1
                    save_doc(f"{S.task_id}-demo", "Demo", f"{S.task_title} · 需求 Demo", S.demo)
                    if feedback.strip():
                        S.history.append({"q": "对需求 Demo 的修改意见", "a": feedback.strip(), "by": "需求方"})
                        save_doc(f"{S.task_id}-demofb-{S.demo_ver}", "问答", "对需求 Demo 的修改意见",
                                 f"答（需求方，看过 Demo 后）：{feedback.strip()}")
                        log("修改 Demo", feedback.strip())
                    else:
                        log("生成 Demo", "根据简报生成了可交互原型")
                    ok = True
                except Exception as e:
                    st.error(f"调用失败：{e}")
                    ok = False
            if ok:
                st.rerun()

        st.button("切换到执行者视角 →", type="primary", on_click=goto, args=("执行者",))

    if S.pending:
        st.subheader("执行者的问题")
        for i, q in enumerate(list(S.pending)):
            with st.container(border=True):
                st.markdown(f"**{q}**")
                reply = st.text_input("你的回答", key=f"pend_{i}_{q}")
                if st.button("回答并记录", key=f"btn_{i}_{q}") and reply:
                    S.history.append({"q": q, "a": reply, "by": "需求方"})
                    save_doc(ask_key(q), "问答", q, f"答（需求方，执行中提问）：{reply}")
                    log("需求方回答", f"{q} → {reply}")
                    S.chat.append({"role": "assistant", "content": f"需求方回答了：**{q}**\n\n{reply}"})
                    S.pending.remove(q)
                    st.rerun()

elif page == "执行者":
    if not S.brief:
        st.info("还没有任务简报。先在“需求方”视角发布任务。")
        st.button("去发布任务 →", on_click=goto, args=("需求方",))
        st.stop()

    left, right = st.columns([1, 1])
    with left:
        with st.container(border=True):
            label("我的任务")
            render_brief(S.brief, checkable=True)
        if S.demo:
            with st.expander("需求 Demo（需求方确认过的样子）"):
                show_demo(S.demo, height=500)
    with right:
        st.markdown("#### 干活时有问题就问")
        st.caption("简报里有答案就直接回答并给出依据；没有就整理成问题转给需求方，不瞎编。")
        for m in S.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"], unsafe_allow_html=True)
        sc = scenario_of(S.task)
        preset = preset_buttons(sc["worker_qs"] if sc else [], "wq")
        if q := (st.chat_input("比如：需要支持键盘输入吗？") or preset):
            S.chat.append({"role": "user", "content": q})
            try:
                with st.spinner("查简报和历史问答……"):
                    r = answer_worker(S.task, S.brief, S.history, q)
                if r["can_answer"]:
                    msg = f"{r['answer']}\n\n<small>依据：{r['source']}</small>"
                    save_doc(ask_key(q), "问答", q, f"答（AI 根据简报，依据：{r['source']}）：{r['answer']}")
                    log("AI直接回答", f"{q} → {r['answer']}")
                else:
                    rq = r["question_for_requester"]
                    S.pending.append(rq)
                    save_doc(ask_key(rq), "问答", rq, f"{WAITING}（执行者提问：{q}）")
                    log("转给需求方", rq)
                    msg = f"简报里没有足够依据：{r['answer']}\n\n↪ 已转给需求方：**{rq}**"
            except Exception as e:
                msg = f"调用失败：{e}"
            S.chat.append({"role": "assistant", "content": msg})
            st.rerun()
        if S.chat:
            st.button("看看知识库沉淀了什么 →", on_click=open_project, args=(S.task_project,))

else:  # 项目知识库
    if not S.projects:
        st.info("知识库还是空的。在“需求方”视角发布任务、生成简报后，过程会自动沉淀到这里。")
        st.button("去发布任务 →", on_click=goto, args=("需求方",))
        st.stop()

    # ---- 封面列表 ----
    if S.kb_open not in S.projects:
        gallery, chat = st.columns([2, 1], gap="large")
        with gallery:
            st.subheader("全部项目")
            st.caption("每个项目一份文档：需求、简报、Demo 和所有提问都在里面，随过程自动更新。点开看详情。")
            names = sorted(S.projects, key=lambda p: S.projects[p]["created"], reverse=True)
            per_row = 3
            for start in range(0, len(names), per_row):
                cols = st.columns(per_row)
                for col, name in zip(cols, names[start:start + per_row]):
                    with col:
                        st.markdown(cover_html(name), unsafe_allow_html=True)
                        st.button("查看详情", key=f"open_{name}", on_click=open_project, args=(name,),
                                  use_container_width=True)

        with chat:
            st.subheader("问所有项目")
            st.caption("跨项目检索：谁定了什么、哪里还卡着、哪个项目快到期。")
            msgs = [m for m in S.kb_chat if m["project"] == ALL_KB]
            with st.container(height=420):
                if not msgs:
                    st.caption("还没有提问，试试下面的问题。")
                for i, m in enumerate(msgs):
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"], unsafe_allow_html=True)
                        for p in m.get("projects", []):
                            if p in S.projects:
                                st.button(f"打开「{p}」", key=f"all_open_{i}_{p}", on_click=open_project, args=(p,))
            preset = None
            for i, pq in enumerate(ALL_QS):
                if st.button(pq, key=f"aq_{i}", use_container_width=True):
                    preset = pq
            if q := (st.chat_input("问所有项目，比如：谁负责的需求还没定？") or preset):
                S.kb_chat.append({"project": ALL_KB, "role": "user", "content": q})
                projects = []
                try:
                    with st.spinner("正在检索所有项目……"):
                        r = answer_all(q)
                    cites = "；".join(str(s) for s in r["sources"] if s)
                    msg = r["answer"] if r["found"] else f"所有项目里都没有足够依据：{r['answer']}"
                    if cites:
                        msg += f"\n\n<small>引用：{cites}</small>"
                    projects = [p for p in dict.fromkeys(r["projects"]) if p in S.projects]
                except Exception as e:
                    msg = f"调用失败：{e}"
                S.kb_chat.append({"project": ALL_KB, "role": "assistant", "content": msg, "projects": projects})
                st.rerun()
        st.stop()

    # ---- 项目详情 ----
    proj = S.kb_open
    info = S.projects[proj]
    docs = project_docs(proj)
    events = [e for e in S.events if e["project"] == proj]
    tasks = list(dict.fromkeys(e["task"] for e in events))
    doc_md = project_doc_md(proj)

    st.button("← 全部项目", on_click=open_project, args=(None,))
    st.header(proj)
    c = st.columns(5)
    c[0].metric("创建时间", f"{info['created']:%m-%d %H:%M}")
    c[1].metric("最近更新", f"{events[-1]['time']:%m-%d %H:%M}" if events else "—")
    c[2].metric("任务", len(tasks))
    qas = [d for d in docs if d["type"] == "问答"]
    waiting = sum(d["content"].startswith(WAITING) for d in qas)
    c[3].metric("问答记录", len(qas), delta=f"{waiting} 条待回答" if waiting else None, delta_color="off")
    c[4].metric("距截止", f"{(info['deadline'] - date.today()).days} 天" if info["deadline"] else "未设定")
    with st.expander("修改截止日期"):
        info["deadline"] = st.date_input("截止日期（AI 从任务描述里推算，可手动改）", value=info["deadline"], key=f"ddl_{proj}")

    demos = [d for d in docs if d["type"] == "Demo"]
    if demos:
        st.subheader("需求 Demo")
        for d in demos:
            with st.expander(f"任务：{d['task']} · {d['time']:%m-%d %H:%M}", expanded=len(demos) == 1):
                show_demo(d["content"], height=500)

    st.subheader("项目文档")
    c1, c2 = st.columns([3, 1])
    c1.caption("需求、简报和所有提问都在这一份里，随过程自动更新。")
    c2.download_button("下载 Markdown", doc_md, file_name=f"{proj}.md", mime="text/markdown", use_container_width=True)
    with st.container(border=True):
        st.markdown("\n\n".join(doc_md.split("\n\n")[2:]), unsafe_allow_html=True)  # 标题和概况上面已经显示了

    st.subheader("问知识库")
    st.caption("新人接手、隔周回看时，不用翻聊天记录，直接问。")
    for m in S.kb_chat:
        if m["project"] == proj:
            with st.chat_message(m["role"]):
                st.markdown(m["content"], unsafe_allow_html=True)
    sc = scenario_by_project(proj)
    preset = preset_buttons(sc["kb_qs"] if sc else [], "kq")
    if q := (st.chat_input("比如：最后定了哪些功能？哪些问题还没定？") or preset):
        S.kb_chat.append({"project": proj, "role": "user", "content": q})
        try:
            with st.spinner("正在查找……"):
                r = answer_kb(proj, info, doc_md, events, q)
            cites = "；".join(str(s) for s in r["sources"] if s)
            msg = r["answer"] if r["found"] else f"知识库里没有足够依据：{r['answer']}"
            if cites:
                msg += f"\n\n<small>引用：{cites}</small>"
        except Exception as e:
            msg = f"调用失败：{e}"
        S.kb_chat.append({"project": proj, "role": "assistant", "content": msg})
        st.rerun()

    st.subheader("项目时间线")
    if events:
        df = pd.DataFrame(events)
        df["时间"] = df["time"].dt.strftime("%m-%d %H:%M:%S")
        span = df.groupby("task", sort=False)["time"].agg(["min", "max"]).reset_index()
        y = alt.Y("task:N", title=None, sort=tasks)
        x_scale = alt.Scale(type="utc")  # 时间不带时区，按原样显示，避免浏览器再换算一次
        spans = alt.Chart(span).mark_rule(strokeWidth=8, opacity=0.2).encode(
            x=alt.X("min:T", scale=x_scale, title=None), x2="max:T", y=y
        )
        dots = alt.Chart(df).mark_circle(size=160, opacity=0.9).encode(
            x=alt.X("time:T", scale=x_scale),
            y=y,
            color=alt.Color("type:N", title="事件"),
            tooltip=[alt.Tooltip("时间:N"), alt.Tooltip("type:N", title="事件"), alt.Tooltip("text:N", title="内容")],
        )
        st.altair_chart((spans + dots).properties(height=80 + 50 * len(tasks)), width="stretch")
        with st.expander("事件明细"):
            table = df[["时间", "task", "type", "text"]].rename(columns={"task": "任务", "type": "事件", "text": "内容"})
            st.dataframe(table.iloc[::-1], hide_index=True, width="stretch")
    else:
        st.caption("还没有事件")

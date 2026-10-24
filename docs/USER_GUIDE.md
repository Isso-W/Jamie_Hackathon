# User guide

Requirement Clarifier turns a rough request into something the team can build without guessing. This guide walks through every screen. For setup, see the [README](../README.md); for how it works inside, see the [technical documentation](TECHNICAL.md).

## Key ideas

| Term | Meaning |
| --- | --- |
| **Project** | A group of tasks with one shared knowledge base and one cast of stakeholders. |
| **Task** | One request, e.g. "Build me a calculator for everyday bookkeeping". |
| **Stakeholders** | The people whose wishes matter for a project. The AI suggests them from the task; you can edit them. |
| **Owner** | The one stakeholder who settles conflicts and can end the questioning. |
| **Acting as** | Whose role you are playing right now. One person can play every stakeholder, which is handy for demos. |
| **Worker** | The person who builds the task and asks questions while working. |
| **Q / W / D / M** | References to where a decision came from: an answer to a **Q**uestion, a stakeholder's **W**ish, the owner's **D**ecision on a conflict, or a **M**eeting. Briefs, prototypes and change cards cite them, e.g. "Q3 Anna". |

The AI writes in the language of your task. Everything stays in your browser session: refreshing the page starts over.

## Requirements

**To run the app**

| What | Version / detail |
| --- | --- |
| Python | 3.10 or newer (tested with 3.12) |
| Python packages | `pip install -r requirements.txt` installs `streamlit` (tested with 1.65) and `openai` |
| Node.js | 22.19 or newer (tested with 24), for the pi agent |
| pi agent | `npm install` in the `harness/` folder |
| Qwen API key | From Alibaba Cloud Model Studio, saved in `apikey.txt` next to `app.py`. International-site keys work as-is; mainland China keys need the endpoint change described in the README. |
| Reson8 API key | Optional, only for meeting transcription. Save it in `reson8key.txt`, or set `RESON8_API_KEY`. |

**To use it**

- **Browser:** a current Chrome, Edge, Firefox or Safari. Recording a meeting needs microphone permission for the page.
- **Network:** live AI calls need internet access to Alibaba Cloud Model Studio and, for meetings, `api.reson8.dev`. Without network, use [offline demo mode](#offline-demo-mode) to replay a rehearsal.
- **Languages:** tasks can be written in any language Qwen handles; the AI answers in the same language. Meeting recordings must be in one of Reson8's ten languages: English, German, French, Spanish, Italian, Dutch, Portuguese, Polish, Swedish or Frisian. For other languages, paste a transcript.
- **Meeting files:** wav, mp3, m4a, ogg, flac, webm or mp4.

## The screen

The **sidebar** has:

- **View**: *Stakeholders*, *Worker* or *Project Knowledge Base*.
- **Project**: the project new tasks go into. It's locked while a task is in progress.
- **Acting as** (on the Stakeholders view): the stakeholder you are answering for.
- **New task** archives the current task and keeps the knowledge base; **Clear all** removes everything.
- **⚙️ Settings**: the AI engine (*pi* or *direct*), the model, **Offline demo mode**, and how many answers are recorded.

The **progress bar** at the top shows where the current task is: *Post task → AI questions → Task brief → Requirement demo → Worker Q&A → Knowledge captured*.

## 1. Post a task (Stakeholders view)

1. Type what you need in plain words, or pick an example from the list and click **Fill in example**.
2. Click **Let AI understand the task**.

The AI breaks the task down and opens the first round of questions. The **📝 Task** fold at the top shows its first breakdown, what is already clear and whether it suggests talking face to face. It also has the **stakeholder table**: edit names, roles and what each person cares about, add or remove rows, and tick the owner. A project keeps its stakeholders for later tasks.

## 2. Answer the AI's questions

Each round shows up to three questions. Each one is addressed to one stakeholder (the *for …* tag) and says why it matters.

- If you are acting as that stakeholder, pick an answer. **★** marks the AI's recommendation; **Other…** lets you type your own, and leaving it empty means "not sure".
- Otherwise click **Answer as …** to switch to that person.
- Click **Submit …'s answers**. The round waits until every stakeholder has answered, then the AI checks what is still open and asks follow-ups.

**⚖️ Conflict cards** appear when stakeholders want contradicting things. The card shows who wants what and two or three resolutions, with the recommended compromise first; hover a resolution to see its trade-off. Only the owner can decide: pick one, or use **Other…** to decide in your own words.

The owner can click **Enough, write the brief** at any time; anything still open goes into the brief as an open issue. The questioning also stops by itself after four rounds.

**Wishes:** at the bottom of the page, any stakeholder can add a wish at any time ("It must work on my phone"). During questioning it is considered in the next round; after questioning has finished, the AI reacts straight away and may ask follow-ups or raise a conflict.

## 3. Hold a meeting (optional)

Open **🎙️ Meeting notes** below the current step. It opens by itself when the AI suggested a face-to-face talk.

1. Choose a source:
   - **🎙️ Record**: record the meeting in the browser.
   - **📁 Upload**: upload a recording (wav, mp3, m4a, ogg, flac, webm, mp4), or click **Use the sample meeting**.
   - **📝 Paste transcript**: paste text instead, e.g. for a meeting in a language Reson8 doesn't support.
2. Click **Transcribe with Reson8**. You get a transcript with timestamps and speakers.
3. Click **Write the minutes**. The AI matches speakers to stakeholders and lists the summary, decisions, new wishes, open questions and action items. It records only what was said.
4. Click **Add to the task**. Decisions and open questions become **M** entries and new wishes become **W** entries; then the AI checks whether the meeting raises a conflict or new questions. The minutes and the transcript are saved in the knowledge base.

Reson8 transcribes English, German, French, Spanish, Italian, Dutch, Portuguese, Polish, Swedish and Frisian. It doesn't support Chinese.

## 4. The task brief

Click **Write the task brief** once questioning is done. The brief shows the request **before** (the one-liner) and **after**: goal, decisions with their source, open issues, and a *Details* fold with scope, acceptance criteria and subtasks.

The green banner counts the issues settled before a line of code and estimates the rework that saved (2 hours per issue is an assumption, not a measurement).

Click **📤 Export as ticket** to get the brief as a **GitHub issue** (Markdown with task lists) or in **Jira** markup. Copy it with the copy button, or download it.

## 5. The clickable demo

Click **Build clickable demo**. The pi agent builds an interactive prototype of the brief in its workspace. A line under the demo shows what pi did, e.g. "write index.html → edit index.html ×2".

- Try it right in the page. Fake data stands in for real data.
- Open **Built from your answers** at the bottom of the demo to see which answer each feature comes from.
- Type what's wrong ("bigger buttons, show history") and click **Redo with feedback**. pi edits the existing prototype rather than starting over.
- **Download HTML** saves the prototype as one file that works offline.

## 6. When someone changes their mind

After the brief is written, any new wish, meeting or answer shows a warning: *New answers, wishes or meeting notes since this brief was written*. Click **Rewrite brief**.

The **📣 Requirement changed** card then shows:

- each change, with **before**, **after** and **because of** (e.g. "M1 Anna");
- **Tell**: the stakeholders who should hear about it, and why;
- **rework**: *none*, *small* or *large*.

If a demo exists, it's kept and marked out of date. Click **Update the demo for these changes** and pi applies just those changes. The worker sees the same change card on the Worker view.

## 7. Worker view

Switch to **Worker** (or click **Switch to worker view →**).

- **My task** on the left is the brief, with subtasks you can tick off as you go. **📤 Export as ticket** is here too.
- On the right, ask anything: type a question or click a preset one.
  - If the brief or earlier answers cover it, the AI answers and names its basis.
  - If not, it doesn't guess: it turns the question into a clear request and forwards it to the stakeholder who owns that decision.
- Forwarded questions appear at the top of the Stakeholders view (**📨 Worker asks**). Once that stakeholder answers, the answer lands in the worker's chat and is remembered, so the next similar question is answered directly.

## 8. Project knowledge base

Everything above is captured automatically: one living document per project.

**All projects** shows the totals (projects, tasks, Q&A, questions awaiting answers, issues caught before coding, rework avoided), then the projects:

- **🕸️ Map**: a floating map of projects, tasks and records. Hover to explore, drag to move, click to open. Pink dots are meetings; pulsing amber ones are questions still waiting for an answer.
- **🗂️ Cards**: one cover per project, with search, sorting and a **Needs answers** filter.
- **Ask all projects**: questions across every project, e.g. "Which project is closest to its deadline?". Answers name the projects they come from, with buttons to open them.

**Open a project** to see its metrics and five tabs:

| Tab | What's there |
| --- | --- |
| 📋 Overview | The project's map and one card per task: what's answered and what's still open. **Change deadline** is at the bottom. |
| 📄 Document | The whole project document: requirement, brief, demo, meeting minutes, change log and Q&A log. Filter by task and section, or download it as Markdown. |
| 🖥️ Demo | Every prototype of the project, to try or download. |
| 💬 Ask | Questions about this project, answered only from its document and timeline, with sources. |
| 🕒 Timeline | Every event on a timeline chart and in a table. |

## Offline demo mode

Every AI answer and every Reson8 transcription is recorded as you go. Before going on stage:

1. Rehearse the exact steps you will show, with offline mode **off**, so the answers are recorded.
2. On stage, turn on **⚙️ Settings → Offline demo mode** and repeat the same steps. Recorded answers replay without network.
3. For the meeting, upload the same file you rehearsed with (or use the sample meeting). A newly recorded meeting can't be replayed.

Even with offline mode off, a failed AI call falls back to its recording when one exists.

## A 4-minute demo script

1. **The pain (20 s):** "A one-line request, read three ways. We find out after building."
2. **Post the task (30 s):** fill in the *Calculator* example and let the AI understand it.
3. **Grill (60 s):** answer a round as two different stakeholders and settle a conflict card as the owner.
4. **Brief and demo (40 s):** show before/after and the "issues settled before a line of code" banner, then build the demo and click through it.
5. **They change their minds (60 s):** add the sample meeting, write the minutes, add them to the task, rewrite the brief. Show the change card, then **Update the demo for these changes**.
6. **Worker and memory (30 s):** ask one question the brief answers and one it can't; answer the forwarded one as the stakeholder; ask again.
7. **Close (20 s):** export the GitHub issue and show the knowledge-base map and its numbers.

## FAQ

**The page hangs on "Running start_harness()".** The pi sidecar didn't start. Check `harness/harness.log`, or run `node server.mjs` in `harness/` to see the error. Common causes: Node.js older than 22.19, missing `apikey.txt`, or port 8765 taken by another program. You can keep working with **Settings → AI engine → direct**.

**Every call says "Call failed".** Check the Qwen key in `apikey.txt`. A key from mainland China needs the mainland endpoint (see the README).

**"Transcription failed: Reson8 401".** The Reson8 key in `reson8key.txt` is wrong or missing.

**The minutes put everyone under "Speaker 1".** Reson8 couldn't tell the voices apart. This happens with the sample meeting, whose voices are computer-generated; real recordings are separated by speaker. The minutes still attribute decisions when people say who they are ("I'm the requester…").

**The answers ignore a skill I just changed.** You are probably replaying recordings made before the change. Turn off offline mode and run the flow again to re-record.

**Where did my work go after a refresh?** Everything lives in the browser session. Download the project document or the demo HTML to keep them.

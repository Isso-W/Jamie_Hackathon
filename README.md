# 🧭 Requirement Clarifier

**A rough idea → AI grills every stakeholder → brief + clickable demo → answers while you work.**

Most rework starts with a one-line request that everyone reads differently. Requirement Clarifier takes that one-liner and, before anyone writes code:

- **asks the right people the right questions**, round after round, until nothing open would change what gets built;
- **spots conflicts between stakeholders** and proposes compromises for the owner to pick;
- **turns meetings into decisions**: record or upload a meeting, [Reson8](https://reson8.dev) transcribes it, and the AI writes the minutes back into the task;
- writes a **task brief** in which every decision traces back to who said it;
- has a [pi](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) agent **build a clickable HTML prototype** so the requester can confirm "yes, that's it";
- when someone changes their mind, shows **what changed, who to tell and how much rework it costs**, then updates the prototype in place;
- answers the worker's questions from the brief and **forwards only what it can't answer** to the stakeholder who owns it;
- keeps everything in a **project knowledge base** you can browse as a map or simply ask.

## Quick start

**You need:** Python 3.10+ (tested with 3.12), Node.js 22.19+ (tested with 24), a Qwen API key (Alibaba Cloud Model Studio) and, for meeting transcription, a Reson8 API key.

```powershell
pip install -r requirements.txt
cd harness; npm install; cd ..          # the pi agent sidecar
# Put your keys next to app.py (both files are git-ignored):
#   apikey.txt     your Qwen key
#   reson8key.txt  your Reson8 key (optional: only needed for meeting transcription)
streamlit run app.py
```

The app starts the pi sidecar (`harness/server.mjs`) on `127.0.0.1:8765` by itself. If the sidecar can't start, the error is shown in the sidebar and written to `harness/harness.log`, and every AI call falls back to the Qwen API directly.

> **Mainland China keys:** the app uses the international Model Studio endpoint. To use a mainland key, change `BASE_URL` in `app.py` and `baseUrl` in `harness/agent/models.json` to `https://dashscope.aliyuncs.com/compatible-mode/v1`.

## Try it in two minutes

1. In **Stakeholders**, pick the *Calculator* example and click **Fill in example**, then **Let AI understand the task**.
2. Answer the questions (use **Answer as …** to switch between stakeholders) and settle any conflict cards as the owner.
3. Click **Write the task brief**, then **Build clickable demo**.
4. Open **🎙️ Meeting notes**, click **Use the sample meeting** under *Upload* and then **Transcribe with Reson8** → **Write the minutes** → **Add to the task**.
5. Click **Rewrite brief**: the **📣 Requirement changed** card shows the impact. Click **Update the demo for these changes**.
6. Click **📤 Export as ticket** for a GitHub issue or Jira version of the brief.
7. Switch to **Worker** and ask a question, then open the **Project Knowledge Base**.

## Offline demo mode

Every AI answer, including Reson8 transcriptions, is recorded to `demo_cache.json`. Turn on **Settings → Offline demo mode** in the sidebar and repeat the steps from your rehearsal: the recorded answers replay without network. Even with offline mode off, a failed call falls back to its recording when there is one.

## Documentation

| Document | For |
| --- | --- |
| [User guide](docs/USER_GUIDE.md) | Everyone who uses the app: every screen, the meeting and change flows, a demo script, FAQ |
| [Technical documentation](docs/TECHNICAL.md) | Developers: architecture, the pi harness protocol, AI calls, state model, record/replay, extending |

## Repository layout

```
app.py                 Streamlit app: UI, AI calls, state, knowledge base
harness/server.mjs     pi agent sidecar: one fresh pi session per job (HTTP on 127.0.0.1:8765)
harness/agent/         pi model registry (models.json) and auth store
.agents/skills/        Skills inlined into AI prompts (grill-me, keep-it-short)
workspaces/<task>/     Where pi builds each task's prototype (index.html)
samples/               Sample meeting recording for the meeting-notes flow
demo_cache.json        Recorded AI answers for offline replay
app_zh.py              Earlier standalone Chinese version (no pi, no stakeholders); kept for reference
```

## Built with

[Streamlit](https://streamlit.io) · [pi coding agent](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) · [Qwen](https://www.alibabacloud.com/en/product/modelstudio) via Alibaba Cloud Model Studio · [Reson8](https://reson8.dev) speech-to-text

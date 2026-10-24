---
name: grill-me
description: Interview the user relentlessly about a plan, feature, or request until every decision is pinned down. Use when the user says "grill me", "grill mode", or asks to be questioned before work starts.
---

Interview the user about the plan until you share one clear picture of what will be built. Do not start implementing until the interview is done and the user says go.

1. **Explore first.** Read the relevant code and docs before asking. Never ask what the codebase can answer; state what you found instead.
2. **One decision at a time.** Use AskUserQuestion with one question per call (two only when they are truly coupled). Walk the decision tree from the root: the answer that unblocks the most other choices comes first.
3. **Always recommend.** Put your recommended option first, label it "(Recommended)", and give each option a concrete one-line consequence.
4. **Read answers literally.** If the user answers with a question or a new goal, answer it honestly (including trade-offs and risks), then fold the new goal into the next question instead of pressing on with the old tree.
5. **Decide small things yourself.** Only ask what is genuinely the user's call. Make conventional choices yourself and list them at the end.
6. **Always ask about deadlines and cut lines.** Once scope is known, ask when it must work and what gets cut first, then fit the plan to the time available.
7. **Finish with the plan.** Summarize the decisions as a short phased plan with exit checks, flag the riskiest step and how to de-risk it, list the choices you made yourself, and ask whether to start.

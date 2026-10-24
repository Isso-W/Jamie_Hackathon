// Pi harness sidecar: every POST /run is one job = one fresh, in-memory pi agent session.
// Job: { system, prompt, schema?, tools?, workspace?, model?, skills?, thinking? }
//   schema    -> a submit_result tool with that JSON schema is added; the run ends when it is called
//   tools     -> pi built-in tools to enable (read, write, edit, bash, grep, find, ls); default none
//   workspace -> tools act inside workspaces/<workspace>/ under the project
//   skills    -> names under .agents/skills/ whose SKILL.md is inlined into the system prompt
// Reply: { result?, text, steps, model, ms }  or  { error }
// GET /health -> { ok, models }
import { mkdirSync, readFileSync } from "node:fs";
import { createServer } from "node:http";
import { join, resolve } from "node:path";
import { createAgentSession, createExtensionRuntime, ModelRuntime, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const HERE = import.meta.dirname;
const ROOT = resolve(HERE, "..");
const AGENT_DIR = join(HERE, "agent");
const PORT = Number(process.env.HARNESS_PORT || 8765);
const DEFAULT_MODEL = "dashscope/qwen3.8-max";

process.env.DASHSCOPE_API_KEY ||= readFileSync(join(ROOT, "apikey.txt"), "utf8").trim();
const runtime = await ModelRuntime.create({ authPath: join(AGENT_DIR, "auth.json"), modelsPath: join(AGENT_DIR, "models.json") });

function getModel(name) {
  const [provider, ...rest] = (name || DEFAULT_MODEL).split("/");
  const model = runtime.getModel(provider, rest.join("/"));
  if (!model) throw new Error(`Unknown model: ${name}`);
  return model;
}

function loadSkill(name) {
  if (!/^[\w-]+$/.test(name)) throw new Error(`Bad skill name: ${name}`);
  const text = readFileSync(join(ROOT, ".agents", "skills", name, "SKILL.md"), "utf8");
  return text.replace(/^---[\s\S]*?---\s*/, "").trim(); // Drop the frontmatter, keep the instructions
}

// A resource loader that discovers nothing: the job alone decides what the agent sees
function jobLoader(systemPrompt) {
  return {
    getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
    getSkills: () => ({ skills: [], diagnostics: [] }),
    getPrompts: () => ({ prompts: [], diagnostics: [] }),
    getThemes: () => ({ themes: [], diagnostics: [] }),
    getAgentsFiles: () => ({ agentsFiles: [] }),
    getSystemPrompt: () => systemPrompt,
    getSystemPromptSource: () => undefined,
    getAppendSystemPrompt: () => [],
    getAppendSystemPromptSources: () => [],
    extendResources: () => {},
    reload: async () => {},
  };
}

async function runJob(job) {
  const t0 = Date.now();
  const model = getModel(job.model);
  const cwd = job.workspace ? join(ROOT, "workspaces", job.workspace.replace(/[^\w-]/g, "_")) : HERE;
  mkdirSync(cwd, { recursive: true });

  let system = job.system || "";
  for (const s of job.skills || []) system += `\n\n# Skill: ${s}\n\n${loadSkill(s)}`;

  let result;
  const customTools = [];
  const tools = [...(job.tools || [])];
  if (job.schema) {
    system += "\n\nWhen you are done, call submit_result exactly once with the final result. Do not reply with plain text instead.";
    customTools.push({
      name: "submit_result",
      label: "Submit result",
      description: "Submit the final structured result. Call exactly once when done.",
      parameters: Type.Unsafe(job.schema),
      async execute(_id, params) {
        result = params;
        return { content: [{ type: "text", text: "Result received." }], details: undefined, terminate: true };
      },
    });
    tools.push("submit_result");
  }

  const { session } = await createAgentSession({
    cwd,
    agentDir: AGENT_DIR,
    model,
    thinkingLevel: job.thinking || "off",
    modelRuntime: runtime,
    resourceLoader: jobLoader(system),
    tools,
    customTools,
    sessionManager: SessionManager.inMemory(cwd),
    settingsManager: SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: true, maxRetries: 2 } }),
  });

  const steps = [];
  let error;
  session.subscribe((e) => {
    if (e.type === "tool_execution_start" && e.toolName !== "submit_result") steps.push({ tool: e.toolName, args: e.args });
    if (e.type === "message_end" && e.message.role === "assistant" && e.message.stopReason === "error") error = e.message.errorMessage;
  });
  try {
    await session.prompt(job.prompt);
    if (job.schema && result === undefined && !error) await session.prompt("Call submit_result now with the final result.");
  } finally {
    session.dispose();
  }
  if (error) throw new Error(error);
  if (job.schema && result === undefined) throw new Error("The model did not return a structured result");
  return { result, text: session.getLastAssistantText?.() || "", steps, model: `${model.provider}/${model.id}`, ms: Date.now() - t0 };
}

function send(res, status, body) {
  res.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

createServer(async (req, res) => {
  try {
    if (req.method === "GET" && req.url === "/health") {
      const models = (await runtime.getAvailable()).map((m) => `${m.provider}/${m.id}`);
      return send(res, 200, { ok: true, models, default: DEFAULT_MODEL });
    }
    if (req.method === "POST" && req.url === "/run") {
      let body = "";
      for await (const chunk of req) body += chunk;
      return send(res, 200, await runJob(JSON.parse(body)));
    }
    send(res, 404, { error: "not found" });
  } catch (e) {
    send(res, 500, { error: String(e?.message || e) });
  }
}).listen(PORT, "127.0.0.1", () => console.log(`pi harness listening on 127.0.0.1:${PORT}`));

"""FastAPI web UI: chat, live recall panel, and the time-travel console.

The right-hand panel shows, for every turn, which facts, episodes, and
tasks were recalled from CockroachDB and which node served the request.
The time-travel console rewinds the agent's beliefs to any timestamp
(AS OF SYSTEM TIME when possible) and diffs beliefs between two moments.
Clicking any agent reply shows its decision audit: which memory rows were
recalled into its context and which message originally taught each fact.

Endpoints:
  GET  /                     chat page (inline HTML, no external assets)
  POST /api/chat             {conversation_id?, message} -> reply + trace
  GET  /api/status           node + backend + memory counts
  GET  /api/beliefs?at=      belief state at a timestamp
  GET  /api/diff?start=&end= belief changes between two timestamps
  GET  /api/explain/{id}     decision audit for a reply episode
  GET  /healthz              liveness probe

Invoked by: ./run.sh web  (uvicorn web:app).
"""

import logging
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import config
import llm
import timetravel
from agent import Agent

config.setup_logging()
log = logging.getLogger("web")

app = FastAPI(title="Unforgettable")
_agent = None
# One turn at a time: the Database owns a single connection, and FastAPI
# runs sync endpoints on a thread pool, so concurrent turns must not
# interleave statements on it.
_turn_lock = threading.Lock()


_init_lock = threading.Lock()


def get_agent():
    global _agent
    if _agent is None:
        with _init_lock:
            if _agent is None:
                _agent = Agent()
    return _agent


_fallback_client = None


def llm_replies_today(database):
    """LLM-backed replies since UTC midnight (the daily spend meter)."""
    row = database.execute(
        "SELECT count(*) FROM episodes WHERE role = 'assistant'"
        " AND created_at >= date_trunc('day', now())"
        " AND meta->>'llm' != 'scripted'", fetch="one")
    return int(row[0])


def client_for_turn(agent):
    """The paid LLM client, or the scripted one past the daily ceiling.
    Returns (client, capped). The ceiling only applies when a paid
    backend is configured; 0 disables it."""
    global _fallback_client
    cap = config.DEMO_DAILY_LLM_REPLIES
    if cap <= 0 or config.LLM_BACKEND == "off":
        return agent.client, False
    if llm_replies_today(agent.db) < cap:
        return agent.client, False
    if _fallback_client is None:
        _fallback_client = llm.ScriptedClient()
    return _fallback_client, True


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


@app.post("/api/chat")
def chat(request: ChatRequest):
    agent = get_agent()
    with _turn_lock:
        client, capped = client_for_turn(agent)
        conversation_id = request.conversation_id
        if not conversation_id or not agent.store.conversation_exists(conversation_id):
            conversation_id = agent.new_conversation(title="web session")
        result = agent.turn(conversation_id, request.message, client=client)
    result["llm_mode"] = client.name
    if capped:
        result["note"] = ("daily LLM budget reached; replies come from the "
                          "deterministic scripted client until UTC midnight "
                          "- memory, time travel, and audit work identically")
    result["conversation_id"] = conversation_id
    try:
        node_id, _ = agent.db.node_info()
        result["node_id"] = node_id
    except Exception as error:  # status is best-effort, chat already worked
        log.warning("node_info failed: %s", error)
        result["node_id"] = None
    return JSONResponse(result)


@app.get("/api/status")
def status():
    return JSONResponse(get_agent().status())


@app.get("/api/beliefs")
def beliefs(at: str):
    try:
        return JSONResponse(timetravel.beliefs_at(get_agent().db, at))
    except ValueError:
        raise HTTPException(400, "invalid timestamp")


@app.get("/api/diff")
def diff(start: str, end: str):
    try:
        return JSONResponse(timetravel.belief_diff(get_agent().db, start, end))
    except ValueError:
        raise HTTPException(400, "invalid timestamp")


@app.get("/api/explain/{episode_id}")
def explain(episode_id: str):
    result = timetravel.explain_reply(get_agent().db, episode_id)
    if result is None:
        raise HTTPException(404, "no recall trace for that episode")
    return JSONResponse(result)


@app.get("/healthz")
def healthz():
    node_id, _ = get_agent().db.node_info()
    return {"ok": True, "node_id": node_id}


PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Unforgettable</title>
<style>
  body { margin: 0; font-family: system-ui, sans-serif; background: #10141a;
         color: #e6e8eb; display: flex; height: 100vh; }
  #chat { flex: 3; display: flex; flex-direction: column; padding: 16px; }
  #side { flex: 2; border-left: 1px solid #2a3038; padding: 16px;
          overflow-y: auto; font-size: 13px; }
  #log { flex: 1; overflow-y: auto; padding-right: 8px; }
  .msg { margin: 8px 0; padding: 10px 12px; border-radius: 8px;
         max-width: 80%; white-space: pre-wrap; }
  .msg .time { display: block; font-size: 10px; color: #9fb3c8;
               opacity: 0.8; margin-top: 6px; text-align: right; }
  .user { background: #2b6cb0; margin-left: auto; }
  .agent { background: #1f242c; cursor: pointer; }
  .agent:hover { outline: 1px solid #4a5361; }
  #bar { display: flex; gap: 8px; margin-top: 12px; }
  #input { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid
           #2a3038; background: #171c23; color: #e6e8eb; }
  button { padding: 8px 14px; border: 0; border-radius: 6px;
           background: #2b6cb0; color: white; cursor: pointer; }
  h1 { font-size: 18px; margin: 0 0 8px 0; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px;
       color: #8a93a0; margin: 18px 0 6px 0; }
  .item { padding: 6px 8px; background: #171c23; border-radius: 6px;
          margin: 4px 0; }
  .score, .note { color: #8a93a0; }
  .learned { border-left: 3px solid #48bb78; }
  .revised { border-left: 3px solid #ecc94b; }
  .retired { border-left: 3px solid #f56565; text-decoration: line-through; }
  #node { font-weight: bold; color: #63b3ed; }
  input[type=datetime-local] { background: #171c23; color: #e6e8eb;
    border: 1px solid #2a3038; border-radius: 6px; padding: 6px; }
  .row { display: flex; gap: 6px; margin: 4px 0; flex-wrap: wrap; }
</style>
</head>
<body>
<div id="chat">
  <h1>Unforgettable</h1>
  <div class="note">An agent whose entire memory is rows in CockroachDB.
  Rewind its beliefs to any moment, diff what changed, click a reply to see
  why it said that - and kill a database node without losing a thing.</div>
  <div id="log"></div>
  <div id="bar">
    <input id="input" placeholder="Say something the agent should remember..."
           autofocus>
    <button onclick="send()">Send</button>
  </div>
</div>
<div id="side">
  <h2>Cluster</h2>
  <div class="item">Serving node: <span id="node">?</span>
    <span id="totals" class="note"></span></div>
  <h2>Current beliefs</h2>
  <div id="beliefs" class="note">Nothing yet - teach the agent something.</div>
  <h2>Time travel</h2>
  <div class="row">
    <button onclick="preset(300)">Diff last 5 min</button>
    <button onclick="presetSincePageLoad()">Diff since page load</button>
  </div>
  <div class="row">
    <input type="datetime-local" id="t1" step="1">
    <button onclick="beliefsAt()">Beliefs at t1</button>
  </div>
  <div class="row">
    <input type="datetime-local" id="t2" step="1">
    <button onclick="diffBeliefs()">Diff t1 to t2</button>
  </div>
  <div class="note">t1 and t2 are two moments; the diff shows which beliefs
  changed between them. Message times are on each chat bubble, and each
  belief above shows when it started being true - use those to aim t1/t2.</div>
  <div class="note">Tip: click any agent reply to see the memories recalled
  for it.</div>
  <h2 id="paneltitle">Recalled this turn</h2>
  <div id="panel" class="note">Nothing yet.</div>
</div>
<script>
let conversationId = null;

function nowLocal(offsetSeconds) {
  const d = new Date(Date.now() + (offsetSeconds || 0) * 1000);
  d.setMilliseconds(0);
  const pad = n => String(n).padStart(2, "0");
  return d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" +
    pad(d.getDate()) + "T" + pad(d.getHours()) + ":" +
    pad(d.getMinutes()) + ":" + pad(d.getSeconds());
}
const pageLoadedAt = nowLocal(0);
document.getElementById("t1").value = nowLocal(-60);
document.getElementById("t2").value = nowLocal(0);

function clock() {
  const d = new Date();
  const pad = n => String(n).padStart(2, "0");
  return pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" +
    pad(d.getSeconds());
}

function preset(seconds) {
  document.getElementById("t1").value = nowLocal(-seconds);
  document.getElementById("t2").value = nowLocal(0);
  diffBeliefs();
}

function presetSincePageLoad() {
  document.getElementById("t1").value = pageLoadedAt;
  document.getElementById("t2").value = nowLocal(0);
  diffBeliefs();
}

function esc(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function add(role, text, episodeId) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.textContent = text;
  const stamp = document.createElement("span");
  stamp.className = "time";
  stamp.textContent = clock();
  div.appendChild(stamp);
  if (role === "agent" && episodeId) {
    div.title = "Click to see why the agent said this";
    div.onclick = () => explain(episodeId);
  }
  document.getElementById("log").appendChild(div);
  div.scrollIntoView();
}

async function refreshBeliefs() {
  // Read the belief state as of two seconds ago so the query stays on the
  // fast AS OF SYSTEM TIME path (which needs a timestamp in the past).
  const at = new Date(Date.now() - 2000).toISOString();
  try {
    const response = await fetch("/api/beliefs?at=" + encodeURIComponent(at));
    const data = await response.json();
    const parts = data.beliefs.map(b => {
      const since = new Date(b.valid_from);
      const pad = n => String(n).padStart(2, "0");
      const label = pad(since.getHours()) + ":" + pad(since.getMinutes()) +
        ":" + pad(since.getSeconds());
      return '<div class="item">' + esc(b.content) +
        ' <span class="score">since ' + label + '</span></div>';
    });
    document.getElementById("beliefs").innerHTML = parts.length ?
      parts.join("") :
      '<span class="note">Nothing yet - teach the agent something.</span>';
  } catch (err) { /* panel is best-effort */ }
}

function setPanel(title, html) {
  document.getElementById("paneltitle").textContent = title;
  document.getElementById("panel").innerHTML = html;
}

async function send() {
  const input = document.getElementById("input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  add("user", text);
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: text, conversation_id: conversationId}),
    });
    const data = await response.json();
    conversationId = data.conversation_id;
    add("agent", data.reply, data.reply_episode_id);
    document.getElementById("node").textContent =
      data.node_id === null ? "?" : "node " + data.node_id;
    renderRecalled(data.recalled);
    refreshTotals();
    setTimeout(refreshBeliefs, 2100);
  } catch (err) {
    add("agent", "[connection error: " + err + " - the client fails over " +
        "to a surviving node on the next message]");
  }
}

function renderRecalled(recalled) {
  const parts = [];
  for (const f of recalled.facts) {
    parts.push('<div class="item">' + esc(f.content) +
      ' <span class="score">(score ' + f.score.toFixed(2) + ')</span></div>');
  }
  for (const e of recalled.episodes) {
    parts.push('<div class="item">[past] ' + esc(e.content) + '</div>');
  }
  for (const t of recalled.tasks) {
    parts.push('<div class="item">[task] ' + esc(t.title) + '</div>');
  }
  setPanel("Recalled this turn", parts.length ? parts.join("") :
    '<span class="note">Nothing relevant.</span>');
}

async function beliefsAt() {
  const at = new Date(document.getElementById("t1").value).toISOString();
  const response = await fetch("/api/beliefs?at=" + encodeURIComponent(at));
  const data = await response.json();
  const parts = data.beliefs.map(b =>
    '<div class="item">' + esc(b.content) +
    ' <span class="score">(' + b.confidence.toFixed(2) + ', held since ' +
    new Date(b.valid_from).toLocaleTimeString() + ')</span></div>');
  setPanel("Beliefs at " + at + " via " + data.mechanism,
    parts.length ? parts.join("") :
    '<span class="note">The agent believed nothing at that moment.</span>');
}

async function diffBeliefs() {
  const start = new Date(document.getElementById("t1").value).toISOString();
  const end = new Date(document.getElementById("t2").value).toISOString();
  const response = await fetch("/api/diff?start=" +
    encodeURIComponent(start) + "&end=" + encodeURIComponent(end));
  const data = await response.json();
  const parts = [];
  for (const f of data.learned) {
    parts.push('<div class="item learned">learned: ' + esc(f.content) + '</div>');
  }
  for (const r of data.revised) {
    parts.push('<div class="item revised">' + r.change + ': "' +
      esc(r.before.content) + '" (' + r.before.confidence.toFixed(2) +
      ') -> "' + esc(r.after.content) + '" (' +
      r.after.confidence.toFixed(2) + ')</div>');
  }
  for (const f of data.retired) {
    parts.push('<div class="item retired">' + esc(f.content) + '</div>');
  }
  setPanel("Belief changes t1 to t2", parts.length ? parts.join("") :
    '<span class="note">No beliefs changed in that window.</span>');
}

async function explain(episodeId) {
  const response = await fetch("/api/explain/" + episodeId);
  if (!response.ok) {
    setPanel("Decision audit", '<span class="note">No trace found.</span>');
    return;
  }
  const data = await response.json();
  const parts = ['<div class="item">Asked: ' + esc(data.user_message) +
    '</div>'];
  for (const f of data.used_facts) {
    let source = "";
    if (f.taught_by) {
      source = '<br><span class="score">learned from: "' +
        esc(f.taught_by.content) + '"</span>';
    }
    parts.push('<div class="item">recalled: ' + esc(f.content) + source +
      '</div>');
  }
  for (const e of data.used_episodes) {
    parts.push('<div class="item">recalled past moment: ' + esc(e.content) +
      '</div>');
  }
  if (parts.length === 1) {
    parts.push('<div class="note">No long-term memory was recalled for ' +
      'this reply.</div>');
  }
  setPanel("What informed this reply", parts.join(""));
}

async function refreshTotals() {
  const response = await fetch("/api/status");
  const s = await response.json();
  if (s.node_id !== null && s.node_id !== undefined) {
    document.getElementById("node").textContent = "node " + s.node_id;
  }
  document.getElementById("totals").textContent =
    " - " + s.counts.episodes + " episodes, " + s.counts.facts +
    " beliefs (" + s.counts.fact_versions + " versions), " +
    s.counts.open_tasks + " tasks";
}

document.getElementById("input").addEventListener("keydown",
  e => { if (e.key === "Enter") send(); });
refreshTotals();
refreshBeliefs();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT)

"""Resilience showcase: kill a CockroachDB node mid-conversation, on purpose.

What it does, end to end, with zero external services:
  1. starts a local 3-node CockroachDB cluster (data/chaos/node1..3),
  2. runs a scripted conversation that stores facts and a task,
  3. SIGKILLs the node the agent is connected to,
  4. continues the conversation - the client fails over to a surviving
     node, recall still returns every memory, and new writes keep landing,
  5. prints a PASS/FAIL verdict and tears the cluster down.

Requires the `cockroach` binary on PATH (or --cockroach /path/to/binary).
Uses MEM_LLM=off so no keys are needed.

Invoked by: ./run.sh chaos  (or: python tools/chaos_demo.py [--keep]).
Inputs: none. Outputs: exit code 0 on PASS; cluster logs in data/chaos/.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
CHAOS_DIR = ROOT / "data" / "chaos"
REPLICATION_TIMEOUT = 300

BASE_SQL_PORT = 26261
BASE_HTTP_PORT = 8091
NODES = 3


def sql_url(port, database="unforgettable"):
    return ("postgresql://root@localhost:{0}/{1}?sslmode=disable"
            .format(port, database))


def start_node(cockroach, index, join_addrs):
    node_dir = CHAOS_DIR / "node{0}".format(index)
    node_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(node_dir / "stdout.log", "w")
    command = [
        cockroach, "start", "--insecure",
        "--store=path={0}".format(node_dir / "store"),
        "--listen-addr=localhost:{0}".format(BASE_SQL_PORT + index),
        "--http-addr=localhost:{0}".format(BASE_HTTP_PORT + index),
        "--join=" + ",".join(join_addrs),
    ]
    process = subprocess.Popen(command, stdout=log_file, stderr=log_file)
    return process


def wait_for_sql(cockroach, port, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        probe = subprocess.run(
            [cockroach, "sql", "--insecure", "--port", str(port),
             "-e", "SELECT 1"],
            capture_output=True)
        if probe.returncode == 0:
            return True
        time.sleep(1)
    return False


def wait_for_full_replication(cockroach, port, timeout=REPLICATION_TIMEOUT):
    """Block until every range has 3 replicas. Killing a node before
    up-replication finishes would lose quorum on the lagging ranges -
    that would be data loss theater, not a fair resilience test."""
    deadline = time.time() + timeout
    query = ("SELECT count(*) FROM crdb_internal.ranges_no_leases"
             " WHERE array_length(replicas, 1) < 3")
    while time.time() < deadline:
        probe = subprocess.run(
            [cockroach, "sql", "--insecure", "--port", str(port),
             "--format", "csv", "-e", query],
            capture_output=True, text=True)
        if probe.returncode == 0:
            lines = probe.stdout.strip().splitlines()
            remaining = int(lines[-1]) if lines else -1
            if remaining == 0:
                return True
            print("   waiting for up-replication "
                  "({0} ranges still below 3 replicas)".format(remaining))
        time.sleep(3)
    return False


def run_demo(keep_cluster, store_dir=None):
    global CHAOS_DIR
    if store_dir:
        CHAOS_DIR = Path(store_dir)
    cockroach = os.environ.get("COCKROACH") or shutil.which("cockroach")
    if not cockroach:
        print("ERROR: cockroach binary not found on PATH. Install it or set "
              "COCKROACH=/path/to/cockroach.")
        return 2

    if CHAOS_DIR.exists():
        shutil.rmtree(CHAOS_DIR)
    CHAOS_DIR.mkdir(parents=True)

    join_addrs = ["localhost:{0}".format(BASE_SQL_PORT + i)
                  for i in range(NODES)]
    print("== Starting a local {0}-node CockroachDB cluster".format(NODES))
    processes = [start_node(cockroach, i, join_addrs) for i in range(NODES)]

    try:
        time.sleep(2)
        subprocess.run(
            [cockroach, "init", "--insecure",
             "--host", "localhost:{0}".format(BASE_SQL_PORT)],
            capture_output=True)
        if not wait_for_sql(cockroach, BASE_SQL_PORT):
            print("ERROR: cluster did not become ready")
            return 2
        subprocess.run(
            [cockroach, "sql", "--insecure", "--port", str(BASE_SQL_PORT),
             "-e", "CREATE DATABASE IF NOT EXISTS unforgettable"],
            check=True, capture_output=True)
        print("   cluster ready; nodes on ports {0}".format(
            ", ".join(str(BASE_SQL_PORT + i) for i in range(NODES))))
        if not wait_for_full_replication(cockroach, BASE_SQL_PORT):
            print("ERROR: ranges did not reach 3 replicas within {0}s"
                  .format(REPLICATION_TIMEOUT))
            return 2
        print("   every range replicated 3 ways; safe to lose any one node")

        # Configure the app for all three nodes and no LLM keys, then import.
        os.environ["MEM_DB_URLS"] = ",".join(
            sql_url(BASE_SQL_PORT + i) for i in range(NODES))
        os.environ["MEM_LLM"] = "off"
        os.environ["MEM_EMBEDDINGS"] = "local"
        sys.path.insert(0, str(SRC))
        import config
        config.setup_logging()
        from agent import Agent

        agent = Agent()
        conversation_id = agent.new_conversation(title="chaos demo")
        node_before, _ = agent.db.node_info()
        print("== Agent connected; conversation served by node {0}"
              .format(node_before))

        script_before = [
            "Hi there. My name is Alice and I live in Singapore.",
            "I work at Acme Capital. My risk tolerance is conservative.",
            "Remind me to review the fund disclosure documents.",
        ]
        for line in script_before:
            result = agent.turn(conversation_id, line)
            print("you>   {0}".format(line))
            print("agent> {0}\n".format(result["reply"]))

        pre_kill_time = agent.db.execute("SELECT now()", fetch="one")[0]

        # Exact memory inventory before the kill: every fact version row
        # (id + subject + content), every task, and the episode count.
        # "0 memories lost" is asserted row-by-row, not by word-matching.
        def memory_inventory():
            fact_rows = agent.db.execute(
                "SELECT id, subject, content FROM facts", fetch="all") or []
            task_rows = agent.db.execute(
                "SELECT id, title FROM tasks", fetch="all") or []
            episode_count = agent.db.execute(
                "SELECT count(*) FROM episodes", fetch="one")[0]
            return ({(str(r[0]), r[1], r[2]) for r in fact_rows},
                    {(str(r[0]), r[1]) for r in task_rows},
                    episode_count)

        facts_before, tasks_before, episodes_before = memory_inventory()

        victim = processes[0]
        print("== CHAOS: sending SIGKILL to node 1 (pid {0}), the node the "
              "agent was using".format(victim.pid))
        victim.send_signal(signal.SIGKILL)
        victim.wait()
        time.sleep(2)

        checks = []
        script_after = [
            ("What is my name?", "alice"),
            ("What do you remember about me?", "conservative"),
            ("Also remind me to schedule the compliance review.",
             "compliance review"),
        ]
        for line, expected in script_after:
            result = agent.turn(conversation_id, line)
            print("you>   {0}".format(line))
            print("agent> {0}\n".format(result["reply"]))
            checks.append((line, expected,
                           expected in result["reply"].lower()))

        # Time travel across the failure: read the belief state from the
        # moment before the kill, served by a surviving node.
        import timetravel
        snapshot = timetravel.beliefs_at(agent.db, pre_kill_time)
        print("== Time travel: beliefs as of the moment before the kill "
              "({0} beliefs, via {1})".format(
                  len(snapshot["beliefs"]), snapshot["mechanism"]))
        checks.append(("time travel across node failure", "beliefs",
                       len(snapshot["beliefs"]) >= 3))

        # Row-exact survival: every pre-kill fact version and task must
        # still exist with identical id and content on the survivors
        # (rows are append-only, so even superseded versions must remain).
        facts_after, tasks_after, episodes_after = memory_inventory()
        checks.append(("every pre-kill fact row survived byte-identical",
                       "id+subject+content superset",
                       facts_before <= facts_after))
        checks.append(("every pre-kill task row survived byte-identical",
                       "id+title superset", tasks_before <= tasks_after))
        checks.append(("no episode rows lost", "count never shrinks",
                       episodes_after >= episodes_before))
        print("== Row-exact check: {0}/{1} pre-kill fact versions and "
              "{2}/{3} tasks present after failover".format(
                  len(facts_before & facts_after), len(facts_before),
                  len(tasks_before & tasks_after), len(tasks_before)))

        node_after, _ = agent.db.node_info()
        counts = agent.store.counts()
        print("== Failover: node {0} -> node {1}; memory now holds "
              "{2} episodes, {3} facts, {4} open tasks".format(
                  node_before, node_after, counts["episodes"],
                  counts["facts"], counts["open_tasks"]))

        passed = all(ok for _, _, ok in checks) and node_after != node_before
        for line, expected, ok in checks:
            print("   [{0}] '{1}' -> expected '{2}' in reply".format(
                "ok" if ok else "FAIL", line, expected))
        print("\nRESULT: {0} - a database node died mid-conversation and "
              "the agent {1} a single memory.".format(
                  "PASS" if passed else "FAIL",
                  "did not lose" if passed else "LOST"))
        return 0 if passed else 1

    finally:
        if keep_cluster:
            print("(--keep: leaving surviving nodes running; "
                  "stop them with: pkill -f 'cockroach start')")
        else:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            time.sleep(1)
            for process in processes:
                if process.poll() is None:
                    process.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true",
                        help="leave the surviving nodes running afterwards")
    parser.add_argument("--store-dir", default=None,
                        help="cluster data directory (default: data/chaos; "
                        "use fast local disk, e.g. /tmp, on WSL)")
    args = parser.parse_args()
    return run_demo(keep_cluster=args.keep, store_dir=args.store_dir)


if __name__ == "__main__":
    sys.exit(main())

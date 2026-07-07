"""Shared test setup: env, sys.path, and a real CockroachDB test database.

DB-backed tests run against an actual CockroachDB server because the memory
layer leans on CockroachDB semantics (VECTOR columns, JSONB, retry states)
that a fake would misrepresent. The server is found in this order:
  1. MEM_TEST_DSN env var (any reachable CockroachDB),
  2. the default local node on port 26257,
  3. a disposable single node started here, if `cockroach` is on PATH.
If none is available, DB-backed tests are skipped with a clear message.
Each test gets truncated tables inside a dedicated test database.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ["MEM_LLM"] = "off"
os.environ["MEM_EMBEDDINGS"] = "local"

FIXTURES = Path(__file__).parent / "fixtures"
_TEST_PORT = 26277
_started_process = None
_started_dir = None


def _try_connect(url):
    import psycopg
    try:
        conn = psycopg.connect(url, connect_timeout=2, autocommit=True)
        conn.close()
        return True
    except psycopg.Error:
        return False


def _start_disposable_node():
    global _started_process, _started_dir
    cockroach = shutil.which("cockroach")
    if not cockroach:
        return None
    _started_dir = tempfile.mkdtemp(prefix="unforgettable-test-crdb-")
    log_file = open(os.path.join(_started_dir, "stdout.log"), "w")
    _started_process = subprocess.Popen(
        [cockroach, "start-single-node", "--insecure",
         "--store=path={0}/store".format(_started_dir),
         "--listen-addr=localhost:{0}".format(_TEST_PORT),
         "--http-addr=localhost:{0}".format(_TEST_PORT + 1000)],
        stdout=log_file, stderr=log_file)
    url = "postgresql://root@localhost:{0}/defaultdb?sslmode=disable".format(
        _TEST_PORT)
    for _ in range(30):
        if _try_connect(url):
            return url
        time.sleep(1)
    return None


def _server_url():
    candidates = []
    if os.environ.get("MEM_TEST_DSN"):
        candidates.append(os.environ["MEM_TEST_DSN"])
    candidates.append("postgresql://root@localhost:26257/defaultdb?sslmode=disable")
    for url in candidates:
        if _try_connect(url):
            return url
    return _start_disposable_node()


@pytest.fixture(scope="session")
def test_db_url():
    server = _server_url()
    if server is None:
        pytest.skip("no CockroachDB reachable and no cockroach binary on "
                    "PATH; set MEM_TEST_DSN to run DB-backed tests")
    import psycopg
    db_name = "test_mem_{0}".format(uuid.uuid4().hex[:8])
    conn = psycopg.connect(server, autocommit=True)
    conn.execute("CREATE DATABASE {0}".format(db_name))
    conn.close()
    base, _, tail = server.rpartition("/")
    _, _, query = tail.partition("?")
    url = "{0}/{1}".format(base, db_name)
    if query:
        url = "{0}?{1}".format(url, query)
    yield url
    conn = psycopg.connect(server, autocommit=True)
    conn.execute("DROP DATABASE IF EXISTS {0} CASCADE".format(db_name))
    conn.close()


@pytest.fixture(scope="session")
def _session_database(test_db_url):
    from db import Database
    database = Database([test_db_url])
    database.ensure_schema()
    yield database
    database.close()


@pytest.fixture()
def database(_session_database):
    """Function-scoped handle with clean tables."""
    for table in ("episodes", "tasks", "facts", "conversations"):
        _session_database.execute("DELETE FROM {0}".format(table))
    return _session_database


def pytest_sessionfinish(session, exitstatus):
    if _started_process is not None:
        _started_process.terminate()
        try:
            _started_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _started_process.kill()
    if _started_dir is not None:
        shutil.rmtree(_started_dir, ignore_errors=True)

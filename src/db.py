"""CockroachDB access layer: multi-node failover, retries, schema bootstrap.

Owns the single connection to the cluster. Two resilience mechanisms:
  1. Client-side failover across the URLs in config.DB_URLS - when the
     connected node dies (the chaos demo kills one on purpose), the next
     statement reconnects to a surviving node and retries.
  2. Transaction retries on CockroachDB serialization errors (SQLSTATE
     40001), with exponential backoff, as recommended by Cockroach Labs.

Invoked by: memory_store.py, recall.py, consolidate.py, agent.py, tools/.
Inputs: SQL text + params. Outputs: rows as tuples.
"""

import logging
import random
import time

import psycopg

import config

log = logging.getLogger("db")

RETRYABLE_SQLSTATES = {"40001", "40003", "08003", "08006", "57P01"}


class Database:
    """One logical connection with failover across a list of node URLs."""

    def __init__(self, urls=None):
        self.urls = list(urls) if urls else list(config.DB_URLS)
        if not self.urls:
            raise ValueError("at least one database URL is required")
        self._conn = None
        self._url_index = 0

    # -- connection management ------------------------------------------

    def _connect(self):
        """Try each URL starting from the last good one; keep the first
        connection that answers. Raises the final error if all fail."""
        last_error = None
        for offset in range(len(self.urls)):
            index = (self._url_index + offset) % len(self.urls)
            url = self.urls[index]
            try:
                conn = psycopg.connect(
                    url,
                    autocommit=True,
                    connect_timeout=config.DB_CONNECT_TIMEOUT_SECONDS,
                )
                self._conn = conn
                self._url_index = index
                log.info("connected to node %d of %d", index + 1, len(self.urls))
                return
            except psycopg.OperationalError as error:
                last_error = error
                log.warning("node %d unreachable: %s", index + 1, error)
        raise last_error

    def _ensure_conn(self):
        if self._conn is None or self._conn.closed:
            self._connect()

    def close(self):
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    # -- statement execution --------------------------------------------

    def execute(self, sql, params=None, fetch="none"):
        """Run one autocommit statement with failover + 40001 retries.

        fetch: "none" | "one" | "all". Returns None, a tuple, or a list.
        """
        attempt = 0
        while True:
            try:
                self._ensure_conn()
                cur = self._conn.execute(sql, params)
                if fetch == "one":
                    return cur.fetchone()
                if fetch == "all":
                    return cur.fetchall()
                return None
            except psycopg.OperationalError as error:
                # Connection-level failure: node died or network dropped.
                attempt = self._handle_retry(attempt, error, reconnect=True)
            except psycopg.Error as error:
                state = getattr(error, "sqlstate", None)
                if state in RETRYABLE_SQLSTATES:
                    attempt = self._handle_retry(
                        attempt, error, reconnect=state != "40001")
                else:
                    raise

    def _handle_retry(self, attempt, error, reconnect):
        if attempt >= config.DB_MAX_RETRIES:
            raise error
        sleep = config.DB_RETRY_BASE_SLEEP * (2 ** attempt) + random.uniform(0, 0.1)
        log.warning("retry %d after error (%s); sleeping %.2fs",
                    attempt + 1, error, sleep)
        if reconnect:
            self.close()
            # Move past the dead node so failover tries the next URL first.
            self._url_index = (self._url_index + 1) % len(self.urls)
        time.sleep(sleep)
        return attempt + 1

    def transaction(self, statements):
        """Run several statements atomically, with the same failover and
        40001 retry behavior as execute(). statements is a list of
        (sql, params, fetch) tuples; returns one result per statement.
        Used where memory versioning must supersede and insert together.
        """
        attempt = 0
        while True:
            try:
                self._ensure_conn()
                results = []
                with self._conn.transaction():
                    for sql, params, fetch in statements:
                        cur = self._conn.execute(sql, params)
                        if fetch == "one":
                            results.append(cur.fetchone())
                        elif fetch == "all":
                            results.append(cur.fetchall())
                        else:
                            results.append(None)
                return results
            except psycopg.OperationalError as error:
                attempt = self._handle_retry(attempt, error, reconnect=True)
            except psycopg.Error as error:
                state = getattr(error, "sqlstate", None)
                if state in RETRYABLE_SQLSTATES:
                    attempt = self._handle_retry(
                        attempt, error, reconnect=state != "40001")
                else:
                    raise

    # -- schema -----------------------------------------------------------

    def ensure_schema(self):
        """Apply schema.sql idempotently and create vector indexes.

        Vector indexes are attempted separately: on self-hosted v25.2 they
        need a cluster setting; on some plans the setting is preset or not
        writable. If a vector index cannot be created, recall still works -
        similarity queries fall back to an exhaustive scan.
        """
        schema_sql = "\n".join(
            line for line in config.SCHEMA_PATH.read_text().splitlines()
            if not line.lstrip().startswith("--"))
        for statement in schema_sql.split(";"):
            statement = statement.strip()
            if statement:
                self.execute(statement)
        try:
            self.execute("SET CLUSTER SETTING feature.vector_index.enabled = true")
        except psycopg.Error as error:
            log.info("could not set vector index cluster setting: %s", error)
        for name, table in (("episodes_embedding_idx", "episodes"),
                            ("facts_embedding_idx", "facts")):
            try:
                self.execute(
                    "CREATE VECTOR INDEX IF NOT EXISTS {name} ON {table} (embedding)"
                    .format(name=name, table=table))
            except psycopg.Error as error:
                log.warning("vector index %s not created (%s); "
                            "similarity search will scan", name, error)

    # -- diagnostics -----------------------------------------------------

    def node_info(self):
        """Return (node_id, url_index) of the currently connected node."""
        row = self.execute("SELECT crdb_internal.node_id()", fetch="one")
        return row[0], self._url_index

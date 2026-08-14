# Container build for the Unforgettable agent (one stateless web process).
# Local demo: `docker compose up --build` runs this plus a CockroachDB node
# (see compose.yaml). AWS: the same image runs on ECS / App Runner / EC2 -
# set MEM_DB_URLS (and the Bedrock env vars from docs/DEPLOYMENT.md) and
# give the task role Bedrock invoke permissions.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tools/ tools/
COPY run.sh .

ENV PYTHONPATH=/app/src \
    MEM_WEB_HOST=0.0.0.0 \
    MEM_WEB_PORT=8400

EXPOSE 8400
CMD ["python", "src/web.py"]

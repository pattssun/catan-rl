"""Local run browser. Training runs independently of the HTTP service."""

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal

from catan_rl.telemetry import now


class Review(BaseModel):
    status: Literal["reviewed", "dismissed", "confirmed"]
    note: str = Field(min_length=1, max_length=2000)


class Store:
    def __init__(self, roots, database):
        self.roots = [Path(p).resolve() for p in roots]
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, run TEXT, episode TEXT, status TEXT, note TEXT, created_at TEXT)")

    def connect(self):
        conn = sqlite3.connect(self.database, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def directories(self):
        found = {}
        for i, root in enumerate(self.roots):
            if root.exists():
                for manifest in sorted(root.glob("*/manifest.json")):
                    if manifest.resolve().is_relative_to(root):
                        found[f"{i}-{manifest.parent.name}"] = manifest.parent
        return found

    def directory(self, run):
        path = self.directories().get(run)
        if path is None:
            raise HTTPException(404, "Run not found")
        return path

    def events(self, directory):
        path = directory / "events.jsonl"
        if not path.exists():
            return [], []
        if not path.resolve().is_relative_to(directory.resolve()):
            raise HTTPException(400, "Invalid event path")
        events, warnings = [], []
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("Event must be an object")
                events.append(event)
            except (ValueError, json.JSONDecodeError):
                warnings.append(f"Unreadable event on line {i + 1}; it was not counted.")
        return events, warnings

    def run(self, key):
        directory = self.directory(key)
        manifest = json.loads((directory / "manifest.json").read_text())
        events, warnings = self.events(directory)
        metrics = [e for e in events if e.get("kind") == "metrics"]
        episodes = [e for e in events if e.get("kind") == "episode"]
        with self.connect() as db:
            reviews = [dict(row) for row in db.execute(
                "SELECT * FROM reviews WHERE run = ? ORDER BY id", (key,))]
        latest = {r["episode"]: r for r in reviews}
        for episode in episodes:
            episode["review"] = latest.get(episode["id"])
        return {**manifest, "id": key, "metrics": metrics, "episodes": episodes,
                "warnings": warnings, "latest": metrics[-1] if metrics else {}}

    def episode(self, key, episode):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", episode):
            raise HTTPException(404, "Episode not found")
        directory = self.directory(key)
        path = directory / "episodes" / f"{episode}.json"
        if not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
            raise HTTPException(404, "Episode not found")
        result = json.loads(path.read_text())
        with self.connect() as db:
            result["reviews"] = [dict(r) for r in db.execute(
                "SELECT * FROM reviews WHERE run = ? AND episode = ? ORDER BY id",
                (key, episode))]
        return result

    def audit(self, key):
        directory = self.directory(key)
        path = directory / "audit.json"
        if not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
            raise HTTPException(404, "Reward audit not available")
        raw = path.read_bytes()
        manifest = json.loads((directory / "manifest.json").read_text())
        expected = manifest.get("audit_sha256")
        if expected and hashlib.sha256(raw).hexdigest() != expected:
            raise HTTPException(409, "Reward audit hash does not match its run manifest")
        try:
            report = json.loads(raw)
            if not isinstance(report, dict) or "gate" not in report:
                raise ValueError("Missing gate")
        except ValueError:
            raise HTTPException(409, "Reward audit is unreadable")
        return report


def create_app(roots=None, database="runs/reviews.sqlite3", dist="dashboard/dist"):
    store = Store(roots or ["runs", "examples/runs"], database)
    app = FastAPI(title="Catan RL Observatory")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.get("/api/health")
    def health():
        return {"status": "ok", "schema_version": 1}

    @app.get("/api/runs")
    def runs():
        results = []
        for key in store.directories():
            try:
                run = store.run(key)
                results.append({k: v for k, v in run.items() if k not in ("metrics", "episodes")}
                               | {"episode_count": len(run["episodes"]),
                                  "flag_count": sum(bool(e.get("flags")) for e in run["episodes"])})
            except (ValueError, OSError):
                results.append({"id": key, "title": key, "status": "unreadable",
                                "algorithm": "unknown", "latest": {}, "episode_count": 0,
                                "flag_count": 0, "config": {}, "provenance": {}, "source": "unknown"})
        return sorted(results, key=lambda r: r.get("created_at", ""), reverse=True)

    @app.get("/api/runs/{run}")
    def run_detail(run: str):
        return store.run(run)

    @app.get("/api/runs/{run}/episodes/{episode}")
    def episode_detail(run: str, episode: str):
        return store.episode(run, episode)

    @app.get("/api/runs/{run}/audit")
    def audit_detail(run: str):
        return store.audit(run)

    @app.post("/api/runs/{run}/episodes/{episode}/reviews")
    def review(run: str, episode: str, value: Review):
        store.episode(run, episode)
        with store.connect() as db:
            db.execute("INSERT INTO reviews (run, episode, status, note, created_at) VALUES (?, ?, ?, ?, ?)",
                       (run, episode, value.status, value.note, now()))
        return {"saved": True}

    if Path(dist).is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
    return app


def main():
    import uvicorn
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--runs", default="runs")
    p.add_argument("--examples", default="examples/runs")
    a = p.parse_args()
    uvicorn.run(create_app([a.runs, a.examples], Path(a.runs) / "reviews.sqlite3"),
                host=a.host, port=a.port)


if __name__ == "__main__":
    main()

from pathlib import Path


def pytest_sessionstart(session):
    (Path(__file__).resolve().parents[1] / ".tmp").mkdir(parents=True, exist_ok=True)

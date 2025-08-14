import json


class JSONLLogger:
    """Simple JSONL writer with line buffering."""
    def __init__(self, path: str):
        self._f = open(path, 'a', buffering=1)

    def log(self, obj: dict) -> None:
        self._f.write(json.dumps(obj) + "\n")

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass




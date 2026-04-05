from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import List, Dict, Any

@dataclass
class ChatTurn:
    role: str   # "user" or "assistant"
    content: str
    ts: float

class ChatHistory:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[ChatTurn]:
        if not self.path.exists():
            return []
        turns: List[ChatTurn] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                turns.append(ChatTurn(role=obj["role"], content=obj["content"], ts=obj["ts"]))
        return turns

    def append(self, role: str, content: str) -> None:
        rec = {"role": role, "content": content, "ts": time.time()}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

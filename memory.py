
\"\"\"Simple append-only JSONL memory store.\"\"\"
import json, os

class MemoryStore:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        if not os.path.exists(path):
            open(path, 'w').close()

    def add(self, record: dict):
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\\n')

    def tail(self, n=10):
        with open(self.path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().splitlines()
        items = [json.loads(l) for l in lines] if lines else []
        return items[-n:]

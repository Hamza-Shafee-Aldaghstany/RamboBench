import json
from typing import List


def read_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl_line(path: str, obj: dict):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")

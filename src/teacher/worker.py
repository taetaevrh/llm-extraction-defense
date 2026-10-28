"""Privileged model host. JSON text transport only; no logits/weights returned."""

import json
import sys
import yaml

from src.models import load_model, predict
from src.runtime import Phase, seed_everything, write_json


def main():
    config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
    resources = {}
    seed_everything(config["seed"])
    try:
        with Phase("teacher_load", resources):
            model, tokenizer = load_model(config["teacher"])
        print(json.dumps({"ready": True}), flush=True)
        with Phase("teacher_queries", resources):
            for line in sys.stdin:
                request = json.loads(line)
                try:
                    if set(request) != {"text"} or not isinstance(request["text"], str):
                        raise ValueError("Only a text request is permitted")
                    raw = predict(model, tokenizer, request["text"], config["decoding"])
                    print(json.dumps({"text": raw}), flush=True)
                except Exception as error:
                    print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)
                    raise
    finally:
        write_json(sys.argv[2], resources)


if __name__ == "__main__":
    main()

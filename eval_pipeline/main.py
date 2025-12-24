import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelLoader:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self._model = None
        self._tokenizer = None

    def get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_path"],
                trust_remote_code=self.config.get("trust_remote_code", True),
                local_files_only=self.config.get("local_files_only", True),
            )
        return self._tokenizer

    def get_model(self):
        if self._model is None:
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }

            torch_dtype = dtype_map.get(
                self.config.get("torch_dtype", "float16"),
                torch.float16,
            )

            device_map = self.config.get("device", "auto")

            self._model = AutoModelForCausalLM.from_pretrained(
                self.config["model_path"],
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=self.config.get("trust_remote_code", True),
                local_files_only=self.config.get("local_files_only", True),
            )

            self._model.eval()

        return self._model

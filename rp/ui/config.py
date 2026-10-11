from __future__ import annotations

import os
from dataclasses import dataclass

from .. import DEFAULT_DATA_DIR, DEFAULT_MODEL, DEFAULT_PORT


@dataclass
class UiConfig:
    host: str = "127.0.0.1"
    ui_port: int = 8847
    rp_host: str = "127.0.0.1"
    rp_port: int = 8627
    data_dir: str = DEFAULT_DATA_DIR
    model: str = DEFAULT_MODEL
    model_id: str = "qwen2.5-0.5b-rp"
    cradle_ui_url: str = "http://127.0.0.1:8765/"

    @property
    def rp_base_url(self) -> str:
        return f"http://{self.rp_host}:{self.rp_port}"

    @property
    def st_endpoint(self) -> str:
        return f"{self.rp_base_url}/v1"

    def narration_data_path(self, root: str) -> str:
        return os.path.join(root, "data", "rp-narration", "train.jsonl")


# RP server default 8619 is reserved for CLI; UI-managed server uses rp_port above.
DEFAULT_UI_PORT = 8847
DEFAULT_RP_PORT = 8627

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .. import DEFAULT_DATA_DIR, DEFAULT_MODEL


@dataclass
class UiConfig:
    host: str = '127.0.0.1'
    ui_port: int = 8847
    rp_host: str = '127.0.0.1'
    rp_port: int = 8627
    data_dir: str = DEFAULT_DATA_DIR
    model: str = DEFAULT_MODEL
    model_id: str = 'qwen2.5-0.5b-rp'
    cradle_ui_url: str = 'http://127.0.0.1:8765/'
    auth_token: str = ''

    def __post_init__(self):
        self.data_dir = str(Path(self.data_dir).expanduser().resolve())
        self.auth_token = self.auth_token or os.environ.get('RP_AUTH_TOKEN', '')
        if self.rp_host not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('The UI-managed model server must listen on loopback. Expose the authenticated UI through your reverse proxy.')
        if self.host not in ('127.0.0.1', 'localhost', '::1') and len(self.auth_token) < 24:
            raise ValueError('Set RP_AUTH_TOKEN to at least 24 random characters before exposing the UI.')

    @property
    def rp_base_url(self) -> str:
        host = f'[{self.rp_host}]' if ':' in self.rp_host else self.rp_host
        return f'http://{host}:{self.rp_port}'

    @property
    def st_endpoint(self) -> str:
        return f'{self.rp_base_url}/v1'

    def narration_data_path(self, root: str) -> str:
        return os.path.join(root, 'data', 'rp-narration', 'train.jsonl')


DEFAULT_UI_PORT = 8847
DEFAULT_RP_PORT = 8627

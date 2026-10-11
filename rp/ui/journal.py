from __future__ import annotations

import json
import os
import threading
import tempfile
from pathlib import Path


class JobJournal:
    def __init__(self, data_dir):
        self.root = Path(data_dir) / 'jobs'
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def save(self, record):
        identifier = record['id']
        if not identifier.isalnum():
            raise ValueError('Invalid job identifier.')
        with self.lock:
            path = self.root / f'{identifier}.json'
            descriptor, temp = tempfile.mkstemp(prefix='.job-',suffix='.json',dir=self.root)
            try:
                with os.fdopen(descriptor,'w',encoding='utf-8') as output:
                    json.dump(record,output,ensure_ascii=False,indent=2)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temp,path)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)

    def history(self):
        result = []
        with self.lock:
            for path in self.root.glob('*.json'):
                try:
                    result.append(json.loads(path.read_text(encoding='utf-8')))
                except (OSError, ValueError):
                    continue
        return sorted(result, key=lambda r: r.get('started_at') or 0, reverse=True)[:100]

    def get(self, identifier):
        if not isinstance(identifier,str) or not identifier.isalnum():
            raise ValueError('Invalid job identifier.')
        try:
            return json.loads((self.root / f'{identifier}.json').read_text(encoding='utf-8'))
        except (OSError,ValueError):
            return None

    def log_path(self, identifier):
        if not identifier.isalnum():
            raise ValueError('Invalid job identifier.')
        return self.root / f'{identifier}.log'

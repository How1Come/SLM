from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import psutil

from ..settings import LearningSettings
from ..store import Store
from .journal import JobJournal

ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ('queued', 'running', 'stopping')
PRESETS = {
    'quick': {'epochs': 1, 'max_minutes': 5, 'threads': 1},
    'balanced': {'epochs': 2, 'max_minutes': 15, 'threads': 2},
    'thorough': {'epochs': 4, 'max_minutes': 45, 'threads': 4},
}


def _env(cfg=None):
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    if cfg and cfg.auth_token:
        env['RP_AUTH_TOKEN'] = cfg.auth_token
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    return env


def request_headers(cfg):
    return {'Authorization': 'Bearer ' + cfg.auth_token} if cfg.auth_token else {}


def hardware(data_dir=ROOT):
    ram = psutil.virtual_memory()
    return {'cpu_percent': psutil.cpu_percent(), 'ram_total_gb': round(ram.total / 2**30, 2),
            'ram_available_gb': round(ram.available / 2**30, 2),
            'disk_free_gb': round(psutil.disk_usage(data_dir).free / 2**30, 2),
            'threads': os.cpu_count() or 1}


class RpServerProcess:
    def __init__(self, cfg):
        self.cfg = cfg
        self.proc = None
        self.lock = threading.RLock()
        self.started_at = None
        self.error = None
        self.log_path = Path(cfg.data_dir) / 'server.log'
        self.log_handle = None
        self.process_file = Path(cfg.data_dir) / 'server-process.json'

    def managed_process(self):
        if self.proc and self.proc.poll() is None:
            return self.proc
        try:
            record = json.loads(self.process_file.read_text(encoding='utf-8'))
            if record['model'] != self.cfg.model or record['endpoint'] != self.cfg.rp_base_url:
                return None
            process = psutil.Process(record['pid'])
            if process.is_running() and abs(process.create_time() - record['process_created']) < 1:
                return process
        except (OSError, ValueError, KeyError, TypeError, psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return None

    def status(self):
        with self.lock:
            process = self.managed_process()
            process_running = process is not None
            ready = False
            external = False
            detail = None
            try:
                response = httpx.get(self.cfg.rp_base_url + '/health', headers=request_headers(self.cfg), timeout=.7, trust_env=False)
                if response.is_success:
                    detail = response.json()
                    ready = detail.get('status') == 'ready' and detail.get('model') == self.cfg.model
                    external = not process_running and ready
            except (httpx.HTTPError, ValueError):
                pass
            if self.proc and self.proc.poll() is not None and self.proc.returncode and not self.error:
                self.error = 'The model server exited. Open the server log for details; check model access, dependencies and available RAM.'
            return {'running': process_running or external, 'ready': ready,
                    'starting': process_running and not ready, 'external': external, 'managed':process_running,
                    'pid': process.pid if process_running else None, 'error': self.error,
                    'endpoint': self.cfg.st_endpoint, 'model_id': self.cfg.model_id,
                    'base_url': self.cfg.rp_base_url, 'started_at': self.started_at,
                    'adapter': (detail or {}).get('adapter')}

    def start(self):
        with self.lock:
            existing = self.status()
            if existing['running']:
                return existing
            self.error = None
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_handle:
                self.log_handle.close()
            self.log_handle = self.log_path.open('a', encoding='utf-8')
            cmd = [sys.executable, '-m', 'rp.server', '--host', self.cfg.rp_host,
                   '--port', str(self.cfg.rp_port), '--data-dir', self.cfg.data_dir,
                   '--model', self.cfg.model, '--model-id', self.cfg.model_id, '--threads', '2']
            self.proc = subprocess.Popen(cmd, cwd=ROOT, env=_env(self.cfg), stdout=self.log_handle,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            self.started_at = time.time()
            try:
                record = {'pid':self.proc.pid,'process_created':psutil.Process(self.proc.pid).create_time(),
                          'model':self.cfg.model,'endpoint':self.cfg.rp_base_url}
                temporary = self.process_file.with_suffix('.tmp')
                temporary.write_text(json.dumps(record),encoding='utf-8')
                os.replace(temporary,self.process_file)
            except psutil.NoSuchProcess:
                self.error = 'The model server exited before it was ready. Check the server log.'
            return self.status()

    def stop(self):
        with self.lock:
            process = self.managed_process()
            if process is None:
                return False
            if isinstance(process,psutil.Process):
                process.terminate()
                try:
                    process.wait(timeout=15)
                except psutil.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                self.process_file.unlink(missing_ok=True)
                self.proc = None
                self.error = None
                return True
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
            if self.log_handle:
                self.log_handle.close()
            self.proc = None
            self.process_file.unlink(missing_ok=True)
            self.error = None
            return True


@dataclass
class JobSnapshot:
    id: str
    kind: str
    state: str
    progress: float
    log_tail: list[str]
    result: dict | None
    error: str | None
    started_at: float | None
    finished_at: float | None
    metrics: list[dict] = field(default_factory=list)
    checkpoint: str | None = None
    config: dict = field(default_factory=dict)


@dataclass
class _Job:
    id: str
    kind: str
    state: str = 'queued'
    progress: float = 0
    log: deque = field(default_factory=lambda: deque(maxlen=100))
    result: dict | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    metrics: list = field(default_factory=list)
    checkpoint: str | None = None
    config: dict = field(default_factory=dict)
    proc: object = None
    pid: int | None = None
    process_created: float | None = None

    def snapshot(self):
        return JobSnapshot(self.id, self.kind, self.state, self.progress, list(self.log),
                           self.result, self.error, self.started_at, self.finished_at,
                           list(self.metrics[-1000:]), self.checkpoint, self.config)


class JobManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.journal = JobJournal(cfg.data_dir)
        self._current = None
        self.listeners = set()
        self.reconcile()

    @staticmethod
    def process_alive(record):
        try:
            process = psutil.Process(record['pid'])
            return process.is_running() and abs(process.create_time() - record.get('process_created', -1)) < 1
        except (KeyError, TypeError, psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def reconcile(self):
        for record in self.journal.history():
            if record['state'] not in ACTIVE or self._current and record['id'] == self._current.id:
                continue
            if record.get('source') and self.process_alive(record):
                continue
            path = self.journal.log_path(record['id'])
            if path.is_file():
                with path.open(encoding='utf-8', errors='replace') as source:
                    source.seek(0,os.SEEK_END)
                    source.seek(max(0,source.tell() - 65536))
                    tail = source.read().splitlines()[-100:]
                record['log_tail'] = tail
                for line in tail:
                    try:
                        if line.startswith('METRIC '):
                            metric = json.loads(line[7:])
                            record['metrics'] = (record.get('metrics',[]) + [metric])[-1000:]
                            record['progress'] = metric.get('progress',record.get('progress',0))
                            record['checkpoint'] = metric.get('checkpoint') or record.get('checkpoint')
                        elif line.startswith('RESULT '):
                            record['result'] = json.loads(line[7:])
                            record['checkpoint'] = record['result'].get('checkpoint') or record.get('checkpoint')
                    except (ValueError, TypeError):
                        pass
            if not self.process_alive(record):
                result = record.get('result') or {}
                record['state'] = 'paused' if result.get('status') == 'paused' else 'done' if result.get('status') in ('ok','skipped') else 'interrupted'
                record['error'] = None if record['state'] != 'interrupted' else 'The trainer stopped unexpectedly. Complete checkpoints are retained; resume the latest saved state.'
                record['finished_at'] = time.time()
            self.journal.save(record)

    def current(self):
        with self.lock:
            if self._current and self._current.state in ACTIVE:
                return self._current.snapshot()
            self.reconcile()
            history = self.journal.history()
            active = [record for record in history if record['state'] in ACTIVE]
            history = active or history
            return JobSnapshot(**{k: v for k, v in history[0].items() if k in JobSnapshot.__dataclass_fields__}) if history else None

    def history(self):
        with self.lock:
            self.reconcile()
            return self.journal.history()

    def subscribe(self, cb):
        with self.lock:
            self.listeners.add(cb)

    def unsubscribe(self, cb):
        with self.lock:
            self.listeners.discard(cb)

    def emit(self, event):
        with self.lock:
            listeners = list(self.listeners)
        for callback in listeners:
            callback(event)

    def persist(self, job):
        self.journal.save({**asdict(job.snapshot()), 'pid': job.pid, 'process_created': job.process_created})

    def stop(self):
        with self.lock:
            job = self._current
            if not job or job.state not in ACTIVE:
                snapshot = self.current()
                if not snapshot or snapshot.state not in ACTIVE:
                    return False
                record = self.journal.get(snapshot.id)
                if not self.process_alive(record):
                    return False
                psutil.Process(record['pid']).send_signal(signal.SIGTERM)
                record['state'] = 'stopping'
                self.journal.save(record)
                return True
            job.state = 'stopping'
            if job.proc and job.proc.poll() is None:
                job.proc.send_signal(signal.SIGTERM)
            self.persist(job)
            return True

    def options(self, body=None, *, sleep=False):
        body = body or {}
        preset = body.get('preset', 'quick' if sleep else 'balanced')
        if preset not in PRESETS:
            raise ValueError('Choose quick, balanced or thorough training.')
        options = dict(PRESETS[preset])
        if sleep:
            settings = LearningSettings(self.cfg.data_dir).get()
            options['guard_max_drift'] = settings['guard_max_drift']
            if 'preset' not in body:
                options.update(epochs=settings['epochs'], max_minutes=settings['max_minutes'], threads=settings['threads'])
        for key, low, high in (('epochs', 1, 20), ('max_minutes', .1, 120), ('threads', 1, max(1, os.cpu_count() or 1))):
            if key in body:
                value = body[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                    raise ValueError(f'{key} must be between {low} and {high}.')
                if key != 'max_minutes' and not isinstance(value,int):
                    raise ValueError(f'{key} must be a whole number.')
                options[key] = float(value) if key == 'max_minutes' else int(value)
        if 'lr' in body:
            lr = float(body['lr'])
            if not 1e-6 <= lr <= .005:
                raise ValueError('Learning rate must be between 0.000001 and 0.005.')
            options['lr'] = lr
        return options

    def start_narration_train(self, body=None):
        options = self.options(body)
        corpus = (body or {}).get('data', 'seed')
        if corpus not in ('seed', 'scale'):
            raise ValueError('Choose seed or scale dataset.')
        path = ROOT / 'data' / ('rp-scale' if corpus == 'scale' else 'rp-narration') / 'train.jsonl'
        if not path.is_file():
            raise ValueError('That training dataset is unavailable. Use the bundled seed dataset or build the scaled corpus first.')
        options['data'] = str(path)
        return self._start('narration_train', options)

    def start_sleep(self, body=None):
        return self._start('sleep', self.options(body, sleep=True))

    def start_narration_eval(self):
        return self._start('narration_eval', {})

    def resume(self, job_id=None):
        record = self.journal.get(job_id) if job_id else next((r for r in self.history() if r.get('checkpoint')), None)
        if not record or not record.get('checkpoint'):
            raise ValueError('No complete checkpoint is available to resume.')
        if record['state'] not in ('paused', 'interrupted', 'error', 'stopped'):
            raise ValueError('Only a paused or interrupted training job can resume.')
        pid = record.get('pid')
        try:
            old = psutil.Process(pid) if pid else None
            if old and old.is_running() and abs(old.create_time() - record.get('process_created', -1)) < 1:
                raise ValueError('The earlier trainer is still running. Wait for its checkpoint and exit before resuming.')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        checkpoint = Path(record['checkpoint']).resolve()
        if not checkpoint.is_relative_to(Path(self.cfg.data_dir).resolve()) or not checkpoint.is_dir():
            raise ValueError('Checkpoint is missing or outside the project storage directory.')
        return self._start(record['kind'], {**record['config'], 'resume': str(checkpoint)})

    def _start(self, kind, config):
        with self.lock:
            if self._current and self._current.state in ACTIVE or any(record['state'] in ACTIVE for record in self.history()):
                raise RuntimeError('A training or evaluation job is already running.')
            disk = psutil.disk_usage(self.cfg.data_dir).free / 2**30
            if disk < 1:
                raise ValueError('Less than 1 GB of storage is available. Free space before training.')
            job = _Job(id=uuid.uuid4().hex[:16], kind=kind, config=config, started_at=time.time())
            self._current = job
            self.persist(job)
            threading.Thread(target=self._run, args=(job,), daemon=True).start()
            return job.snapshot()

    def command(self, job):
        module = {'sleep': 'rp.sleep', 'narration_train': 'rp.train_narration', 'narration_eval': 'rp.narration_eval'}[job.kind]
        command = [sys.executable, '-m', module, '--model', self.cfg.model]
        if job.kind == 'narration_eval':
            command += ['--versions', 'base', 'narration', '--data-dir', self.cfg.data_dir,
                        '--out', str(Path(self.cfg.data_dir) / 'evaluations' / job.id)]
        else:
            if job.kind == 'sleep':
                command += ['--data-dir', self.cfg.data_dir, '--server', self.cfg.rp_base_url]
            else:
                command += ['--out', str(Path(self.cfg.data_dir) / 'adapters' / 'narration'),
                            '--checkpoint-dir', str(Path(self.cfg.data_dir) / 'checkpoints' / job.id)]
            for key, value in job.config.items():
                if value is True:
                    command += ['--' + key.replace('_', '-')]
                elif value is not False and value is not None:
                    command += ['--' + key.replace('_', '-'), str(value)]
        return command

    def _run(self, job):
        code = -1
        try:
            with self.journal.log_path(job.id).open('w', encoding='utf-8') as output:
                with self.lock:
                    if job.state == 'stopping':
                        job.state = 'stopped'
                        return
                    job.proc = subprocess.Popen(self.command(job), cwd=ROOT, env=_env(self.cfg), stdout=output,
                                                stderr=subprocess.STDOUT, start_new_session=True)
                    job.pid = job.proc.pid
                    job.process_created = psutil.Process(job.pid).create_time()
                    job.state = 'running'
                    self.persist(job)
                with self.journal.log_path(job.id).open(encoding='utf-8') as reader:
                    while True:
                        line = reader.readline()
                        if line:
                            self._log(job, line.rstrip('\n'))
                        elif job.proc.poll() is not None:
                            break
                        else:
                            time.sleep(.1)
                    code = job.proc.wait()
            result = job.result or {}
            if result.get('status') == 'paused':
                job.state = 'paused'
            elif code != 0 or result.get('status') == 'failed' or job.kind != 'narration_eval' and result.get('status') not in ('ok','skipped'):
                job.state = 'error'
                job.error = result.get('reason') or 'Training stopped with an error. Expand the log for details; earlier complete checkpoints remain available.'
            elif job.state == 'stopping':
                job.state = 'paused' if job.checkpoint else 'stopped'
            else:
                job.state = 'done'
                job.progress = 1
                if job.kind == 'narration_eval':
                    md = Path(self.cfg.data_dir) / 'evaluations' / f'{job.id}.md'
                    if md.is_file():
                        job.result = {**result, 'eval_md': md.read_text(encoding='utf-8')}
                if job.kind == 'narration_train' and result.get('status') == 'ok':
                    try:
                        response = httpx.post(self.cfg.rp_base_url + '/rp/reload', headers=request_headers(self.cfg), timeout=120, trust_env=False)
                        job.result = {**result, 'reloaded': response.is_success}
                    except httpx.HTTPError:
                        pass
        except Exception as error:
            job.state = 'error'
            job.error = str(error) if isinstance(error, (ValueError, RuntimeError)) else 'The trainer could not start. Check dependencies, model access and available disk space.'
        finally:
            job.finished_at = time.time()
            self.persist(job)
            self.emit({'type': 'state', 'state': job.state, 'snapshot': asdict(job.snapshot())})

    def _log(self, job, line):
        with self.lock:
            job.log.append(line)
            if line.startswith('METRIC '):
                try:
                    metric = json.loads(line[7:])
                    job.metrics.append(metric)
                    job.progress = max(0, min(.99, float(metric.get('progress', job.progress))))
                    job.checkpoint = metric.get('checkpoint') or job.checkpoint
                except (ValueError, TypeError):
                    pass
            elif line.startswith('RESULT '):
                try:
                    job.result = json.loads(line[7:])
                    job.checkpoint = job.result.get('checkpoint') or job.checkpoint
                except ValueError:
                    pass
            else:
                match = re.search(r'epoch (\d+)/(\d+)', line)
                if match and not job.metrics:
                    job.progress = min(int(match[1]) / max(int(match[2]), 1), .99)
            self.persist(job)
        self.emit({'type': 'snapshot', 'data': asdict(job.snapshot())})


def collect_overview(cfg):
    from ..engine import list_adapters, narration_adapter_dir, read_current
    from ..sleep import build_examples
    store = Store(cfg.data_dir)
    stats = store.stats()
    examples = build_examples(store.active_signals(), 400)
    preferred = sum(item['chosen'] is not None for item in examples)
    current = read_current(cfg.data_dir,cfg.model)
    narration = narration_adapter_dir(cfg.data_dir)
    metadata = None
    if narration and (Path(narration) / 'meta.json').is_file():
        metadata = json.loads((Path(narration) / 'meta.json').read_text())
    train = Path(cfg.narration_data_path(str(ROOT)))
    count = 0
    if train.is_file():
        with train.open(encoding='utf-8') as source:
            count = sum(bool(line.strip()) for line in source)
    store.close()
    scale = ROOT / 'data' / 'rp-scale' / 'train.jsonl'
    return {'adapter': current, 'adapter_label': current, 'adapters': list_adapters(cfg.data_dir),
            'data_dir': cfg.data_dir, 'narration_train_examples': count, 'narration_meta': metadata,
            'narration_adapter_path': narration, 'store': stats, 'hardware': hardware(cfg.data_dir),
            'scale_available': scale.is_file(), 'settings': LearningSettings(cfg.data_dir).get(),
            'sleep': {'ready': preferred >= 1, 'preferred_replies': preferred,
                      'training_items': len(examples), 'message': f'{preferred} preferred replies available' if preferred else 'Chat, then keep a reply or add a correction to teach the model.'},
            'sillytavern': {'endpoint': cfg.st_endpoint, 'model_id': cfg.model_id},
            'cradle_ui_url': cfg.cradle_ui_url}

from __future__ import annotations

import hmac
import io
import json
import os
import queue
import re
import secrets
import zipfile
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from flask import Flask, Response, jsonify, render_template, request, send_file, session

from ..settings import LearningSettings
from ..store import Store
from .runner import JobManager, RpServerProcess, collect_overview, request_headers


def create_app(cfg, jobs=None, server=None):
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Strict'
    app.config['SESSION_COOKIE_SECURE'] = os.environ.get('RP_SECURE_COOKIE', '0') == '1'
    path = Path(cfg.data_dir)
    path.mkdir(parents=True, exist_ok=True)
    cookie_key = path / '.ui-secret'
    try:
        descriptor = os.open(cookie_key, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(descriptor, 'w') as output:
            output.write(secrets.token_hex(32))
    app.secret_key = cookie_key.read_text()
    jobs = jobs or JobManager(cfg)
    server = server or RpServerProcess(cfg)
    learning = LearningSettings(cfg.data_dir)
    app.extensions['jobs'] = jobs
    app.extensions['server'] = server

    @app.before_request
    def authorize():
        if not request.path.startswith(('/api/', '/v1/')):
            return None
        if request.method not in ('GET', 'HEAD'):
            origin = request.headers.get('Origin')
            if origin and urlsplit(origin).netloc != request.host:
                return jsonify(error='This request came from another site.'), 403
            if not request.is_json:
                return jsonify(error='Send a JSON request.'), 415
        if cfg.auth_token and request.path != '/api/auth/login' and (not session.get('authenticated') or request.path.startswith('/v1/')):
            authorization = request.headers.get('Authorization', '')
            if not hmac.compare_digest(authorization.encode('utf-8'),('Bearer '+cfg.auth_token).encode('utf-8')):
                return jsonify(error='Sign in with the server access token.'), 401
        return None

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)), 422

    @app.errorhandler(RuntimeError)
    def conflict(error):
        return jsonify(error=str(error)), 409

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error='The request is too large. Split the conversation or upload into smaller parts.'), 413

    def body():
        value = request.get_json(silent=True)
        if not isinstance(value, dict):
            raise ValueError('Use a JSON object.')
        return value

    def proxy(method, endpoint, payload=None, timeout=20):
        try:
            response = httpx.request(method, cfg.rp_base_url + endpoint, json=payload,
                                     headers=request_headers(cfg), timeout=timeout, trust_env=False)
            value = response.json()
        except httpx.TimeoutException:
            return jsonify(error='The model server is still working. CPU generation can take a while; check its status before retrying.'), 504
        except (httpx.HTTPError, ValueError):
            return jsonify(error='The model server is unavailable. Start it and wait for Ready before chatting.'), 503
        if not response.is_success:
            return jsonify(error=value.get('detail', 'The model server could not complete this request.')), response.status_code
        return jsonify(value)

    @app.get('/')
    def index():
        return render_template('index.html', cradle_ui_url=cfg.cradle_ui_url, ui_port=cfg.ui_port, rp_port=cfg.rp_port)

    @app.post('/api/auth/login')
    def login():
        token = body().get('token', '')
        if cfg.auth_token and (not isinstance(token, str) or not hmac.compare_digest(token.encode('utf-8'),cfg.auth_token.encode('utf-8'))):
            return jsonify(error='That access token was not accepted.'), 401
        session.clear()
        session['authenticated'] = True
        return jsonify(ok=True)

    @app.post('/api/auth/logout')
    def logout():
        session.clear()
        return jsonify(ok=True)

    @app.get('/api/overview')
    def overview():
        result = collect_overview(cfg)
        result['server'] = server.status()
        result['selected_adapter'] = result['adapter']
        if result['server']['ready'] and result['server'].get('adapter'):
            result['adapter'] = result['server']['adapter']
            result['adapter_label'] = result['adapter']
        snapshot = jobs.current()
        result['job'] = asdict(snapshot) if snapshot else None
        if result['server']['ready']:
            try:
                response = httpx.get(cfg.rp_base_url + '/rp/sleep', headers=request_headers(cfg), timeout=1, trust_env=False)
                if response.is_success:
                    result['auto_training'] = response.json()
            except httpx.HTTPError:
                pass
        return jsonify(result)

    @app.get('/api/settings')
    def settings():
        return jsonify(learning.get())

    @app.patch('/api/settings')
    def save_settings():
        return jsonify(learning.update(body()))

    @app.post('/api/narration/train')
    def train():
        return jsonify(asdict(jobs.start_narration_train(body())))

    @app.post('/api/narration/eval')
    def evaluate():
        return jsonify(asdict(jobs.start_narration_eval()))

    @app.post('/api/sleep')
    def sleep():
        if not collect_overview(cfg)['sleep']['ready']:
            return jsonify(error='Add a preferred reply or correction before training from chat.'), 409
        return jsonify(asdict(jobs.start_sleep(body())))

    @app.post('/api/job/stop')
    def stop_job():
        if jobs.stop():
            return jsonify(stopped=True)
        return proxy('POST', '/rp/sleep/stop', {})

    @app.post('/api/auto-training/stop')
    def stop_auto_training():
        return proxy('POST', '/rp/sleep/stop', {})

    @app.post('/api/job/resume')
    def resume():
        return jsonify(asdict(jobs.resume(body().get('job_id'))))

    @app.get('/api/job')
    def job_status():
        snapshot = jobs.current()
        return jsonify(asdict(snapshot) if snapshot else {'state': 'idle'})

    @app.get('/api/jobs')
    def history():
        return jsonify(jobs.history())

    @app.get('/api/jobs/<identifier>/logs')
    def logs(identifier):
        if not jobs.journal.get(identifier):
            return jsonify(error='Job not found.'), 404
        return send_file(jobs.journal.log_path(identifier), as_attachment=True, download_name=f'{identifier}.log')

    @app.post('/api/server/start')
    def start_server():
        return jsonify(server.start())

    @app.post('/api/server/stop')
    def stop_server():
        return jsonify(stopped=server.stop())

    @app.get('/api/server/logs')
    def server_log():
        if not server.log_path.is_file():
            return jsonify(error='No server log exists yet.'), 404
        return send_file(server.log_path, as_attachment=True, download_name='server.log')

    @app.post('/api/chat')
    def chat():
        value = body()
        value['stream'] = False
        value['model'] = cfg.model_id
        return proxy('POST', '/v1/chat/completions', value, timeout=600)

    @app.get('/v1/models')
    def model_list():
        return proxy('GET', '/v1/models')

    @app.post('/v1/chat/completions')
    def openai_chat():
        value = body()
        if not value.get('stream'):
            return proxy('POST', '/v1/chat/completions', value, timeout=600)
        client = httpx.Client(timeout=600, trust_env=False)
        outgoing = client.build_request('POST', cfg.rp_base_url + '/v1/chat/completions',
                                        json=value, headers=request_headers(cfg))
        try:
            upstream = client.send(outgoing, stream=True)
        except httpx.HTTPError:
            client.close()
            return jsonify(error='The model server is unavailable. Start it and wait for Ready.'), 503
        if not upstream.is_success:
            upstream.read()
            try:
                detail = upstream.json().get('detail', 'The request could not complete.')
            except ValueError:
                detail = 'The request could not complete.'
            upstream.close()
            client.close()
            return jsonify(error=detail), upstream.status_code

        def generate():
            try:
                for block in upstream.iter_bytes():
                    yield block
            finally:
                upstream.close()
                client.close()
        return Response(generate(), mimetype='text/event-stream', headers={'Cache-Control':'no-cache', 'X-Accel-Buffering':'no'})

    @app.post('/api/feedback')
    def feedback():
        value = body()
        if not value.get('gen_id'):
            raise ValueError('Choose the generated reply to give feedback on.')
        return proxy('POST', '/rp/feedback', value)

    @app.get('/api/signals')
    def signals():
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 25, type=int)
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError('Use a positive page and 1–100 rows per page.')
        with Store(cfg.data_dir) as store:
            return jsonify(items=store.recent_signals(page_size, offset=(page-1)*page_size), total=store.stats()['total_signals'])

    @app.patch('/api/signals/<int:identifier>')
    def review(identifier):
        active = body().get('active')
        if not isinstance(active, bool):
            raise ValueError('active must be true or false.')
        with Store(cfg.data_dir) as store:
            if not store.get_signal(identifier):
                return jsonify(error='Feedback not found.'), 404
            return jsonify(store.review_signal(identifier, active))

    @app.get('/api/signals/export')
    def export_signals():
        with Store(cfg.data_dir) as store:
            output = '\n'.join(json.dumps(item, ensure_ascii=False) for item in store.export_feedback(active_only=False))
        return Response(output + '\n', mimetype='application/x-ndjson', headers={'Content-Disposition': 'attachment; filename=roleplay-feedback.jsonl'})

    @app.post('/api/adapters/activate')
    def activate():
        version = body().get('version')
        if not isinstance(version, str) or not re.fullmatch(r'base|narration|v[0-9]+', version):
            raise ValueError('Choose base, narration or a saved chat version.')
        if not server.status()['ready']:
            from ..engine import adapter_path, narration_adapter_dir, narration_matches_model, write_current
            if version == 'narration':
                location = narration_adapter_dir(cfg.data_dir)
                if not location or not narration_matches_model(location,cfg.model):
                    raise ValueError('No compatible narration adapter exists for this base model.')
            elif version != 'base':
                try:
                    adapter_path(cfg.data_dir,version)
                except FileNotFoundError:
                    return jsonify(error='Adapter not found.'),404
            write_current(cfg.data_dir,version)
            return jsonify(ok=True,active=version)
        return proxy('POST', '/rp/activate', {'version': version}, timeout=180)

    @app.get('/api/adapters/<version>/export')
    def export_adapter(version):
        if not re.fullmatch(r'narration|v[0-9]+', version):
            raise ValueError('Only trained adapters can be exported.')
        from ..engine import narration_adapter_dir, adapter_path
        try:
            selected = narration_adapter_dir(cfg.data_dir) if version == 'narration' else adapter_path(cfg.data_dir,version)
        except FileNotFoundError:
            return jsonify(error='Adapter not found.'),404
        if not selected:
            return jsonify(error='Adapter is missing or incomplete.'), 404
        folder = Path(selected)
        if not folder.is_dir() or not (folder / 'adapter_config.json').is_file():
            return jsonify(error='Adapter is missing or incomplete.'), 404
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in folder.rglob('*'):
                if file.is_file() and not file.is_symlink():
                    archive.write(file, str(Path(version) / file.relative_to(folder)))
        buffer.seek(0)
        return send_file(buffer, mimetype='application/zip', as_attachment=True, download_name=f'{version}.zip')

    @app.get('/api/events')
    def events():
        updates = queue.Queue(maxsize=100)

        def listener(event):
            try:
                updates.put_nowait(event)
            except queue.Full:
                pass

        jobs.subscribe(listener)
        snapshot = jobs.current()
        if snapshot:
            updates.put({'type': 'snapshot', 'data': asdict(snapshot)})

        def stream():
            try:
                while True:
                    try:
                        event = updates.get(timeout=15)
                    except queue.Empty:
                        event = {'type': 'ping'}
                    yield 'data: ' + json.dumps(event, ensure_ascii=False) + '\n\n'
            finally:
                jobs.unsubscribe(listener)
        return Response(stream(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    return app

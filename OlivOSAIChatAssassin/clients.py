import json
import time
from urllib.parse import urljoin

import requests

RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
RESPONSE_FALLBACK_HTTP_STATUS = {400, 404, 405, 408, 409, 415, 425, 429, 500, 502, 503, 504}
MODEL_ALIAS_CANDIDATES = {
    'gpt-5-codex-mini': ['gpt-5.1-codex-mini'],
    'gpt-5-codex': ['gpt-5.1-codex']
}


class OpenAICompatError(RuntimeError):
    def __init__(self, message, code='', http_status=0):
        super().__init__(message)
        self.code = code
        self.http_status = int(http_status or 0)


class NapCatClient:
    def __init__(self, config, info_logger, warn_logger):
        self.info = info_logger
        self.warn = warn_logger
        self.session = requests.Session()
        self.stopped = False
        self.update_config(config)

    def update_config(self, config):
        self.config = {
            'base_url': str(config.get('base_url', 'http://127.0.0.1:3000')).rstrip('/') + '/',
            'event_base_url': str(config.get('event_base_url', config.get('base_url', 'http://127.0.0.1:3000'))).rstrip('/') + '/',
            'event_path': str(config.get('event_path', '/_events')),
            'headers': config.get('headers', {}) if isinstance(config.get('headers', {}), dict) else {},
            'request_timeout_ms': int(config.get('request_timeout_ms', 20000))
        }

    def stop(self):
        self.stopped = True
        self.session.close()

    def _join_url(self, base_url, path):
        return urljoin(base_url, str(path).lstrip('/'))

    def call(self, action, params=None):
        response = self.session.post(
            self._join_url(self.config['base_url'], action),
            headers={'Content-Type': 'application/json', **self.config['headers']},
            json=params or {},
            timeout=max(5, self.config['request_timeout_ms'] / 1000)
        )
        if response.status_code >= 400:
            raise RuntimeError(f'NapCat API {action} 返回 HTTP {response.status_code}: {response.text[:300]}')
        payload = response.json()
        if payload.get('status') not in (None, 'ok'):
            raise RuntimeError(f'NapCat API {action} 失败: {payload.get("message") or payload.get("wording") or payload.get("status")}')
        if payload.get('retcode') not in (None, 0):
            raise RuntimeError(f'NapCat API {action} retcode={payload.get("retcode")}')
        return payload.get('data', payload)

    def send_group_message(self, group_id, text, reply_to_message_id=None):
        message = []
        if reply_to_message_id:
            message.append({'type': 'reply', 'data': {'id': str(reply_to_message_id)}})
        message.append({'type': 'text', 'data': {'text': str(text)}})
        return self.call('send_group_msg', {'group_id': str(group_id), 'message': message})

    def start_event_loop(self, on_event):
        backoff_seconds = 2
        while not self.stopped:
            try:
                self._run_event_stream(on_event)
                backoff_seconds = 2
            except Exception as error:
                if self.stopped:
                    break
                self.warn(f'NapCat SSE 连接断开: {error}')
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 30)

    def _run_event_stream(self, on_event):
        response = self.session.get(
            self._join_url(self.config['event_base_url'], self.config['event_path']),
            headers={'Accept': 'text/event-stream', **self.config['headers']},
            stream=True,
            timeout=(10, None)
        )
        if response.status_code >= 400:
            raise RuntimeError(f'NapCat SSE 返回 HTTP {response.status_code}')
        self.info('NapCat SSE 已连接。')
        block = []
        for raw_line in response.iter_lines(decode_unicode=True):
            if self.stopped:
                break
            line = '' if raw_line is None else str(raw_line)
            if line == '':
                event = self._parse_sse_block(block)
                block = []
                if event is not None:
                    on_event(event)
                continue
            block.append(line)
        if not self.stopped:
            raise RuntimeError('NapCat SSE 连接已结束')

    def _parse_sse_block(self, lines):
        data_lines = []
        for line in lines:
            if line.startswith('data:'):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            return None
        try:
            return json.loads('\n'.join(data_lines))
        except Exception:
            self.warn('收到无法解析的 SSE 数据。')
            return None


class OpenAICompatClient:
    def __init__(self, config, info_logger, warn_logger):
        self.info = info_logger
        self.warn = warn_logger
        self.session = requests.Session()
        self.cooldown_until = 0
        self.cooldown_reason = ''
        self.retryable_failure_streak = 0
        self.transport_suppressed_until = {}
        self.update_config(config)

    def update_config(self, config):
        self.config = {
            'api_key': str(config.get('api_key', '')).strip(),
            'api_base': str(config.get('api_base', 'http://127.0.0.1:15721/v1')).rstrip('/') + '/',
            'model': str(config.get('model', '')).strip(),
            'temperature': float(config.get('temperature', 0.7)),
            'max_tokens': int(config.get('max_tokens', 512)),
            'retry_attempts': max(1, int(config.get('retry_attempts', 3))),
            'retry_delay_ms': max(200, int(config.get('retry_delay_ms', 1500))),
            'request_timeout_ms': max(5000, int(config.get('request_timeout_ms', 90000))),
            'failure_cooldown_ms': max(1000, int(config.get('failure_cooldown_ms', 60000))),
            'failure_cooldown_threshold': max(1, int(config.get('failure_cooldown_threshold', 2)))
        }

    def _join_url(self, path):
        return urljoin(self.config['api_base'], str(path).lstrip('/'))

    def _validate(self):
        if not self.config['api_base']:
            raise OpenAICompatError('chat.baseUrl 未配置', code='CHAT_BACKEND_INVALID_CONFIG')
        if not self.config['model']:
            raise OpenAICompatError('chat.model 未配置', code='CHAT_BACKEND_INVALID_CONFIG')

    def _is_cc_switch_proxy(self):
        base_url = self.config['api_base'].lower()
        return '127.0.0.1:15721/v1' in base_url or 'localhost:15721/v1' in base_url

    def _build_headers(self):
        headers = {'Content-Type': 'application/json'}
        if self.config['api_key']:
            headers['Authorization'] = f'Bearer {self.config["api_key"]}'
        if self._is_cc_switch_proxy():
            headers['Connection'] = 'close'
        return headers

    def _normalize_error_text(self, text):
        source = str(text or '').strip()
        if not source:
            return ''
        try:
            parsed = json.loads(source)
            message = parsed.get('error', {}).get('message') or parsed.get('message') or parsed.get('detail')
            if isinstance(message, str) and message.strip():
                return message.strip()[:400]
        except Exception:
            pass
        return source.replace('\r', ' ').replace('\n', ' ')[:240]

    def _is_retryable_error(self, error):
        if isinstance(error, OpenAICompatError) and error.http_status in RETRYABLE_HTTP_STATUS:
            return True
        message = str(error).lower()
        return any(keyword in message for keyword in ['timeout', 'timed out', 'network', 'socket', 'econnreset', 'enotfound', 'eai_again'])

    def _should_fallback_transport(self, error):
        if self._is_retryable_error(error):
            return True
        return isinstance(error, OpenAICompatError) and error.http_status in RESPONSE_FALLBACK_HTTP_STATUS

    def _build_model_candidates(self, model):
        normalized = str(model or '').strip()
        if not normalized:
            return []
        candidates = [normalized]
        for alias in MODEL_ALIAS_CANDIDATES.get(normalized, []):
            if alias not in candidates:
                candidates.append(alias)
        return candidates

    def _extract_chat_text(self, payload):
        content = payload.get('choices', [{}])[0].get('message', {}).get('content')
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get('type') == 'text':
                    parts.append(str(item.get('text', '')))
            return ''.join(parts).strip()
        return ''

    def _extract_responses_text(self, payload):
        direct = payload.get('output_text')
        if isinstance(direct, str):
            return direct.strip()
        if isinstance(direct, list):
            return ''.join(str(item.get('text') or item.get('output_text') or item.get('value') or '') for item in direct if isinstance(item, dict)).strip()
        output = payload.get('output')
        if isinstance(output, list):
            text = ''.join(
                ''.join(str(item.get('text') or item.get('output_text') or item.get('value') or '') for item in block.get('content', []) if isinstance(item, dict))
                for block in output if isinstance(block, dict)
            ).strip()
            if text:
                return text
        nested = payload.get('response')
        if isinstance(nested, dict) and nested is not payload:
            return self._extract_responses_text(nested)
        return ''

    def _build_responses_input(self, messages):
        result = []
        for message in messages:
            content = message.get('content')
            if isinstance(content, str) and content.strip():
                result.append({'role': message.get('role', 'user'), 'content': [{'type': 'input_text', 'text': content.strip()}]})
        return result

    def _build_flattened_input(self, messages):
        parts = []
        for message in messages:
            content = message.get('content')
            if isinstance(content, str) and content.strip():
                parts.append(f'{str(message.get("role", "user")).upper()}:\n{content.strip()}')
        return '\n\n'.join(parts).strip()

    def _request_json(self, path, payload, headers):
        response = self.session.post(
            self._join_url(path),
            headers=headers,
            json=payload,
            timeout=max(5, self.config['request_timeout_ms'] / 1000)
        )
        if response.status_code >= 400:
            raise OpenAICompatError(
                f'聊天接口返回 HTTP {response.status_code}：{self._normalize_error_text(response.text)}',
                code='CHAT_BACKEND_HTTP_ERROR',
                http_status=response.status_code
            )
        return response.json()

    def _run_retriable(self, func):
        last_error = None
        for attempt in range(1, self.config['retry_attempts'] + 1):
            try:
                return func()
            except Exception as error:
                last_error = error
                if attempt < self.config['retry_attempts'] and self._is_retryable_error(error):
                    self.warn(f'聊天接口请求异常，准备重试（{attempt}/{self.config["retry_attempts"]}）：{error}')
                    time.sleep((self.config['retry_delay_ms'] * attempt) / 1000)
                    continue
                raise
        raise last_error

    def _complete_via_chat(self, messages, model, temperature, max_tokens, headers):
        last_error = None
        for index, candidate in enumerate(self._build_model_candidates(model)):
            try:
                payload = self._run_retriable(lambda: self._request_json('chat/completions', {
                    'model': candidate,
                    'messages': messages,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'stream': False
                }, headers))
                text = self._extract_chat_text(payload)
                if not text:
                    raise OpenAICompatError('聊天接口未返回可用文本', code='CHAT_BACKEND_INVALID_RESPONSE')
                return text
            except Exception as error:
                last_error = error
                if index < len(self._build_model_candidates(model)) - 1 and self._should_fallback_transport(error):
                    self.warn(f'聊天接口 chat 当前模型 {candidate} 不稳定，切换到 {self._build_model_candidates(model)[index + 1]}：{error}')
                    continue
                raise
        raise last_error

    def _complete_via_responses(self, messages, model, temperature, max_tokens, headers):
        structured = self._build_responses_input(messages)
        variants = []
        if structured:
            variants.append(('structured-input_text', {'input': structured}))
        flattened = self._build_flattened_input(messages)
        if flattened and not self._is_cc_switch_proxy():
            variants.append(('flattened-string', {'input': flattened}))
        if not variants:
            raise OpenAICompatError('聊天接口未提供可发送内容', code='CHAT_BACKEND_INVALID_REQUEST')

        last_error = None
        for variant_name, variant_body in variants:
            for index, candidate in enumerate(self._build_model_candidates(model)):
                try:
                    payload = self._run_retriable(lambda: self._request_json('responses', {
                        'model': candidate,
                        'temperature': temperature,
                        'max_output_tokens': max_tokens,
                        **variant_body
                    }, headers))
                    text = self._extract_responses_text(payload)
                    if not text:
                        raise OpenAICompatError('聊天接口未返回可用文本', code='CHAT_BACKEND_INVALID_RESPONSE')
                    return text
                except Exception as error:
                    last_error = error
                    if index < len(self._build_model_candidates(model)) - 1 and self._should_fallback_transport(error):
                        self.warn(f'聊天接口 responses 当前载荷 {variant_name} 下模型 {candidate} 不稳定，切换模型：{error}')
                        continue
                    break
        raise last_error

    def complete(self, messages, model=None, temperature=None, max_tokens=None):
        self._validate()
        now = time.time()
        if now < self.cooldown_until:
            remaining = int(max(1, self.cooldown_until - now))
            raise OpenAICompatError(f'聊天接口暂时不可用，已进入 {remaining} 秒冷却：{self.cooldown_reason}', code='CHAT_BACKEND_COOLDOWN')

        model = str(model or self.config['model']).strip()
        temperature = self.config['temperature'] if temperature is None else temperature
        max_tokens = self.config['max_tokens'] if max_tokens is None else max_tokens
        headers = self._build_headers()
        transports = ['responses', 'chat'] if self._is_cc_switch_proxy() else ['chat', 'responses']
        last_error = None

        for index, transport in enumerate(transports):
            suppressed_until = self.transport_suppressed_until.get(transport, 0)
            if suppressed_until > now:
                continue
            try:
                if transport == 'responses':
                    text = self._complete_via_responses(messages, model, temperature, max_tokens, headers)
                else:
                    text = self._complete_via_chat(messages, model, temperature, max_tokens, headers)
                self.retryable_failure_streak = 0
                self.cooldown_until = 0
                self.cooldown_reason = ''
                return text
            except Exception as error:
                last_error = error
                if transport == 'chat' and self._is_cc_switch_proxy() and self._should_fallback_transport(error):
                    self.transport_suppressed_until['chat'] = time.time() + 600
                    self.warn(f'聊天接口 chat 暂时熔断 600 秒：{error}')
                if index < len(transports) - 1 and self._should_fallback_transport(error):
                    self.warn(f'聊天接口 {transport} 不稳定，切换到 {transports[index + 1]}：{error}')
                    continue
                break

        if self._is_retryable_error(last_error):
            self.retryable_failure_streak += 1
            if self.retryable_failure_streak >= self.config['failure_cooldown_threshold']:
                self.cooldown_until = time.time() + (self.config['failure_cooldown_ms'] / 1000)
                self.cooldown_reason = str(last_error)
                raise OpenAICompatError(f'聊天接口暂时不可用，已进入冷却：{last_error}', code='CHAT_BACKEND_COOLDOWN')
        else:
            self.retryable_failure_streak = 0

        if isinstance(last_error, OpenAICompatError):
            raise last_error
        raise OpenAICompatError(str(last_error), code='CHAT_BACKEND_UNKNOWN')

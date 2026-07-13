import json
import urllib.request


_model = 'gpt-3.5-turbo-0613'
_enable_thinking = None
_api_key = ''
_base_url = 'https://api.openai.com/v1'
_organization = ''


def configure(args):
    global _api_key, _base_url, _enable_thinking, _model, _organization
    _model = args.model
    _api_key = args.api_key
    _base_url = args.base_url.rstrip('/')
    _organization = args.openai_organization
    if args.enable_thinking is None:
        _enable_thinking = None
    else:
        _enable_thinking = args.enable_thinking == 'true'


def create_chat_completion(messages):
    request = {'model': _model, 'messages': messages, 'n': 1}
    if _enable_thinking is not None:
        request['enable_thinking'] = _enable_thinking
    headers = {
        'Authorization': f'Bearer {_api_key}',
        'Content-Type': 'application/json',
    }
    if _organization:
        headers['OpenAI-Organization'] = _organization
    http_request = urllib.request.Request(
        f'{_base_url}/chat/completions',
        data=json.dumps(request).encode('utf-8'),
        headers=headers,
        method='POST',
    )
    with urllib.request.urlopen(http_request, timeout=600) as response:
        return json.load(response)


def safe_args(args):
    values = vars(args).copy()
    if values.get('api_key'):
        values['api_key'] = '***'
    return values

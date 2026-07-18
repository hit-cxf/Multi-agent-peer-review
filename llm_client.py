import json
import urllib.request

from llm_adapter import LLMAdapter, default_model


_model = 'gpt-3.5-turbo-0613'
_enable_thinking = None
_api_key = ''
_base_url = 'https://api.openai.com/v1'
_organization = ''
_adapter = LLMAdapter()


def configure(args):
    global _adapter, _api_key, _base_url, _enable_thinking, _model, _organization
    _model = args.model
    _api_key = args.api_key
    _base_url = args.base_url.rstrip('/')
    _organization = args.openai_organization
    if args.enable_thinking is None:
        _enable_thinking = None
    else:
        _enable_thinking = args.enable_thinking == 'true'
    _adapter = LLMAdapter({
        'api_key': _api_key,
        'base_url': _base_url,
        'enable_thinking': args.enable_thinking,
        'organization': _organization,
    })


def create_chat_completion(messages, model=None, **overrides):
    """Call the endpoint selected by logical model name."""
    model_name = model or _model or default_model()
    settings = _adapter.resolve(model_name)
    request = {'model': settings.request_model, 'messages': messages, 'n': 1}
    if settings.enable_thinking is not None:
        request['enable_thinking'] = settings.enable_thinking
    request.update({key: value for key, value in overrides.items() if value is not None})
    headers = {
        'Authorization': f'Bearer {settings.api_key}',
        'Content-Type': 'application/json',
    }
    if settings.organization:
        headers['OpenAI-Organization'] = settings.organization
    http_request = urllib.request.Request(
        f'{settings.base_url}/chat/completions',
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


def safe_model_settings(model):
    settings = _adapter.resolve(model)
    return {
        'name': settings.name,
        'request_model': settings.request_model,
        'base_url': settings.base_url,
        'api_key': '***' if settings.api_key else '',
        'organization': settings.organization,
        'enable_thinking': settings.enable_thinking,
    }

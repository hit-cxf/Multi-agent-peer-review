# LLM endpoint configuration

The project loads `.env` automatically. Do not run `source .env`; experiment
processes resolve provider settings from the logical model name.

## Naming convention

A model name is converted to an uppercase environment prefix:

```text
qwen3-8b  -> LLM_QWEN3_8B_*
qwen3-14b -> LLM_QWEN3_14B_*
llama3-8b -> LLM_LLAMA3_8B_*
```

Supported fields are:

```text
BASE_URL
API_KEY
MODEL
ORGANIZATION
ENABLE_THINKING
```

`MODEL` is the actual model ID sent to the server. It can differ from the
logical routing name.

## Single-model scripts

Select a configured model with the existing argument:

```bash
python peer_review.py --task GSM8K --max_example_num 500 --model llama3-8b
```

When `--model` is omitted, `LLM_DEFAULT_MODEL` is used.

## Heterogeneous callers

One Python process can route different agents to different endpoints:

```python
llm_client.create_chat_completion(messages_a, model="qwen3-14b")
llm_client.create_chat_completion(messages_b, model="llama3-8b")
```

The local Llama URL `127.0.0.1:18080` is resolved on the machine running the
experiment. It therefore works after the project is copied to the inference
server, without exposing that endpoint to the local Mac.

See `.env.example` for a complete template. Legacy `OPENAI_*` variables remain
available as fallback settings for old launch commands.

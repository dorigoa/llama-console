from __future__ import annotations

from dataclasses import dataclass, field

from typing import TypedDict



class ModelConfig(TypedDict):
    path: str
    ctxsize: int



@dataclass
class Settings:
    UI_TITLE: str = "LLama Launcher"

    # NiceGUI bind parameters
    ui_host: str = "0.0.0.0"
    ui_port: int = 80
    ui_ssl_keyfile: str = "/Volumes/Home/dorigo_a/llama-server.key"
    ui_ssl_certfile: str = "/Volumes/Home/dorigo_a/llama-server.crt"

    # llama-server bind parameters
    llama_server_host: str = "0.0.0.0"
    llama_server_port: int = 8080#8443

    rpc_host: str = "192.168.20.2"
    rpc_host_ssh: str = "192.168.1.190"
    rpc_port: int = 50000
    openbrowser: bool = True

    rpc_server_path: str = "/usr/local/bin/rpc-server"
    llama_server_path: str = "/usr/local/bin/llama-server"

    llama_param: dict = field(default_factory=lambda: {
        "fit": "on",
        "threads": "6",
        "threadsbunch": "8",
        "parallel": "1",
        "ngl": "auto",
        "splitmodes": ["none", "layer", "row", "tensor"],
        "defaultsplitmode": "layer",
        "tensorsplit": "7,10",
        "ctxsize": "0",
        "sslkeyfile": "/Volumes/Home/dorigo_a/llama-server.key",
        "sslcertfile": "/Volumes/Home/dorigo_a/llama-server.crt",
    })

    AVAILABLE_MODELS: dict[str, ModelConfig]  = field(default_factory=lambda: {
        "Devstral-Small-2-24B-Instruct-2512-F16": {
            "path": "/Volumes/Home/gguf_models/Devstral-Small-2-24B-Instruct-2512-F16/Devstral-Small-2-24B-Instruct-2512-F16.gguf",
            "ctxsize": 0,
        },
        "Devstral-Small-2-24B-Instruct-2512-Q4_K_M": {
            "path": "/Volumes/Home/gguf_models/Devstral-Small-2-24B-Instruct-2512-Q4_K_M/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf",
            "ctxsize": 0,
        },
        "Devstral-Small-2-24B-Instruct-2512-Q6_K": {
            "path": "/Volumes/Home/gguf_models/Devstral-Small-2-24B-Instruct-2512-Q6_K/Devstral-Small-2-24B-Instruct-2512-Q6_K.gguf",
            "ctxsize": 0,
        },
        "Devstral-Small-2-24B-Instruct-2512-Q8_0": {
            "path": "/Volumes/Home/gguf_models/Devstral-Small-2-24B-Instruct-2512-Q8_0/Devstral-Small-2-24B-Instruct-2512-Q8_0.gguf",
            "ctxsize": 0,
        },
        "Gemma-3-27B-it-QAT-Q4_0": {
            "path": "/Volumes/Home/gguf_models/Gemma-3-27B-it-QAT-Q4_0/Gemma-3-27B-it-QAT-Q4_0.gguf",
            "ctxsize": 0,
        },
        "Gemma-4-26B-A4B-it-Q6_K": {
            "path": "/Volumes/Home/gguf_models/Gemma-4-26B-A4B-it-Q6_K/Gemma-4-26B-A4B-it-Q6_K.gguf",
            "ctxsize": 0,
        },
        "Gemma-4-26B-A4B-it-Q8_0": {
            "path": "/Volumes/Home/gguf_models/Gemma-4-26B-A4B-it-Q8_0/Gemma-4-26B-A4B-it-Q8_0.gguf",
            "ctxsize": 0,
        },
        "Gemma-4-31B-it-Q4_K_M": {
            "path": "/Volumes/Home/gguf_models/Gemma-4-31B-it-Q4_K_M/Gemma-4-31B-it-Q4_K_M.gguf",
            "ctxsize": 0,
        },
        "Gemma-4-31B-it-Q6_K": {
            "path": "/Volumes/Home/gguf_models/Gemma-4-31B-it-Q6_K/Gemma-4-31B-it-Q6_K.gguf",
            "ctxsize": 65546,
        },
        "Gemma-4-31B-it-Q8_0": {
            "path": "/Volumes/Home/gguf_models/Gemma-4-31B-it-Q8_0/Gemma-4-31B-it-Q8_0.gguf",
            "ctxsize": 32768,
        },
        "Gemma-4-E4B-it": {
            "path": "/Volumes/Home/gguf_models/Gemma-4-E4B-it/Gemma-4-E4B-it-Q8_0.gguf",
            "ctxsize": 262144,
        },
        "GPT-OSS-20B": {
            "path": "/Volumes/Home/gguf_models/GPT-OSS-20B/GPT-OSS-20B-MXFP4.gguf",
            "ctxsize": 0,
        },
        "Granite-3.2-8b-instruct": {
            "path": "/Volumes/Home/gguf_models/Granite-3.2-8b-instruct/Granite-3.2-8b-instruct-Q8_0.gguf",
            "ctxsize": 0,
        },
        "Meta-Llama-3.1-8B-Instruct": {
            "path": "/Volumes/Home/gguf_models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
            "ctxsize": 0,
        },
        "Ministral-3-14B-Reasoning-2512-Q4_K_M": {
            "path": "/Volumes/Home/gguf_models/Ministral-3-14B-Reasoning-2512-Q4_K_M/Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf",
            "ctxsize": 0,
        },
        "Ministral-3-14B-Reasoning-2512-Q6_K": {
            "path": "/Volumes/Home/gguf_models/Ministral-3-14B-Reasoning-2512-Q6_K/Ministral-3-14B-Reasoning-2512-Q6_K.gguf",
            "ctxsize": 0,
        },
        "Ministral-3-14B-Reasoning-2512-Q8_0": {
            "path": "/Volumes/Home/gguf_models/Ministral-3-14B-Reasoning-2512-Q8_0/Ministral-3-14B-Reasoning-2512-Q8_0.gguf",
            "ctxsize": 0,
        },
        "Mistral-Small-3.2-24B-Instruct-2506-F16": {
            "path": "/Volumes/Home/gguf_models/Mistral-Small-3.2-24B-Instruct-2506-F16/Mistral-Small-3.2-24B-Instruct-2506-F16.gguf",
            "ctxsize": 0,
        },
        "Mistral-Small-3.2-24B-Instruct-2506-Q8_0": {
            "path": "/Volumes/Home/gguf_models/Mistral-Small-3.2-24B-Instruct-2506-Q8_0/Mistral-Small-3.2-24B-Instruct-2506-Q8_0.gguf",
            "ctxsize": 0,
        },
        "Mistral-Small-3.2-24B-Instruct-2506": {
            "path": "/Volumes/Home/gguf_models/Mistral-Small-3.2-24B-Instruct-2506/Mistral-Small-3.2-24B-Instruct-2506-F16.gguf",
            "ctxsize": 0,
        },
        "NVIDIA-Nemotron-3-Nano-4B": {
            "path": "/Volumes/Home/gguf_models/NVIDIA-Nemotron-3-Nano-4B/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf",
            "ctxsize": 0,
        },
        "Phi-4-reasoning-plus-Q4_K_M": {
            "path": "/Volumes/Home/gguf_models/Phi-4-reasoning-plus-Q4_K_M/Phi-4-reasoning-plus-Q4_K_M.gguf",
            "ctxsize": 0,
        },
        "Phi-4-reasoning-plus-Q6_K": {
            "path": "/Volumes/Home/gguf_models/Phi-4-reasoning-plus-Q6_K/Phi-4-reasoning-plus-Q6_K.gguf",
            "ctxsize": 0,
        },
        "Phi-4-reasoning-plus-Q8_0": {
            "path": "/Volumes/Home/gguf_models/Phi-4-reasoning-plus-Q8_0/Phi-4-reasoning-plus-Q8_0.gguf",
            "ctxsize": 0,
        },
        "Qwen3-14B": {
            "path": "/Volumes/Home/gguf_models/Qwen3-14B/Qwen3-14B-Q8_0.gguf",
            "ctxsize": 0,
        },
        "Qwen3-30B-A3B-Q4_K_M": {
            "path": "/Volumes/Home/gguf_models/Qwen3-30B-A3B/Qwen3-30B-A3B-Q4_K_M.gguf",
            "ctxsize": 0,
        },
        "Qwen3-30B-A3B-Q5_0": {
            "path": "/Volumes/Home/gguf_models/Qwen3-30B-A3B/Qwen3-30B-A3B-Q5_0.gguf",
            "ctxsize": 131072,
        },
        "Qwen3-30B-A3B-Q5_K_M": {
            "path": "/Volumes/Home/gguf_models/Qwen3-30B-A3B/Qwen3-30B-A3B-Q5_K_M.gguf",
            "ctxsize": 65536,
        },
        "Qwen3-30B-A3B-Q6_K": {
            "path": "/Volumes/Home/gguf_models/Qwen3-30B-A3B/Qwen3-30B-A3B-Q6_K.gguf",
            "ctxsize": 0,
        },
        "Qwen3-30B-A3B-Q8_0": {
            "path": "/Volumes/Home/gguf_models/Qwen3-30B-A3B/Qwen3-30B-A3B-Q8_0.gguf",
            "ctxsize": 32768,
        },
        "Qwen3-coder-30B-F16": {
            "path": "/Volumes/Home/gguf_models/Qwen3-coder-30B/Qwen3-coder-30B-F16.gguf",
            "ctxsize": 32768,
        },
        "Qwen3-coder-30B-Q4_K_M": {
            "path": "/Volumes/Home/gguf_models/Qwen3-coder-30B/Qwen3-coder-30B-Q4_K_M.gguf",
            "ctxsize": 0,
        },
        "Qwen3-coder-30B-Q5_K_M": {
            "path": "/Volumes/Home/gguf_models/Qwen3-coder-30B/Qwen3-coder-30B-Q5_K_M.gguf",
            "ctxsize": 131072,
        },
        "Qwen3-coder-30B-Q6_K": {
            "path": "/Volumes/Home/gguf_models/Qwen3-coder-30B/Qwen3-coder-30B-Q6_K.gguf",
            "ctxsize": 131072,
        },
        "Qwen3-coder-30B-Q8_0": {
            "path": "/Volumes/Home/gguf_models/Qwen3-coder-30B/Qwen3-coder-30B-Q8_0.gguf",
            "ctxsize": 65536,
        },
        "Qwen3.6-27B-Q4_K_M": {
            "path": "/Volumes/Home/gguf_models/Qwen3.6-27B-Q4_K_M/Qwen3.6-27B-Q4_K_M.gguf",
            "ctxsize": 0,
        },
        "Qwen3.6-27B-Q6_K": {
            "path": "/Volumes/Home/gguf_models/Qwen3.6-27B-Q6_K/Qwen3.6-27B-Q6_K.gguf",
            "ctxsize": 0,
        },
        "Qwen3.6-27B-Q8_0": {
            "path": "/Volumes/Home/gguf_models/Qwen3.6-27B-Q8_0/Qwen3.6-27B-Q8_0.gguf",
            "ctxsize": 131072,
        },
        "Qwen3.6-35B-A3B-Q4_K_M": {
            "path": "/Volumes/Home/gguf_models/Qwen3.6-35B-A3B-Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf",
            "ctxsize": 0,
        },
        "Qwen3.6-35B-A3B-Q6_K": {
            "path": "/Volumes/Home/gguf_models/Qwen3.6-35B-A3B-Q6_K/Qwen3.6-35B-A3B-Q6_K.gguf",
            "ctxsize": 131072,
        },
        "Qwen3.6-35B-A3B-Q8_0": {
            "path": "/Volumes/Home/gguf_models/Qwen3.6-35B-A3B-Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf",
            "ctxsize": 65536,
        },
        "QwQ-32B-Q4_K_M": {
            "path": "/Volumes/Home/gguf_models/QwQ-32B-Q4_K_M/QwQ-32B-Q4_K_M.gguf",
            "ctxsize": 0,
        },
        "QwQ-32B-Q6_K": {
            "path": "/Volumes/Home/gguf_models/QwQ-32B-Q6_K/QwQ-32B-Q6_K.gguf",
            "ctxsize": 65536,
        },
        "QwQ-32B-Q8_0": {
            "path": "/Volumes/Home/gguf_models/QwQ-32B-Q8_0/QwQ-32B-Q8_0.gguf",
            "ctxsize": 32768,
        },
    })

settings = Settings()

import asyncio
from pathlib import Path
from nicegui import ui
from cli import get_llama_command

from config import Settings

settings = Settings()

# Format: { "Logic name": "Full/Path" }
# Generate the following list with the commands:
#   cd /Volumes/LLM/gguf_models/
#   find $PWD -iname "*.gguf"|awk -F"/" '{print "\""$(NF-1)"\": \""$0"\","}'

#------------------------------------------------------------------------------------------------
AVAILABLE_MODELS = {
    "Qwen3.6-27B-Q6_K": "/Volumes/Home/gguf_models/Qwen3.6-27B-Q6_K/Qwen3.6-27B-Q6_K.gguf",
"Devstral-Small-2-24B-Instruct-2512-Q8_0": "/Volumes/Home/gguf_models/Devstral-Small-2-24B-Instruct-2512-Q8_0/Devstral-Small-2-24B-Instruct-2512-Q8_0.gguf",
"Phi-4-reasoning-plus-Q6_K": "/Volumes/Home/gguf_models/Phi-4-reasoning-plus-Q6_K/Phi-4-reasoning-plus-Q6_K.gguf",
"NVIDIA-Nemotron-3-Nano-4B": "/Volumes/Home/gguf_models/NVIDIA-Nemotron-3-Nano-4B/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf",
"Granite-3.2-8b-instruct": "/Volumes/Home/gguf_models/Granite-3.2-8b-instruct/Granite-3.2-8b-instruct-Q8_0.gguf",
"Gemma-4-31B-it-Q4_K_M": "/Volumes/Home/gguf_models/Gemma-4-31B-it-Q4_K_M/Gemma-4-31B-it-Q4_K_M.gguf",
"Ministral-3-14B-Reasoning-2512-Q8_0": "/Volumes/Home/gguf_models/Ministral-3-14B-Reasoning-2512-Q8_0/Ministral-3-14B-Reasoning-2512-Q8_0.gguf",
"Devstral-Small-2-24B-Instruct-2512-Q4_K_M": "/Volumes/Home/gguf_models/Devstral-Small-2-24B-Instruct-2512-Q4_K_M/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf",
"Gemma-4-E4B-it": "/Volumes/Home/gguf_models/Gemma-4-E4B-it/Gemma-4-E4B-it-Q8_0.gguf",
"GPT-OSS-20B": "/Volumes/Home/gguf_models/GPT-OSS-20B/GPT-OSS-20B-MXFP4.gguf",
"Phi-4-reasoning-plus-Q4_K_M": "/Volumes/Home/gguf_models/Phi-4-reasoning-plus-Q4_K_M/Phi-4-reasoning-plus-Q4_K_M.gguf",
"Ministral-3-14B-Reasoning-2512-Q4_K_M": "/Volumes/Home/gguf_models/Ministral-3-14B-Reasoning-2512-Q4_K_M/Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf",
"QwQ-32B-Q8_0": "/Volumes/Home/gguf_models/QwQ-32B-Q8_0/QwQ-32B-Q8_0.gguf",
"Gemma-4-26B-A4B-it-Q8_0": "/Volumes/Home/gguf_models/Gemma-4-26B-A4B-it-Q8_0/Gemma-4-26B-A4B-it-Q8_0.gguf",
"Mistral-Small-3.2-24B-Instruct-2506-F16": "/Volumes/Home/gguf_models/Mistral-Small-3.2-24B-Instruct-2506-F16/Mistral-Small-3.2-24B-Instruct-2506-F16.gguf",
"QwQ-32B-Q4_K_M": "/Volumes/Home/gguf_models/QwQ-32B-Q4_K_M/QwQ-32B-Q4_K_M.gguf",
"Devstral-Small-2-24B-Instruct-2512-F16": "/Volumes/Home/gguf_models/Devstral-Small-2-24B-Instruct-2512-F16/Devstral-Small-2-24B-Instruct-2512-F16.gguf",
"Mistral-Small-3.2-24B-Instruct-2506-Q8_0": "/Volumes/Home/gguf_models/Mistral-Small-3.2-24B-Instruct-2506-Q8_0/Mistral-Small-3.2-24B-Instruct-2506-Q8_0.gguf",
"Qwen3-14B": "/Volumes/Home/gguf_models/Qwen3-14B/Qwen3-14B-Q8_0.gguf",
"Gemma-4-31B-it-Q6_K": "/Volumes/Home/gguf_models/Gemma-4-31B-it-Q6_K/Gemma-4-31B-it-Q6_K.gguf",
"Devstral-Small-2-24B-Instruct-2512-Q6_K": "/Volumes/Home/gguf_models/Devstral-Small-2-24B-Instruct-2512-Q6_K/Devstral-Small-2-24B-Instruct-2512-Q6_K.gguf",
"Phi-4-reasoning-plus-Q8_0": "/Volumes/Home/gguf_models/Phi-4-reasoning-plus-Q8_0/Phi-4-reasoning-plus-Q8_0.gguf",
"Qwen3-30B-A3B": "/Volumes/Home/gguf_models/Qwen3-30B-A3B/Qwen3-30B-A3B-Q5_0.gguf",
"Qwen3-30B-A3B": "/Volumes/Home/gguf_models/Qwen3-30B-A3B/Qwen3-30B-A3B-Q6_K.gguf",
"Qwen3-30B-A3B": "/Volumes/Home/gguf_models/Qwen3-30B-A3B/Qwen3-30B-A3B-Q4_K_M.gguf",
"Qwen3-30B-A3B": "/Volumes/Home/gguf_models/Qwen3-30B-A3B/Qwen3-30B-A3B-Q5_K_M.gguf",
"Qwen3-30B-A3B": "/Volumes/Home/gguf_models/Qwen3-30B-A3B/Qwen3-30B-A3B-Q8_0.gguf",
"Qwen3.6-27B-Q4_K_M": "/Volumes/Home/gguf_models/Qwen3.6-27B-Q4_K_M/Qwen3.6-27B-Q4_K_M.gguf",
"Ministral-3-14B-Reasoning-2512-Q6_K": "/Volumes/Home/gguf_models/Ministral-3-14B-Reasoning-2512-Q6_K/Ministral-3-14B-Reasoning-2512-Q6_K.gguf",
"Gemma-4-26B-A4B-it-Q6_K": "/Volumes/Home/gguf_models/Gemma-4-26B-A4B-it-Q6_K/Gemma-4-26B-A4B-it-Q6_K.gguf",
"QwQ-32B-Q6_K": "/Volumes/Home/gguf_models/QwQ-32B-Q6_K/QwQ-32B-Q6_K.gguf",
"Meta-Llama-3.1-8B-Instruct": "/Volumes/Home/gguf_models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
"Gemma-3-27B-it-QAT-Q4_0": "/Volumes/Home/gguf_models/Gemma-3-27B-it-QAT-Q4_0/Gemma-3-27B-it-QAT-Q4_0.gguf",
"Qwen3-coder-30B": "/Volumes/Home/gguf_models/Qwen3-coder-30B/Qwen3-coder-30B-Q6_K.gguf",
"Qwen3-coder-30B": "/Volumes/Home/gguf_models/Qwen3-coder-30B/Qwen3-coder-30B-Q8_0.gguf",
"Qwen3-coder-30B": "/Volumes/Home/gguf_models/Qwen3-coder-30B/Qwen3-coder-30B-Q5_K_M.gguf",
"Qwen3-coder-30B": "/Volumes/Home/gguf_models/Qwen3-coder-30B/Qwen3-coder-30B-Q4_K_M.gguf",
"Qwen3-coder-30B": "/Volumes/Home/gguf_models/Qwen3-coder-30B/Qwen3-coder-30B-F16.gguf",
"Mistral-Small-3.2-24B-Instruct-2506": "/Volumes/Home/gguf_models/Mistral-Small-3.2-24B-Instruct-2506/Mistral-Small-3.2-24B-Instruct-2506-F16.gguf",
"Gemma-4-31B-it-Q8_0": "/Volumes/Home/gguf_models/Gemma-4-31B-it-Q8_0/Gemma-4-31B-it-Q8_0.gguf",
}

#------------------------------------------------------------------------------------------------
class LlamaManager:
    def __init__(self):
        self.process = None

    async def start_server(self, model_path):
        if self.process:
            ui.notify("LLama server is already running!", type='warning')
            return

        try:
            cmd = get_llama_command(Path(model_path))
            
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            print(f"CMD={cmd}")

            ui.notify(f"Starting {model_path}...", type='info')
            
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break
                
                log_text = line.decode().strip()
                log_area.push(log_text)
                
        except Exception as e:
            ui.notify(f"Errore: {str(e)}", type='negative')
        finally:
            self.process = None

    async def stop_server(self):
        if self.process:
            self.process.terminate()
            ui.notify("Server terminated", type='info')
            log_area.push("--- Server Stopped ---")
            self.process = None
        else:
            ui.notify("No server active", type='warning')

manager = LlamaManager()

ui.colors(primary='#6e93d6')

with ui.header().classes('items-center justify-between'):
    ui.label('LLM Server Control Panel').classes('text-h6')
    ui.button('Stop Server', on_click=manager.stop_server, icon='stop', color='red')

with ui.column().classes('w-full max-w-4xl mx-auto p-4 gap-4'):
    with ui.card().classes('w-full p-4'):
        ui.label('Select a model').classes('text-subtitle1')
        
        model_select = ui.select(
            options=list(AVAILABLE_MODELS.keys()),
            value=None,
            label='Select a model from the list...'
        ).classes('w-full')
        
        ui.button('Start Server', 
                  on_click=lambda: manager.start_server(str(Path(AVAILABLE_MODELS[model_select.value]).parent)) if model_select.value else ui.notify("Select a model!"),
                  icon='play_arrow').classes('mt-4')

    ui.label('Server Logs').classes('text-subtitle2')
    log_area = ui.log().classes('w-full h-96 font-mono text-xs bg-black text-green-400')

#------------------------------------------------------------------------------------------------
ui.run(
    title=settings.UI_TITLE,
    host=settings.llama_server_host,
    port=settings.llama_server_port,
    ssl_certfile=settings.llama_param['sslcertfile'],
    ssl_keyfile=settings.llama_param['sslkeyfile'],
)

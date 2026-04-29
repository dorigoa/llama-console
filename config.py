from dataclasses import dataclass, field
import os


@dataclass
class Settings:
    UI_TITLE = "LLama Launcher"

    llama_server_host: str = '0.0.0.0'
    llama_server_port: int = 443
    rpc_host: str = "192.168.20.2"
    rpc_port: int = 50000
    openbrowser: bool = True
    rpc_server_path: str = "/usr/local/bin/rpc-server"
    llama_server_path: str = "/usr/local/bin/llama-server"
    llama_param = {}
    llama_param['fit'] = 'on'
    llama_param['threads'] = 6
    llama_param['threadsbunch'] = 8
    llama_param['parallel'] = 1
    llama_param['ngl'] = "auto"
    llama_param['splitmodes'] = ['none','layer','row','tensor']
    llama_param['defaultsplitmode'] = 'layer'
    llama_param['sslkeyfile'] = "/Volumes/Home/dorigo_a/llama-server.key"
    llama_param['sslcertfile'] = "/Volumes/Home/dorigo_a/llama-server.crt"

    #models_folder = "/Volumes/LLM/mlx_models"

settings = Settings()

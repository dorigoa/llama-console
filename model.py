from dataclasses import dataclass
from pathlib import Path
import json
import sys

#___________________________________________________________________________________
@dataclass
class rpc_address:
    IP: str
    PORT: int

#___________________________________________________________________________________
@dataclass
class Model:
    model_name: str
    model_path: Path
    mmproj_path: Path | None
    ctxsize: int
    temperature: float
    top_p: float
    top_k: int
    min_p: int
    reasoning: str
    last_started: int
    fitt: str
    gpus: str
    rpcservers: list[rpc_address]

#___________________________________________________________________________________
def load_models(config_path: str | Path) -> list[Model]:
    """Istanzia una lista di Model dalla sezione 'models' del config JSON.

    Solleva KeyError se mancano campi obbligatori (scelta voluta: meglio
    fallire esplicitamente che inventare default).
    """
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    base_dir = Path(config["MODEL_BASE_DIR"])
    models_section = config.get("models", {})

    models: list[Model] = []
    for name, spec in models_section.items():
        filename = name if name.endswith(".gguf") else f"{name}.gguf"

        rpcservers = [
            rpc_address(IP=ip, PORT=int(srv["port"]))
            for ip, srv in spec.get("RPC_SERVERS", {}).items()
        ]

        models.append(
            Model(
                model_name=name,
                model_path=base_dir / filename,
                mmproj_path=str(spec["MMPROJ"]),                       
                ctxsize=int(spec["ctx"]),
                temperature=float(spec["TEMP"]),
                top_p=float(spec["TOPP"]),
                top_k=int(spec["TOPK"]),
                min_p=int(spec["MINP"]),
                reasoning=str(spec["REAS"]),
                last_started=0,                         
                fitt=str(spec["FITT"]),
                gpus=str(spec["GPUS"]),
                rpcservers=rpcservers,
            )
        )
    return models

#___________________________________________________________________________________
if __name__ == "__main__":
    if len(sys.argv)<2:
        print(f"Usage: python model.py <filename>")
        sys.exit(1)
    if not sys.argv[1]:
        sys.exit(0)
    if not Path(sys.argv[1]).exists():
        print(f"Error: file {sys.argv[1]} not found")
        sys.exit(1)
    ms = load_models(sys.argv[1])
    print(f"{len(ms)} modelli caricati")
    for m in ms:
        rpc = ",".join(f"{s.IP}:{s.PORT}" for s in m.rpcservers) or ""
        print(f"  {m.model_name:50s} ctx={m.ctxsize:<7d} rpc=[{rpc}]")
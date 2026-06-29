from dataclasses import dataclass
from logzero import logger
from typing import Tuple
from pathlib import Path
import json
import shlex
import subprocess
import sys

#___________________________________________________________________________________
@dataclass
class rpc_server:
    IP: str
    #PUB_IP: str
    PORT: int
    cachepath: str
    bin: str
    remuser: str
    cachedisk: str | None = None

#___________________________________________________________________________________
@dataclass
class Model:
    alias: str
    model_name: str
    model_path: Path
    size_gib: float | None
    mmproj_path: Path | None
    ctxsize: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    reasoning: str
    last_started: int
    fitt: str
    rpcservers: list[rpc_server]
    #extras: list[str]
    ub: int
    b: int
    kvquant: str

#___________________________________________________________________________________
def _file_exists(path: Path, remote_host: str = "", remote_user: str = "") -> bool:
    if remote_host:
        dest = f"{remote_user}@{remote_host}" if remote_user else remote_host
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             dest, "test", "-f", str(path)],
            capture_output=True,
            text=True,
        )
        if result.returncode > 1:
            # returncode 0 = exists, 1 = not found (normal test -f); >1 = SSH error
            raise RuntimeError(
                f"SSH to {dest} failed (rc={result.returncode}): "
                f"{result.stderr.strip() or '(no stderr)'}"
            )
        return result.returncode == 0
    return path.exists()

#___________________________________________________________________________________
def _file_size_gib(path: Path, remote_host: str = "", remote_user: str = "") -> float | None:
    """Return the size of `path` in GiB, or None if it does not exist.

    Obtained from the remote host over SSH, analogously to _file_exists().
    Portable across Linux and macOS: tries GNU `stat -c %s`, then falls back to
    BSD `stat -f %z`. The local branch uses pathlib. Raises RuntimeError on SSH
    failure (rc > 1), as _file_exists does."""
    if remote_host:
        dest = f"{remote_user}@{remote_host}" if remote_user else remote_host
        q = shlex.quote(str(path))
        # GNU coreutils: `stat -c %s`; BSD/macOS: `stat -f %z`. Try GNU, fall
        # back to BSD. Return codes stay 0 = ok / 1 = not found / 255 = SSH error,
        # so the (rc > 1) SSH-failure test below still holds for both variants.
        remote_cmd = f"stat -c %s {q} 2>/dev/null || stat -f %z {q}"
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", dest, remote_cmd],
            capture_output=True,
            text=True,
        )
        if result.returncode > 1:
            # returncode 0 = ok, 1 = not found (both variants); >1 = SSH error
            raise RuntimeError(
                f"SSH to {dest} failed (rc={result.returncode}): "
                f"{result.stderr.strip() or '(no stderr)'}"
            )
        if result.returncode == 1:
            return None
        out = result.stdout.strip()
        if not out.isdigit():
            return None
        size_bytes = int(out)
    else:
        try:
            size_bytes = path.stat().st_size
        except OSError:
            return None
    return size_bytes / (1024 ** 3)

#___________________________________________________________________________________
def load_models(config_path: str | Path, remote_host: str = "", remote_user: str = "") -> list[Model]:

    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    base_dir = Path(config["MODEL_BASE_DIR"])
    models_section = config.get("models", {})

    models: list[Model] = []
    for name, spec in models_section.items():
        filename = name if name.endswith(".gguf") else f"{name}.gguf"
        model_path = base_dir / filename

        if not _file_exists(model_path, remote_host, remote_user):
            print(f"[SKIP] Model '{name}': file not found: {model_path}", file=sys.stderr)
            continue

        size_gib = _file_size_gib(model_path, remote_host, remote_user)

        rpcservers = [
            rpc_server(
                IP=ip,
                #PUB_IP=str(["public_ip"]),
                PORT=int(srv["port"]),
                cachepath=str(srv["cachepath"]),
                bin=str(srv["bin"]),
                remuser=str(srv["remuser"]),
                cachedisk=str(srv["cachedisk"]) if srv.get("cachedisk") is not None else None,
            )
            for ip, srv in spec.get("RPC_SERVERS", {}).items()
        ]

        models.append(
            Model(
                alias=str(spec["ALIAS"]),
                model_name=name,
                model_path=model_path,
                size_gib=size_gib,
                mmproj_path=Path(spec["MMPROJ"]) if spec["MMPROJ"] is not None else None,
                ctxsize=int(spec["ctx"]),
                temperature=float(spec["TEMP"]),
                top_p=float(spec["TOPP"]),
                top_k=int(spec["TOPK"]),
                min_p=float(spec["MINP"]),
                reasoning=str(spec["REAS"]),
                last_started=0,                         
                fitt=str(spec["FITT"]),
                rpcservers=rpcservers,
                #extras=spec["EXTRAS"],
                kvquant=spec["KVQUANT"],
                ub=spec["UB"],
                b=spec["B"]
            )
        )
    return models

#___________________________________________________________________________________
if __name__ == "__main__":
    import argparse
    from config_manager import get_settings

    _default_models = Path(__file__).parent / "models.json"

    parser = argparse.ArgumentParser(description="List models from a models config file")
    parser.add_argument(
        "config", nargs="?", default=str(_default_models),
        help=f"Path to models JSON config (default: {_default_models})",
    )
    parser.add_argument(
        "--remote-host", default=None,
        help="SSH host for file existence check (overrides config.json)",
    )
    parser.add_argument(
        "--remote-user", default=None,
        help="SSH user for file existence check (overrides config.json)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file '{config_path}' not found", file=sys.stderr)
        sys.exit(1)

    settings = get_settings()
    remote_host = args.remote_host if args.remote_host is not None else settings.LLAMA_SERVER_HOST
    remote_user = args.remote_user if args.remote_user is not None else settings.LLAMA_SERVER_USER

    ms = load_models(config_path, remote_host=remote_host, remote_user=remote_user)
    print(f"{len(ms)} models loaded (host: {remote_host or 'local'})")
    for m in ms:
        rpc = ",".join(f"{s.IP}:{s.PORT}" for s in m.rpcservers) or "-"
        size = f"{m.size_gib:.2f} GiB" if m.size_gib is not None else "n/a"
        print(f"  {m.model_name:50s} ctx={m.ctxsize:<7d} size={size:>11s} rpc=[{rpc}]")
        for s in m.rpcservers:
            disk = f"  disk={s.cachedisk}" if s.cachedisk else ""
            print(f"    {s.IP}:{s.PORT}  user={s.remuser}  bin={s.bin}  cache={s.cachepath}{disk}")

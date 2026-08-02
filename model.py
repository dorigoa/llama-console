from dataclasses import dataclass
from logzero import logger
from pathlib import Path
import json
import shlex
import sys

from remote_cmd_executor import remote_exec

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
    reasoning: str | None
    preserv_think: bool | None
    last_started: int
    rpcservers: list[str]
    ub: int
    b: int
    kvquant: str
    mtp: bool
    native_ctx: int
    rep_pen: float
    pres_pen: float | None

#___________________________________________________________________________________
def _file_exists(path: Path, remote_host: str = "", remote_user: str = "") -> bool:
    if remote_host:
        dest = f"{remote_user}@{remote_host}" if remote_user else remote_host
        args = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",  "-o", "StrictHostKeyChecking=no", 
                     dest, "test", "-f", str(path)]
        logger.debug(f"Executing command {args}")
        result = remote_exec( args )
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
        args = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", dest, remote_cmd]
        result = remote_exec( args )

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
def load_models(config_path: Path, 
                remote_host: str = "", 
                remote_user: str = "", 
                check_remote_file: bool = True, 
                check_model_name: str | None = None) -> list[Model]:

    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    

    base_dir = Path(config["MODEL_BASE_DIR"])
    models_section = config.get("models", {})
    

    models: list[Model] = []

    for name, spec in models_section.items():
        filename = name if name.endswith(".gguf") else f"{name}.gguf"
        model_path = base_dir / filename

        # When check_model_name is set, only check the remote file for that model;
        # all others skip the SSH round-trip.
        do_check = check_remote_file and (check_model_name is None or name == check_model_name)

        if do_check:
            if not _file_exists(model_path, remote_host, remote_user):
                logger.error(f"[SKIP] Model '{name}': file not found: {model_path}")
                continue

            size_gib = _file_size_gib(model_path, remote_host, remote_user)
        else:
            size_gib = None

        mtp = False
        if 'MTP' in spec and spec['MTP'] == True:
            mtp = True
        pp = None
        if spec.get("PP") and spec.get("PP") > -1:
            pp = float( spec.get("pres_pen") )

        rp = None
        if spec.get("RP"):
            rp = float( spec.get("RP") )

        reas_eff = None
        if spec.get("REAS"):
            reas_eff = str(spec["REAS"])

        preserv_think = None
        if spec.get("PRES_THK"):
            preserv_think = spec["PRES_THK"]

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
                reasoning=reas_eff,
                preserv_think=preserv_think,
                last_started=0,
                rpcservers=spec['RPC_SERVERS'],
                kvquant=spec["KVQUANT"],
                ub=spec["UB"],
                b=spec["B"],
                mtp=mtp,
                native_ctx=int(spec.get("native_ctx", spec["ctx"])),
                rep_pen=rp,
                pres_pen=pp
            
            )
        )
    return models

#___________________________________________________________________________________
if __name__ == "__main__":
    import argparse
    from config_manager import get_settings

    parser = argparse.ArgumentParser(description="List models from models config file")

    parser.add_argument(
        "--master-host", default=None,
        help="SSH host for file existence check (overrides config.json)",
    )
    parser.add_argument(
        "--master-user", default=None,
        help="SSH user for file existence check (overrides config.json)",
    )
    parser.add_argument(
        "--models-config", default=Path(__file__).parent / "models.json",
        help="path of models json description file"
    )

    args = parser.parse_args()

    config_path = Path(args.models_config)

    if not config_path.exists():
        logger.error(f"Error: config file '{config_path}' not found")
        sys.exit(1)
    
    settings = get_settings()
    master_host = args.master_host if args.master_host is not None else settings.LLAMA_SERVER_HOST
    master_user = args.master_user if args.master_user is not None else settings.LLAMA_SERVER_USER

    ms = load_models(config_path=config_path, remote_host=master_host, remote_user=master_user)
    logger.debug(f"{len(ms)} models loaded (host: {master_host or 'local'})")
    for m in ms:
        size = f"{m.size_gib:.2f} GiB" if m.size_gib is not None else "n/a"
        logger.debug(f"  {m.model_name:50s} ctx={m.ctxsize:<7d} size={size:>11s} rpc=[{m.rpcservers}]")
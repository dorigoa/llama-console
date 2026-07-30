import subprocess
from logzero import logger

def remote_exec( args, timeout: int | None = 120, capture_output: bool = True ):

    logger.debug(f"Executing command {args}")

    try:
        return subprocess.run(
                    args,
                    capture_output=capture_output,
                    text=True,
                    timeout=timeout
                )
    except subprocess.TimeoutExpired as e:
        logger.error(f"timeout after {timeout}s executing command")
        return

    

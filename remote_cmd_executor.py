import subprocess
from logzero import logger

def remote_exec( args, timeout = 10 ):

    logger.debug(f"Executing command {args}")

    try:
        return subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
    except subprocess.TimeoutExpired as e:
        logger.error(f"timeout after {timeout}s executing command")
        return

    
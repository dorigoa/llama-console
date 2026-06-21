#!/usr/bin/env python3
"""
kv_per_token.py  (v2 — SWA-aware)
Stima la dimensione della KV-cache (per token e a contesto N) di un modello GGUF,
leggendo i soli metadati col parser ufficiale `gguf` (nessun caricamento dei pesi).

Installazione:  pip install gguf
Uso:
    python kv_per_token.py modello.gguf
    python kv_per_token.py modello.gguf --ctx 131072
    python kv_per_token.py modello.gguf --ctx 32768 --cache-type-k q8_0 --cache-type-v q8_0

Formula per layer:  byte/token = n_kv * (d_k*b_k + d_v*b_v)
Totale a contesto N: i layer global scalano con N; i layer sliding-window
si fermano a min(N, finestra).  -> total = grow*N + fixed*min(N, window)

Esatto per i modelli dense e (per i conteggi per-layer) per Gemma 4.
La distinzione global/sliding usa un'euristica: verifica sempre con la riga
"KV self size" di llama-server, che e' la verita' a runtime.
"""

import argparse
import sys

try:
    from gguf import GGUFReader, GGUFValueType
except ImportError:
    sys.exit("Manca il pacchetto 'gguf'. Installa con:  pip install gguf")

# byte per elemento per i tipi di cache K/V di llama.cpp
# (i quantizzati: byte_per_blocco_da_32 / 32)
CACHE_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
    "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32,
}


def _norm(v):
    import numpy as np
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, str):
        return v
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def field_value(reader, key):
    """Valore di un campo di metadati (scalare/array/stringa) o None."""
    field = reader.fields.get(key)
    if field is None:
        return None
    if hasattr(field, "contents"):
        try:
            return _norm(field.contents())
        except Exception:
            pass
    t = field.types[0]
    if t == GGUFValueType.ARRAY:
        et = field.types[1]
        return [bytes(field.parts[i]).decode("utf-8", "replace") if et == GGUFValueType.STRING
                else int(field.parts[i][0]) for i in field.data]
    if t == GGUFValueType.STRING:
        return bytes(field.parts[field.data[-1]]).decode("utf-8", "replace")
    return int(field.parts[field.data[-1]][0])


def human(n):
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or u == "TiB":
            return f"{n:.2f} {u}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser(description="Dimensione KV-cache di un modello GGUF.")
    ap.add_argument("gguf")
    ap.add_argument("--ctx", type=int, default=None,
                    help="contesto N per il totale (default: context_length del modello)")
    ap.add_argument("--cache-type-k", default="f16", choices=sorted(CACHE_BYTES))
    ap.add_argument("--cache-type-v", default="f16", choices=sorted(CACHE_BYTES))
    ap.add_argument("--cache-type", default=None, choices=sorted(CACHE_BYTES),
                    help="imposta K e V insieme")
    args = ap.parse_args()
    if args.cache_type:
        args.cache_type_k = args.cache_type_v = args.cache_type

    reader = GGUFReader(args.gguf)
    arch = field_value(reader, "general.architecture")
    if not arch:
        sys.exit("Impossibile leggere general.architecture.")

    g = lambda suf: field_value(reader, f"{arch}.{suf}")
    L           = g("block_count")
    n_head      = g("attention.head_count")
    n_head_kv   = g("attention.head_count_kv")
    d_k         = g("attention.key_length")
    d_v         = g("attention.value_length")
    n_embd      = g("embedding_length")
    swa         = g("attention.sliding_window")
    swa_pattern = g("attention.sliding_window_pattern")
    n_ctx_train = g("context_length")

    if L is None or n_head_kv is None:
        sys.exit(f"Metadati di attenzione mancanti per arch '{arch}'.")

    nh_first = n_head[0] if isinstance(n_head, list) else n_head
    if d_k is None or d_v is None:
        if n_embd is None or not nh_first:
            sys.exit("key/value_length assenti e head_dim non derivabile.")
        hd = n_embd // nh_first
        d_k, d_v = d_k or hd, d_v or hd

    # n_head_kv -> lista per-layer (0 sui layer ricorrenti)
    if isinstance(n_head_kv, list):
        kv = (n_head_kv + [n_head_kv[-1]] * (L - len(n_head_kv)))[:L] if len(n_head_kv) < L else n_head_kv[:L]
        hybrid = True
    else:
        kv = [n_head_kv] * L
        hybrid = False

    b_k, b_v = CACHE_BYTES[args.cache_type_k], CACHE_BYTES[args.cache_type_v]
    per_layer = [k * (d_k * b_k + d_v * b_v) for k in kv]   # byte/token per layer
    naive_pt  = sum(per_layer)
    attn_layers = sum(1 for x in kv if x > 0)
    recur = L - attn_layers

    # --- classificazione layer global vs sliding (per il tetto SWA) ---
    is_global, method = None, None
    if swa:
        if hybrid and len(set(k for k in kv if k > 0)) >= 2:
            gmin = min(k for k in kv if k > 0)
            is_global = [k == gmin for k in kv]            # Gemma: global = meno teste KV
            method = f"teste KV minime ({gmin}) = global"
        else:
            P = swa_pattern or (6 if str(arch).startswith("gemma") else None)
            if P:
                is_global = [(i % P) == (P - 1) for i in range(L)]
                is_global[-1] = True                        # ultimo layer sempre global
                method = f"pattern {P} (1 global ogni {P}, ultimo global)"

    # --- report ---
    print(f"Modello       : {args.gguf}")
    print(f"Architettura  : {arch}")
    print(f"Layer totali  : {L}  (attenzione: {attn_layers}, ricorrenti/altro: {recur})")
    print(f"head_count_kv : {'per-layer ' + str(sorted(set(kv))) if hybrid else n_head_kv}")
    print(f"key_length    : {d_k}    value_length: {d_v}"
          + (f"    sliding_window: {swa}" if swa else ""))
    print(f"cache K / V   : {args.cache_type_k} ({b_k:.4f} B/el)  /  {args.cache_type_v} ({b_v:.4f} B/el)")
    print("-" * 60)

    ctx = args.ctx or n_ctx_train

    if swa and is_global is not None and ctx and ctx > swa:
        n_glob   = sum(is_global)
        grow_bt  = sum(per_layer[i] for i in range(L) if is_global[i])
        loc_bt   = sum(per_layer[i] for i in range(L) if not is_global[i])
        fixed    = loc_bt * min(ctx, swa)
        total    = grow_bt * ctx + fixed
        print(f"Layer global  : {n_glob}   sliding (@{swa}): {L - n_glob}   [{method}]")
        print(f"Cresce con N  : {human(grow_bt)}/token")
        print(f"Fisso (locali): {human(fixed)}  (saturi a {swa} token)")
        print(f"KV @ {ctx} tok : {human(total)}   [stima SWA-aware]")
        print(f"  upper bound (tutti pieni): {human(naive_pt * ctx)}")
    else:
        print(f"KV per token  : {human(naive_pt)}   ({naive_pt:.0f} byte)")
        if ctx:
            tag = "" if args.ctx else " (context_length del modello)"
            extra = "   [!] SWA presente ma non classificabile -> LIMITE SUPERIORE" \
                    if (swa and is_global is None) else ""
            print(f"KV @ {ctx} tok{tag}: {human(naive_pt * ctx)}{extra}")

    if recur > 0:
        ssm = [k for k in reader.fields if k.startswith(f"{arch}.ssm.")]
        print("-" * 60)
        print(f"[!] {recur} layer ricorrenti (Mamba/linear): stato fisso a parte, non in KV. "
              f"Chiavi SSM: {', '.join(ssm) or 'n/d'}.")

    print('\nVerita\' a runtime:  llama-server -m <gguf> -c <N>   -> riga "KV self size".')


if __name__ == "__main__":
    main()

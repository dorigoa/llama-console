#!/usr/bin/env python3
"""
kv_per_token.py
Stima la dimensione della KV-cache (per token e a contesto pieno) di un modello GGUF,
leggendo i metadati col parser ufficiale `gguf` (nessun caricamento dei pesi).

Installazione:  pip install gguf
Uso:
    python kv_per_token.py modello.gguf
    python kv_per_token.py modello.gguf --ctx 262144
    python kv_per_token.py modello.gguf --ctx 32768 --cache-type-k q8_0 --cache-type-v q8_0

Formula:  byte/token = sum_layer [ n_kv * d_k * b_k  +  n_kv * d_v * b_v ]
          (n_kv = teste KV del layer; d_k/d_v = key/value_length; b = byte/elemento)

NOTA: per sliding-window e attenzione ibrida (Gemma 4, Qwen3.x hybrid, Nemotron-H, ...)
il risultato e' un LIMITE SUPERIORE / approssimato. Il valore reale e' quello stampato
da llama-server all'avvio (riga "KV self size").
"""

import argparse
import sys

try:
    from gguf import GGUFReader, GGUFValueType
except ImportError:
    sys.exit("Manca il pacchetto 'gguf'. Installa con:  pip install gguf")

# byte per elemento per i tipi di cache K/V di llama.cpp
# (i quantizzati usano blocchi da 32 valori: byte_per_blocco / 32)
CACHE_BYTES = {
    "f32":    4.0,
    "f16":    2.0,
    "bf16":   2.0,
    "q8_0":   34 / 32,   # 1.0625
    "q5_1":   24 / 32,   # 0.75
    "q5_0":   22 / 32,   # 0.6875
    "q4_1":   20 / 32,   # 0.625
    "q4_0":   18 / 32,   # 0.5625
    "iq4_nl": 18 / 32,   # 0.5625
}


def _norm(v):
    """Normalizza i tipi numpy/bytes in tipi Python nativi (ricorsivo per gli array)."""
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
    """Ritorna il valore di un campo di metadati (scalare/array/stringa) o None."""
    field = reader.fields.get(key)
    if field is None:
        return None
    # 1) API moderna del pacchetto gguf
    if hasattr(field, "contents"):
        try:
            return _norm(field.contents())
        except Exception:
            pass
    # 2) fallback manuale (copre scalare numerico, stringa, array numerico/stringa)
    t = field.types[0]
    if t == GGUFValueType.ARRAY:
        et = field.types[1]
        out = []
        for i in field.data:
            p = field.parts[i]
            out.append(bytes(p).decode("utf-8", "replace") if et == GGUFValueType.STRING
                       else int(p[0]))
        return out
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
    ap.add_argument("gguf", help="percorso del file .gguf")
    ap.add_argument("--ctx", type=int, default=None,
                    help="contesto N per il totale (default: context_length del modello)")
    ap.add_argument("--cache-type-k", default="f16", choices=sorted(CACHE_BYTES),
                    help="tipo cache K (default f16)")
    ap.add_argument("--cache-type-v", default="f16", choices=sorted(CACHE_BYTES),
                    help="tipo cache V (default f16)")
    ap.add_argument("--cache-type", default=None, choices=sorted(CACHE_BYTES),
                    help="imposta K e V insieme")
    args = ap.parse_args()
    if args.cache_type:
        args.cache_type_k = args.cache_type_v = args.cache_type

    reader = GGUFReader(args.gguf)
    arch = field_value(reader, "general.architecture")
    if not arch:
        sys.exit("Impossibile leggere general.architecture (file GGUF valido?).")

    g = lambda suf: field_value(reader, f"{arch}.{suf}")
    L           = g("block_count")
    n_head      = g("attention.head_count")
    n_head_kv   = g("attention.head_count_kv")
    d_k         = g("attention.key_length")
    d_v         = g("attention.value_length")
    n_embd      = g("embedding_length")
    swa         = g("attention.sliding_window")
    n_ctx_train = g("context_length")

    if L is None or n_head_kv is None:
        sys.exit(f"Metadati di attenzione mancanti per arch '{arch}'.")

    # head_dim di default se key/value_length assenti: n_embd / n_head(query)
    nh_first = n_head[0] if isinstance(n_head, list) else n_head
    if d_k is None or d_v is None:
        if n_embd is None or not nh_first:
            sys.exit("key/value_length assenti e impossibile derivare head_dim.")
        hd = n_embd // nh_first
        d_k = d_k or hd
        d_v = d_v or hd

    # n_head_kv -> lista per-layer (gli array sui modelli ibridi hanno 0 sui layer ricorrenti)
    if isinstance(n_head_kv, list):
        kv = list(n_head_kv)
        kv = (kv + [kv[-1]] * (L - len(kv)))[:L] if len(kv) < L else kv[:L]
        hybrid = True
    else:
        kv = [n_head_kv] * L
        hybrid = False

    sum_kv      = sum(kv)
    attn_layers = sum(1 for x in kv if x > 0)
    recur       = L - attn_layers
    b_k, b_v    = CACHE_BYTES[args.cache_type_k], CACHE_BYTES[args.cache_type_v]
    per_token   = sum_kv * (d_k * b_k + d_v * b_v)

    print(f"Modello       : {args.gguf}")
    print(f"Architettura  : {arch}")
    print(f"Layer totali  : {L}  (attenzione: {attn_layers}, ricorrenti/altro: {recur})")
    if hybrid:
        print(f"head_count_kv : per-layer {sorted(set(kv))}  somma={sum_kv}  -> IBRIDO")
    else:
        print(f"head_count_kv : {n_head_kv} (uniforme)")
    print(f"key_length    : {d_k}    value_length: {d_v}")
    print(f"cache K / V   : {args.cache_type_k} ({b_k:.4f} B/el)  /  {args.cache_type_v} ({b_v:.4f} B/el)")
    print("-" * 60)
    print(f"KV per token  : {human(per_token)}   ({per_token:.0f} byte)")

    ctx = args.ctx or n_ctx_train
    if ctx:
        tag = "" if args.ctx else " (context_length del modello)"
        print(f"KV @ {ctx} tok{tag}: {human(per_token * ctx)}")

    notes = []
    if swa:
        notes.append(f"sliding_window={swa}: i layer locali si saturano alla finestra -> "
                     f"il totale a contesto lungo e' SOVRASTIMATO.")
    if hybrid and len(set(kv)) > 1:
        notes.append("Attenzione ibrida: se anche key/value_length variano per layer "
                     "(es. Gemma 4: 256 locali / 512 globali + trucco K=V), i metadati scalari "
                     "NON lo catturano -> fidati del valore di llama-server.")
    if recur > 0:
        ssm = [k for k in reader.fields if k.startswith(f"{arch}.ssm.")]
        notes.append(f"{recur} layer ricorrenti (Mamba/linear) non entrano nella KV "
                     f"(stato fisso a parte). Chiavi SSM: {', '.join(ssm) or 'n/d'}.")
    if notes:
        print("-" * 60)
        for n in notes:
            print("[!] " + n)

    print("\nVerifica reale:  llama-server -m <gguf> -c <N>   e leggi la riga \"KV self size\".")


if __name__ == "__main__":
    main()

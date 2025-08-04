# pip install orjson joblib tqdm
from __future__ import annotations
import orjson, signal, contextlib, sys
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import joblib.parallel                         # for the progress-bar patch

# ───────────────────────── CONFIG (edit if needed) ─────────────────────────
BASE_PATH   = Path("../dataset_2")
COLLECTIONS = ["lastfm", "suno", "udio"]
N_WORKERS   = 20
TIMEOUT_S   = 30               # per-file JSON parse safety limit
# ───────────────────────────────────────────────────────────────────────────


# ── helpers ────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def time_limit(seconds: int):
    """Raise TimeoutError if the `with` block runs > seconds."""
    def _handler(signum, _frame):
        raise TimeoutError
    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def correct_roman(token: str | None) -> str | None:
    if not token:
        return token
    i, prefix = 0, ""
    while i < len(token) and token[i] in {"b", "#"}:
        prefix += token[i]; i += 1
    rest = "".join(ch.upper() if ch.isalpha() else ch for ch in token[i:])
    return prefix + rest


def load_fast(path: Path) -> dict:
    with time_limit(TIMEOUT_S):
        return orjson.loads(path.read_bytes())


def dump_fast(obj: dict, path: Path) -> None:
    path.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def fix_file(path: Path) -> None:
    data = load_fast(path)
    for chord in data.get("chords", []):
        if "sus" not in (chord.get("chord_type", "") + chord.get("chord_name", "")).lower():
            continue
        fh = chord.get("functional_harmony", {})
        if "functional"    in fh: fh["functional"]    = correct_roman(fh["functional"])
        if "roman_numeral" in fh: fh["roman_numeral"] = correct_roman(fh["roman_numeral"])
        chord["functional_harmony"] = fh
    out_path = path.with_name(path.name.replace("_normalized.json", "_corrected.json"))
    dump_fast(data, out_path)


def gather_files() -> list[Path]:
    every: list[Path] = []
    for coll in COLLECTIONS:
        root = BASE_PATH / coll
        if root.is_dir():
            every.extend(root.rglob("*_analysis.json"))
        else:
            print(f"⨯ {root} (missing)", file=sys.stderr)
    return every


# ── joblib ↔ tqdm glue (no pickling of the bar) ────────────────────────────
class _TqdmCallback(joblib.parallel.BatchCompletionCallBack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_bar = _TqdmCallback.tqdm_bar

    def __call__(self, *args, **kwargs):
        self.tqdm_bar.update(n=self.batch_size)
        return super().__call__(*args, **kwargs)

joblib.parallel.BatchCompletionCallBack_old = joblib.parallel.BatchCompletionCallBack
joblib.parallel.BatchCompletionCallBack    = _TqdmCallback  # patch


def main() -> None:
    files = gather_files()
    if not files:
        print("Nothing to do.")
        return

    print(f"{len(files)} files → {N_WORKERS} workers…")
    with tqdm(total=len(files), desc="Correcting", unit="file") as bar:
        _TqdmCallback.tqdm_bar = bar          # expose bar to callback
        Parallel(n_jobs=N_WORKERS, backend="loky", verbose=0)(
            delayed(fix_file)(p) for p in files
        )
    print("✓ All *_corrected.json files written.")


if __name__ == "__main__":
    main()

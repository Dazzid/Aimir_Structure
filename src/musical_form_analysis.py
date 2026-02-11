"""
Musical form analysis — generate {id}_musical_form.lab for every song.

Uses allin1 (all-in-one) for semantic segmentation (intro, verse, chorus,
bridge, solo, outro…).  Distributes work across 8 GPUs via joblib —
two workers per GPU (16 total), processing songs in batches of 10.

Output:  /workspace/dataset/{collection}/{id}/{id}_musical_form.lab

Usage
-----
  python3.11 musical_form_analysis.py              # 16 workers (2/GPU)
  python3.11 musical_form_analysis.py --n_jobs 8   # 8 workers (1/GPU)
"""

import os, sys, gc, glob, shutil, argparse, atexit, threading
from pathlib import Path
from joblib import Parallel, delayed

# ── config ───────────────────────────────────────────────────────────────────
DATASET_BASE    = Path("/workspace/dataset")
AUDIO_BASE      = Path("/mnt/Aimir_HD")
COLLECTIONS     = ["lastfm", "suno", "udio"]
N_GPUS          = 8
WORKERS_PER_GPU = 2
N_WORKERS       = N_GPUS * WORKERS_PER_GPU
BATCH_SIZE      = 10


# ── helpers ──────────────────────────────────────────────────────────────────

def gather_todo():
    """Return dict {collection: [(col, song_id), …]} for songs needing processing."""
    todo = {}
    for col in COLLECTIONS:
        col_dir = DATASET_BASE / col
        if not col_dir.is_dir():
            continue
        items = []
        for song_id in sorted(os.listdir(col_dir)):
            song_dir = col_dir / song_id
            if not song_dir.is_dir():
                continue
            if (song_dir / f"{song_id}_musical_form.lab").exists():
                continue
            if not (AUDIO_BASE / col / "audio" / f"{song_id}.mp3").exists():
                continue
            items.append((col, song_id))
        if items:
            todo[col] = items
    return todo


def count_done():
    """Count existing _musical_form.lab files per collection."""
    counts = {}
    for col in COLLECTIONS:
        col_dir = DATASET_BASE / col
        if not col_dir.is_dir():
            counts[col] = 0
            continue
        counts[col] = sum(
            1 for d in col_dir.iterdir()
            if d.is_dir() and (d / f"{d.name}_musical_form.lab").exists()
        )
    return counts


def fmt_time(t):
    """Format seconds as decimal (matches project .lab convention)."""
    return f"{t:.6f}"


def result_to_lab(result):
    """Convert an allin1 AnalysisResult to .lab string."""
    lines = [f"{fmt_time(s.start)}\t{fmt_time(s.end)}\t{s.label}" for s in result.segments]
    return "\n".join(lines) + "\n"


def reset_dirs(*dirs):
    """Wipe and recreate directories — keeps temp space from ballooning."""
    for d in dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)


def sweep_tmp():
    """Remove every /tmp/allin1_* directory — catch-all for any leaked temps."""
    for p in glob.glob("/tmp/allin1_*"):
        shutil.rmtree(p, ignore_errors=True)


# ── worker ───────────────────────────────────────────────────────────────────

def worker(chunk, worker_id):
    """Process a chunk of songs on one GPU. Writes {id}_musical_form.lab."""
    gpu_id = worker_id % N_GPUS

    # pin to a single GPU (becomes cuda:0 inside this process)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # silence noise from demucs / torchaudio / allin1
    import warnings, logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    os.environ["TQDM_DISABLE"] = "1"
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 1)
    os.dup2(_devnull, 2)
    os.close(_devnull)

    import torch, allin1
    torch.cuda.set_device(0)
    device = "cuda:0"

    demix_dir = Path(f"/tmp/allin1_demix_w{worker_id}")
    spec_dir  = Path(f"/tmp/allin1_spec_w{worker_id}")
    reset_dirs(demix_dir, spec_dir)

    results = []

    try:
        for batch_start in range(0, len(chunk), BATCH_SIZE):
            batch = chunk[batch_start : batch_start + BATCH_SIZE]
            audio_paths = [
                str(AUDIO_BASE / col / "audio" / f"{sid}.mp3")
                for col, sid in batch
            ]

            try:
                batch_results = allin1.analyze(
                    audio_paths,
                    device=device,
                    demix_dir=str(demix_dir),
                    spec_dir=str(spec_dir),
                    keep_byproducts=False,
                    multiprocess=False,
                )
                for (col, sid), res in zip(batch, batch_results):
                    (DATASET_BASE / col / sid / f"{sid}_musical_form.lab").write_text(
                        result_to_lab(res)
                    )
                    results.append((col, sid, "OK"))

            except Exception as e:
                # batch failed → retry one-by-one
                for col, sid in batch:
                    lab_path = DATASET_BASE / col / sid / f"{sid}_musical_form.lab"
                    if lab_path.exists():
                        results.append((col, sid, "OK"))
                        continue
                    try:
                        res = allin1.analyze(
                            str(AUDIO_BASE / col / "audio" / f"{sid}.mp3"),
                            device=device,
                            demix_dir=str(demix_dir),
                            spec_dir=str(spec_dir),
                            keep_byproducts=False,
                            multiprocess=False,
                        )
                        lab_path.write_text(result_to_lab(res))
                        results.append((col, sid, "OK"))
                    except Exception as e2:
                        results.append((col, sid, f"ERROR: {e2}"))

            # reclaim GPU memory + disk space after every batch
            gc.collect()
            torch.cuda.empty_cache()
            reset_dirs(demix_dir, spec_dir)

    finally:
        # ALWAYS clean up — even on crash, OOM, or KeyboardInterrupt
        shutil.rmtree(demix_dir, ignore_errors=True)
        shutil.rmtree(spec_dir, ignore_errors=True)

    return results


# ── progress monitor ─────────────────────────────────────────────────────────

def progress_monitor(total_todo, done_before, stop_event):
    """Background thread: polls disk every 15 s and updates a tqdm bar."""
    from tqdm import tqdm

    bar = tqdm(
        initial=0, total=total_todo,
        desc="musical_form", unit="song",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        dynamic_ncols=True,
    )
    prev = 0

    while not stop_event.is_set():
        counts = count_done()
        done = sum(counts[c] - done_before.get(c, 0) for c in COLLECTIONS)
        if done > prev:
            bar.update(done - prev)
            prev = done
            bar.set_postfix_str(" | ".join(f"{c}: {counts[c]}" for c in COLLECTIONS))
        if prev >= total_todo:
            break
        stop_event.wait(15)

    # final sync
    counts = count_done()
    done = sum(counts[c] - done_before.get(c, 0) for c in COLLECTIONS)
    if done > prev:
        bar.update(done - prev)
    bar.close()

    for col in COLLECTIONS:
        print(f"  {col:8s}: {counts.get(col, 0)} total _musical_form.lab")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import signal

    # register cleanup for normal exit, SIGINT (Ctrl+C), and SIGTERM (kill)
    atexit.register(sweep_tmp)

    def _on_signal(signum, _frame):
        name = signal.Signals(signum).name
        print(f"\n{name} received — cleaning /tmp/allin1_* …")
        sweep_tmp()
        sys.exit(1)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # wipe any leftovers from a previous crashed run
    sweep_tmp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=N_WORKERS,
                        help=f"Number of parallel workers (default: {N_WORKERS})")
    args = parser.parse_args()
    n_jobs = args.n_jobs

    # ── survey ───────────────────────────────────────────────────────────
    done_before = count_done()
    todo_by_col = gather_todo()

    print("─── Musical form analysis ───")
    for col in COLLECTIONS:
        col_dir = DATASET_BASE / col
        n_dirs = sum(1 for d in col_dir.iterdir() if d.is_dir()) if col_dir.is_dir() else 0
        n_todo = len(todo_by_col.get(col, []))
        n_done = done_before.get(col, 0)
        print(f"  {col:8s}: {n_done:>6d} done / {n_dirs:>6d} total  →  {n_todo:>6d} remaining")

    todo = [item for items in todo_by_col.values() for item in items]
    total_todo = len(todo)

    print(f"\nTotal remaining: {total_todo}")
    print(f"Workers: {n_jobs} ({n_jobs // N_GPUS} per GPU)  |  Batch size: {BATCH_SIZE}")
    print(f"Output:  {{id}}_musical_form.lab\n")

    if not todo:
        print("Nothing to do — all _musical_form.lab files exist.")
        sys.exit(0)

    # ── split round-robin across workers ─────────────────────────────────
    chunks = [[] for _ in range(n_jobs)]
    for i, item in enumerate(todo):
        chunks[i % n_jobs].append(item)

    # ── progress monitor ─────────────────────────────────────────────────
    stop_evt = threading.Event()
    mon = threading.Thread(target=progress_monitor,
                           args=(total_todo, done_before, stop_evt),
                           daemon=True)
    mon.start()

    # ── run ──────────────────────────────────────────────────────────────
    all_results = None
    try:
        all_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(worker)(chunk, wid) for wid, chunk in enumerate(chunks)
        )
    finally:
        # catch-all: wipe every /tmp/allin1_* dir regardless of success/failure
        sweep_tmp()
        stop_evt.set()
        mon.join(timeout=10)

    # ── summary ──────────────────────────────────────────────────────────
    if all_results is None:
        print("\nParallel processing failed — all temps cleaned. Re-run to retry.")
        sys.exit(1)

    ok  = sum(1 for rl in all_results for r in rl if r[2] == "OK")
    err = sum(1 for rl in all_results for r in rl if r[2] != "OK")
    print(f"\nDone.  OK: {ok}  |  Errors: {err}")

    if err:
        print("\nFailed songs:")
        for rl in all_results:
            for col, sid, status in rl:
                if status != "OK":
                    print(f"  {col}/{sid}: {status}")
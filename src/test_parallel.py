from joblib import Parallel, delayed
import os, time

def worker(i):
    print(f"Worker {i} - PID: {os.getpid()}")
    time.sleep(3)
    return i * i

if __name__ == "__main__":
    Parallel(n_jobs=10, backend="multiprocessing")(
        delayed(worker)(i) for i in range(20)
    )
    print("All workers completed.")
"""Pre-cache raw continuous evaluation windows for all 24 cases in parallel."""

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.raw_window_dataset import RawWindowCache, SUBJECT_PAIRS

def main():
    cache = RawWindowCache()
    cases = sorted(list(SUBJECT_PAIRS.keys()))
    print(f"Caching evaluation windows for all {len(cases)} cases...")
    
    def process_case(c):
        t0 = time.time()
        X, y, meta = cache.get_or_extract_evaluation_windows(c, verbose=False)
        t1 = time.time()
        print(f"  [{c}] Cached {len(y)} eval windows in {t1 - t0:.2f} s")
        return c, len(y)

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(process_case, cases))
    t_end = time.time()
    
    tot_windows = sum(r[1] for r in results)
    print(f"\nAll {len(cases)} cases evaluation cached! Total windows: {tot_windows} across cohort. Time: {t_end - t_start:.2f} s")

if __name__ == "__main__":
    main()

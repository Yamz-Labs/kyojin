"""Minimal driver guard for the Inkling 2 job drivers (decode_job.py, backend_ops.py).

main_guard: an uncaught exception prints a traceback and a DRIVER_FAIL line and exits non-zero.
settle_and_warm: measuring rule, idle IDLE_S seconds, then one warm decode whose result is discarded.
A job runner may ship its own drv_guard with more checks: put its directory in INK_GUARD_DIR and it wins over this file.
"""
import sys
import time
import traceback

IDLE_S = 20
RC_FAIL = 1


def main_guard(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("DRIVER_FAIL KeyboardInterrupt", flush=True)
        sys.exit(130)
    except BaseException as e:
        traceback.print_exc()
        print(f"DRIVER_FAIL {type(e).__name__}: {e}", flush=True)
        sys.exit(RC_FAIL)


def settle_and_warm(warm_fn=None, idle_s=None, label=""):
    n = IDLE_S if idle_s is None else idle_s
    print(f"MEASURE_RULE settle {n} s then one warm decode {label}".rstrip(), flush=True)
    time.sleep(n)
    if warm_fn is not None:
        warm_fn()
        print(f"MEASURE_RULE warm decode done {label}".rstrip(), flush=True)

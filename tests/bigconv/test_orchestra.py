"""Operational lessons of the live DS4.1 conversion, as CPU tests (stub quantizers, no GPU)."""
import json, os, struct, subprocess, sys, textwrap, time
import pytest

from bigconv import orchestra as O
from bigconv.orchestra import Layout, StepFailed


def mk_spool(lay, L, done=True):
    d = lay.spool_dir(L)
    os.makedirs(d, exist_ok=True)
    if done:
        open(os.path.join(d, "CAPTURE_DONE"), "w").close()


def finish(lay, L, how="done"):
    os.makedirs(lay.claims, exist_ok=True)
    if how == "done":
        open(lay.claim(L) + ".done", "w").write("{}")
    else:
        os.makedirs(lay.claim(L), exist_ok=True)
        open(os.path.join(lay.claim(L), "COPIED"), "w").close()


# ---- lead / sweeper: the deadlock (capture counted COPIED spools, only swept between captures)
@pytest.mark.parametrize("how", ["done", "copied"])
def test_finished_spools_do_not_block_capture(tmp_path, how):
    lay = Layout(str(tmp_path))
    for L in (2, 3, 6):
        mk_spool(lay, L)
        finish(lay, L, how)
    assert O.pending_spools(lay) == []
    assert O.capture_may_start(lay, lead=3, free_gb=500, floor_gb=60, spool_gb=30)


def test_sweeper_removes_finished_spools_independently(tmp_path):
    lay = Layout(str(tmp_path))
    for L in (1, 2, 3):
        mk_spool(lay, L)
    finish(lay, 1, "done")
    finish(lay, 3, "copied")
    sw = O.Sweeper(lay, period=0.05)
    sw.start()
    t = time.time()
    while os.path.isdir(lay.spool_dir(1)) and time.time() - t < 5:
        time.sleep(0.05)
    sw.stop()
    assert not os.path.isdir(lay.spool_dir(1)) and not os.path.isdir(lay.spool_dir(3))
    assert os.path.isdir(lay.spool_dir(2))          # unclaimed spool is never swept


def test_lead_never_below_two_and_disk_pauses_capture(tmp_path):
    lay = Layout(str(tmp_path))
    assert O.effective_lead(1) == 2 and O.effective_lead(0) == 2 and O.effective_lead(5) == 5
    mk_spool(lay, 0)
    # lead 1 would have blocked here (one pending spool); with the floor of 2 the capture runs the next layer
    assert O.capture_may_start(lay, lead=1, free_gb=500, floor_gb=60, spool_gb=30)
    mk_spool(lay, 1)
    assert not O.capture_may_start(lay, lead=1, free_gb=500, floor_gb=60, spool_gb=30)
    # disk: 100 GB free, floor 60, 30 GB per spool -> room for 1 only
    assert O.capture_cap(5, 100, 60, 30) == 1
    # a spool still being written (no CAPTURE_DONE) is not pending
    mk_spool(lay, 5, done=False)
    assert 5 not in O.pending_spools(lay)


# ---- prefetch: refill a free slot at any time
def test_prefetch_refills_free_slot_not_only_at_layer_start():
    # one layer quantizing (resident), a second copy just finished and was consumed -> slot free, next layer starts at once
    assert O.plan_prefetch([4, 5, 6], resident={4}, slots=2, free_gb=400, floor_gb=60, spool_gb=30) == [5]
    assert O.plan_prefetch([4, 5, 6], resident={4, 5}, slots=2, free_gb=400, floor_gb=60, spool_gb=30) == []
    assert O.plan_prefetch([4, 5, 6], resident=set(), slots=2, free_gb=400, floor_gb=60, spool_gb=30) == [4, 5]
    # disk limits the slots
    assert O.plan_prefetch([4, 5, 6], resident=set(), slots=2, free_gb=100, floor_gb=60, spool_gb=30) == [4]
    assert O.plan_prefetch([4], resident=set(), slots=2, free_gb=70, floor_gb=60, spool_gb=30) == []


# ---- claims: atomic, resume own, FAILED -> resume from partial
def test_claim_is_atomic_and_owner_resumes_first(tmp_path):
    lay = Layout(str(tmp_path))
    for L in (0, 1, 2):
        mk_spool(lay, L)
    assert O.try_claim(lay, 1, "a") and not O.try_claim(lay, 1, "b")
    assert O.pick_work(lay, "a", 40) == (1, "resume")
    assert O.pick_work(lay, "b", 40) == (0, "new")


def test_failed_claim_resumes_and_keeps_partial(tmp_path):
    lay = Layout(str(tmp_path))
    mk_spool(lay, 6)
    assert O.try_claim(lay, 6, "hostb")
    partial = tmp_path / "out" / "partial" / "layers.6"
    partial.mkdir(parents=True)
    (partial / "lin0.pt").write_text("x")
    O.mark_failed(lay, 6, "IncompleteRead")
    assert O.pick_work(lay, "other", 40)[0] is None              # a foreign FAILED claim is never taken
    assert O.pick_work(lay, "hostb", 40) == (6, "retry")
    assert (partial / "lin0.pt").exists()                          # partial/ untouched
    assert not os.path.exists(os.path.join(lay.claim(6), "FAILED"))
    # retry cap
    for _ in range(5):
        O.mark_failed(lay, 6, "again")
        O.recover_failed(lay, "hostb")
    assert O.failed_count(lay, 6) == 3
    assert os.path.exists(os.path.join(lay.claim(6), "FAILED"))
    assert O.pick_work(lay, "hostb", 40)[0] is None


def test_stale_claims_by_heartbeat(tmp_path):
    lay = Layout(str(tmp_path))
    O.try_claim(lay, 3, "a")
    hb = os.path.join(lay.claim(3), "hb")
    old = time.time() - 5000
    os.utime(hb, (old, old))
    assert O.stale_claims(lay, stale_s=900) == [3]
    O.touch(hb)
    assert O.stale_claims(lay, stale_s=900) == []


# ---- shared-GPU gate (iGPU quantizer beside the capture): truth table
@pytest.mark.parametrize("running,blocked,free,own,waits,want", [
    (False, False, 0, [], False, True),      # capture finished: free to run
    (True, False, 5, [], False, False),      # capture not blocked
    (True, True, 1, [], False, False),       # blocked but dedicated cards would starve (1 free layer)
    (True, True, 2, [], False, True),        # blocked and 2 free layers
    (True, True, 0, [3], False, True),       # blocked: always finish own claim
    (True, True, 5, [3], True, False),       # capture waits for the GPU lock: never hold it
])
def test_shared_gpu_gate(running, blocked, free, own, waits, want):
    assert O.shared_gpu_gate(running, blocked, free, own, waits) is want


def test_blocked_follows_the_capture_pause_rule_including_disk(tmp_path):
    lay = Layout(str(tmp_path))
    mk_spool(lay, 0)
    mk_spool(lay, 1)
    # 2 pending, lead 3, plenty of disk: capture runs -> not blocked -> iGPU stays out
    assert not O.capture_blocked(lay, 3, 500, 60, 30)
    assert not O.shared_gpu_gate(True, O.capture_blocked(lay, 3, 500, 60, 30), 2, [])
    # same 2 pending, disk room for only 2 -> capture paused on DISK although pending < lead -> blocked -> iGPU may work
    assert O.capture_blocked(lay, 3, 130, 60, 30)
    assert O.shared_gpu_gate(True, O.capture_blocked(lay, 3, 130, 60, 30), 2, [])


# ---- idle-kill exemption of the GPU wrapper
def test_idle_exempt_matches_wrapper_case_line():
    txt = 'x\n  case "$who" in workerq*) idle=0;; esac   # note\n'
    assert O.idle_exempt("workerq2", txt) and not O.idle_exempt(O.gpurun_who(0), txt)
    txt2 = 'case "$who" in workerq*|bigconv-q*) idle=0;; esac'
    assert O.idle_exempt(O.gpurun_who(1), txt2)


# ---- verified steps: rc, traceback with rc 0, missing outputs
def py(code):
    return [sys.executable, "-c", textwrap.dedent(code)]


def test_traceback_with_rc_zero_is_a_failure():
    cmd = py("""
        import sys, traceback
        try:
            raise ValueError('boom')
        except ValueError:
            traceback.print_exc()
        sys.exit(0)""")
    with pytest.raises(StepFailed, match="traceback"):
        O.checked_run(cmd, out=open(os.devnull, "w"))


def test_handled_traceback_on_a_good_run_is_not_a_failure(tmp_path):
    out = str(tmp_path)
    write_shard(out, 4)
    cmd = py("""
        import traceback
        try:
            1 / 0
        except ZeroDivisionError:
            traceback.print_exc()        # handled and logged, run continues (as a worker does for a re-queued job)
        print('layer done')""")
    r = O.checked_run(cmd, verify=lambda: O.verify_shard(out, 4, 1000), out=open(os.devnull, "w"))
    assert r.rc == 0


def test_rc_zero_without_outputs_is_a_failure(tmp_path):
    lay_out = str(tmp_path)
    cmd = py("print('quantized 1165/1165')")
    with pytest.raises(StepFailed):
        O.checked_run(cmd, verify=lambda: O.verify_shard(lay_out, 3), out=open(os.devnull, "w"))


def write_shard(out, L, nbytes=2000):
    hdr = json.dumps({"a": {"dtype": "U8", "shape": [nbytes], "data_offsets": [0, nbytes]}}).encode()
    open(f"{out}/layers.{L}.safetensors", "wb").write(struct.pack("<Q", len(hdr)) + hdr + b"\0" * nbytes)
    json.dump({"ok": 1}, open(f"{out}/layers.{L}.done.json", "w"))


def test_verify_shard(tmp_path):
    out = str(tmp_path)
    write_shard(out, 3)
    assert O.verify_shard(out, 3, min_bytes=1000)["keys"] == 1
    with pytest.raises(StepFailed):
        O.verify_shard(out, 3, min_bytes=1e8)


def test_holds_resume_until_clean(tmp_path):
    out = str(tmp_path)
    state = tmp_path / "n"
    state.write_text("0")
    stub = f"""
        import os, sys, json
        n = int(open({str(state)!r}).read()) + 1
        open({str(state)!r}, 'w').write(str(n))
        if n < 3:
            os._exit(76)
        sys.exit(0)"""
    write_shard(out, 7)
    holds = O.run_holds(lambda: py(stub), verify=lambda: O.verify_shard(out, 7, 1000), out=open(os.devnull, "w"))
    assert holds == 3
    with pytest.raises(StepFailed, match="holds"):
        state.write_text("-100")
        O.run_holds(lambda: py(stub), verify=lambda: None, max_holds=2, out=open(os.devnull, "w"))


def test_real_failure_is_not_resumed():
    with pytest.raises(StepFailed, match="rc=3"):
        O.run_holds(lambda: py("import sys; sys.exit(3)"), verify=lambda: None, out=open(os.devnull, "w"))


def test_exit_guard_keeps_rc_when_an_import_swallows_exit(tmp_path):
    """The engine import swallows sys.exit codes (rc 0): the guard leaves with os._exit."""
    script = tmp_path / "q.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/tools'!r})
        import atexit, os
        atexit.register(lambda: os._exit(0))       # what the engine import effectively does: rc 0 at teardown
        from bigconv.orchestra import exit_guard
        def main():
            raise RuntimeError('quantizer crashed')
        exit_guard(main)
    """))
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert r.returncode == 1 and "quantizer crashed" in r.stderr
    plain = tmp_path / "p.py"
    plain.write_text(script.read_text().replace("exit_guard(main)", "main()"))
    assert subprocess.run([sys.executable, str(plain)], capture_output=True).returncode == 0   # without the guard: masked


def test_cross_hardware_acceptance_uses_proxy_error_not_bytes():
    assert O.accept_cross_hw(3.0e-4, 3.0e-4) and O.accept_cross_hw(3.2e-4, 3.0e-4)
    assert not O.accept_cross_hw(4.5e-4, 3.0e-4)

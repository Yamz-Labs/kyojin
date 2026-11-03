"""Operational defaults of the big-model converter. Each value is a lesson from a live conversion; change one only with a measurement."""

DEFAULTS = dict(
    # calibration: 250 rows x 2048 tokens (500K). Owner decision 19/10: marginal loss accepted (proxy: 2x tokens = -5.7 % KL) for half the capture time and spool
    rows=250,
    row_len=2048,
    # capture --lead / --max-pending: how many captured-but-unfinished spools may sit on disk. 1 serialises capture and
    # quantizers (capture waits for the previous layer to be fully quantized); never below 2.
    lead=3,
    min_lead=2,
    # spool copy slots per quantizer host, and quantizer processes per host (alternating GPU holds, copies overlap)
    prefetch_slots=2,
    workers_per_host=2,
    # disk
    floor_gb=60,
    # sustained write rate of a DRAM-less QLC SSD once its cache is full (measured: 4.5 GB/s burst -> ~300 MB/s)
    qlc_sustained_write_mbs=300,
    # capture scratch (row-state / X / Y memmaps) lives on tmpfs, never on the SSD; size it from the sheet's memmap disk figure
    capture_scratch="tmpfs",
    # quantizer exit codes meaning "budget hold reached, resumable"
    hold_codes=(75, 76),
    # shared-GPU secondary quantizer: needs at least this many captured, unclaimed layers while the capture is blocked
    shared_gpu_min_free_layers=2,
    # claims: heartbeat period / staleness, FAILED retries (a retry resumes from partial/, never restarts)
    hb_period_s=60,
    hb_stale_s=900,
    max_failed_retries=3,
    # GPU wrapper: --who prefix of long quantizer jobs that have CPU-only phases (must be exempt from the idle kill)
    quantizer_who_prefix="bigconv-q",
    # other hardware (different GPU family) is accepted on the proxy error, not on the bytes: ratio vs the reference arm
    cross_hw_proxy_ratio=1.10,
    # vm.dirty_* upper bounds (kernel defaults). Raising them hard-hung a host (big memmap + dirty 90).
    dirty_ratio_max=20,
    dirty_background_ratio_max=10,
)

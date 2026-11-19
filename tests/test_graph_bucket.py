"""CPU test (no torch, no GPU): python test_graph_bucket.py TREE
Context buckets of the decode graphs: >= rows needed, multiple of the base granule, monotone, <= cap, <= 1/32 masked rows above 16K, `buckets(cap)` lists exactly the
buckets `bucket()` can return, and the fixed-granule behaviour is unchanged below 16K / with REL = 0."""
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("gb", os.path.join(sys.argv[1], "exllamav3", "loader", "graph_bucket.py"))
gb = importlib.util.module_from_spec(spec); spec.loader.exec_module(gb)
B = 512
for cap in (135168, 528384, 1048576 + 512):
    cap -= cap % B
    allb = gb.buckets(cap, B, 32, 16384)
    assert allb == sorted(set(allb)) and allb[-1] == cap and all(b % B == 0 for b in allb)
    seen, prev = set(), 0
    for n in list(range(1, 40000, 37)) + list(range(40000, cap + 1, 997)) + [cap]:
        b = gb.bucket(n, cap, B, 32, 16384)
        assert b >= n and b <= cap and b % B == 0 and b >= prev and b in allb, (n, b)
        if n > 16384 and b < cap:
            assert (b - n) <= max(B, n // 32 + B), (n, b)
        if n <= 16384:
            assert b == min(cap, -(-n // B) * B)
        prev = b
        seen.add(b)
    print(f"cap {cap}: {len(allb)} buckets (fixed 512 would be {cap // B}), sample 262144+1 -> {gb.bucket(262145, cap, B, 32, 16384)}")
assert gb.buckets(135168, B, 0, 16384) == list(range(B, 135168 + 1, B))
assert gb.bucket(262145, 528384, B, 0, 16384) == 262144 + 512
print("OK")

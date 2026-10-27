"""
GDN speculative rewind: the batched per-round dispatch (rewind_batch_begin / rewind_batch_end) and the memoized job
descriptors give the same conv and recurrent state, bit for bit, as the per-layer GDNLayerState.rewind() reference,
with the zero-copy rebase on and off. Needs a CUDA device.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace
import torch
from exllamav3.modules import gated_delta_net as gdn

dev = torch.device("cuda:0")
L, SLOTS, HIST, CD = 6, 4, 5, 4
mod = SimpleNamespace(fdim_qkv = 256, conv_kernel_size = CD, num_v_heads = 4, k_head_dim = 32, v_head_dim = 32)

def build(seed):
    g = torch.Generator(device = dev).manual_seed(seed)
    layers = {}
    for i in range(L):
        l = gdn.GDNLayerState(mod, SLOTS, HIST, cache_id = 1000 + seed)
        l.alloc(dev)
        l.conv_state.copy_(torch.randn(l.conv_state.shape, device = dev, generator = g).bfloat16())
        l.recurrent_state.copy_(torch.randn(l.recurrent_state.shape, device = dev, generator = g))
        layers[i] = l
    cache = SimpleNamespace(model = SimpleNamespace(loaded_tp = False), get_all_recurrent_layers = lambda: layers,
                            __dict__ = {})
    cache.get_all_recurrent_layers = lambda: layers
    return cache, layers

def snap(layers):
    return [(l.conv_state.clone(), l.recurrent_state.clone(), l.state_base.clone()) for l in layers.values()]

# (slot, last_history, num_tokens) per user of one round; the second round repeats keys to hit the memo
rounds = [[(0, 4, 2), (1, 4, 0), (2, 3, 3)], [(0, 4, 2), (1, 4, 1), (3, 2, 1)], [(0, 4, 2), (1, 4, 0), (2, 3, 3)]]

for rebase in ("0", "1"):
    os.environ["EXL3_STATE_REBASE"] = rebase
    gdn._base_value.clear()
    ref_cache, ref_layers = build(1)
    new_cache, new_layers = build(1)
    for r, users in enumerate(rounds):
        for slot, lh, nt in users:      # reference: the per-layer rewind of each user, immediately
            for l in ref_layers.values(): l.rewind(slot, lh, nt)
        # new: GDNState.rewind() deferred to one dispatch, run twice per key to take the memoized path
        gdn.rewind_batch_begin()
        for slot, lh, nt in users:
            st = gdn.GDNState(new_cache, slot, 10, clear = False, test_state = True)
            st.last_history = lh
            st.rewind(nt)
            assert st.position == 10 - nt and st.last_history == 0
        gdn.rewind_batch_end()
        torch.cuda.synchronize()
        for (c0, s0, b0), (c1, s1, b1) in zip(snap(ref_layers), snap(new_layers)):
            assert torch.equal(c0, c1), (rebase, r, "conv")
            assert torch.equal(b0, b1), (rebase, r, "base")
            if rebase == "0":
                assert torch.equal(s0, s1), (rebase, r, "state")
            else:   # rebase: the copy is skipped, the committed index moves; compare the committed state of every slot
                for slot in range(SLOTS):
                    assert torch.equal(s0[slot, int(b0[slot])], s1[slot, int(b1[slot])]), (rebase, r, slot)
    # immediate dispatch (no batch open) and a changed allocation (epoch) still agree
    for l in new_layers.values(): l.free(); l.alloc(dev)
    assert gdn._rw_defer is None
print("gdn rewind batch: ok")

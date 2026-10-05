"""
CPU regression test for the PLE checkpoint stash: id_state lives on the CPU, so stash() must return a
copy. A stashed state must not change when the slot advances (decode, next job), and unstash must
restore the ids that were live at stash time. Fails if the .clone() in PLELayerState.stash is removed.
"""
import os, sys
from types import SimpleNamespace
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.modules.ple import PLELayerState


def make_state():
    module = SimpleNamespace(
        conv_state_len = 3,
        hc_mult = 2,
        hidden_size = 4,
        ple_embedding = SimpleNamespace(context_len = 5, eos_token_id = 7),
    )
    st = PLELayerState(module, max_batch_size = 2, max_history = 4, cache_id = 0)
    st.alloc("cpu")
    return st


def test_stash_is_a_copy():
    st = make_state()
    slot = 1
    st.id_state[slot, :st.ctx] = torch.arange(100, 100 + st.ctx)
    ids_at_stash = st.id_state[slot, :st.ctx].clone()
    stashed = st.stash(slot)

    # slot advances
    st.id_state[slot, :st.ctx] = torch.arange(900, 900 + st.ctx)

    assert torch.equal(stashed[1], ids_at_stash), "stashed id_state changed when the slot advanced"

    st.unstash(slot, stashed)
    assert torch.equal(st.id_state[slot, :st.ctx], ids_at_stash)


if __name__ == "__main__":
    test_stash_is_a_copy()
    print("ok")

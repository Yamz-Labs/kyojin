import torch


class _CAI:
    def __init__(self, ptr, shape, typestr):
        self.__cuda_array_interface__ = {"shape": tuple(shape), "typestr": typestr, "data": (ptr, False), "version": 2}


_TYPESTR = {torch.half: "<f2", torch.bfloat16: "<V2"}
_KEEP = []


def host_view(w: torch.Tensor, device):
    """CUDA tensor aliasing the host (mmapped) tensor `w`: the pages are registered mapped, the device reads rows over PCIe
    (UVA, no VRAM, no second copy). Valid inside CUDA graphs. Returns None when the driver / dtype refuses."""
    try:
        if w.device.type != "cpu" or not w.is_contiguous() or w.dtype != torch.half:
            return None   # bf16 has no array-interface typestr: use the device copy
        rt = torch.cuda.cudart()
        if int(rt.cudaHostRegister(w.data_ptr(), w.numel() * w.element_size(), 2)) != 0:   # cudaHostRegisterMapped
            torch.cuda.synchronize()
            return None
        g = torch.as_tensor(_CAI(w.data_ptr(), w.shape, _TYPESTR[w.dtype]), device = device)
        _KEEP.append(w)   # the mapping must outlive the view
        return g
    except Exception as ex:
        print(f"[lag1] host-mapped embedding table unavailable: {ex}", flush = True)
        return None

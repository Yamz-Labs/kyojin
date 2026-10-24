from __future__ import annotations
import torch

FIRST_MM_EMBEDDING_INDEX = 1000000000

# Assume no model will have more than one billion regular text tokens, and assign dynamic token IDs starting from
# that index.

class MMTokenAllocator:

    next_token_index: int

    def __init__(self):
        self.next_token_index = FIRST_MM_EMBEDDING_INDEX

    def allocate(self, num_tokens):
        idx = self.next_token_index
        self.next_token_index += num_tokens
        return idx

global_allocator = MMTokenAllocator()


class MMEmbedding:
    """
    Container for one embedding (image etc.) and associated metadata
    """

    def __init__(
        self,
        embeddings: torch.Tensor | None = None,
        token_string: torch.Tensor | None = None,
        text_alias: str | None = None,
        deepstack_embeddings: list[torch.Tensor] | None = None,
        grid_thw: tuple | None = None,
        mrope_merge_size: int | None = None,
        align: int = 0,
        align_phase: int = 0,
        align_lead: int = 0,
        imp: dict | None = None
    ):
        """
        :param embeddings:
            Embeddings, shape (num_tokens, input_dim)

        :param token_string:
            Tokenized representation, with -1 as a placeholder for the MM embeddings

        :param text_alias:
            Text string to represent this embedding for tokenizing

        :param align:
            Position alignment for embeddings whose block must start on a fixed phase of a
            model-side grouping (DeepSeek-V4: the CSA compressor's 4-token groups). The
            token_string then begins with align_lead spare rows (pads) and the tokenizer emits
            only the suffix that places row align_lead at a prompt position p with
            p % align == align_phase. 0 = no alignment.
        """

        if imp:
            self.metadata = imp["metadata"]
            self.full_length = imp["full_length"]
            self.mm_length = imp["mm_length"]
            self.first_index = imp["first_index"]
            self.last_index = imp["last_index"]
            self.text_alias = imp["text_alias"]
            self.grid_thw = imp["grid_thw"]
            self.mrope_merge_size = imp["mrope_merge_size"]
            self.embeddings = imp["embeddings"]
            self.deepstack_embeddings = imp["deepstack_embeddings"]
            self.align = imp.get("align", 0)
            self.align_phase = imp.get("align_phase", 0)
            self.align_lead = imp.get("align_lead", 0)
            self.token_string = None
            self.token_list = None
            return

        global global_allocator

        if deepstack_embeddings is not None:
            assert all(de.shape == embeddings.shape for de in deepstack_embeddings), \
                "Deepstack embeddings shape mismatch"

        self.metadata = {}
        self.full_length = token_string.shape[-1]
        self.mm_length = embeddings.shape[-2]
        self.first_index = global_allocator.allocate(self.mm_length)
        self.last_index = self.first_index + self.mm_length
        self.embeddings = embeddings
        self.deepstack_embeddings = deepstack_embeddings
        self.text_alias = text_alias or f"<$EMB_{self.first_index}$>"

        # MRoPE
        self.grid_thw = grid_thw
        self.mrope_merge_size = mrope_merge_size

        # Position alignment (see align)
        assert align == 0 or (0 <= align_phase < align and 0 <= align_lead < align and align_lead < self.full_length), \
            "MMEmbedding: align_lead / align_phase must be smaller than align"
        self.align = align
        self.align_phase = align_phase
        self.align_lead = align_lead

        # not exported for TP
        r = torch.arange(self.first_index, self.first_index + self.mm_length, dtype = torch.long)
        m = (token_string == -1)
        token_string.masked_scatter_(m, r)
        self.token_string = token_string
        self.token_list = token_string[0].tolist()

    def token_list_at(self, position: int) -> list[int]:
        """Token ids to splice in when the embedding's first emitted row lands at prompt
        position `position`: the whole token_list, or, for aligned embeddings, the suffix that
        skips the spare leading rows not needed to put row align_lead on the required phase."""
        if not self.align:
            return self.token_list
        n_pad = (self.align_phase - position) % self.align
        assert n_pad <= self.align_lead
        return self.token_list[self.align_lead - n_pad:]


# Packed storage modes for the picture cache of a model server, next to fp16: 8 bit is a quarter of the fp32
# rows and stays within half a step of the row's peak (measured 0.86 % relative error over four real pages).
MM_QUANT_MODES = ("q8",)


class QuantizedMMRows:
    """One image's rows held 8 bit on the host, with a per-row fp16 scale.

    A server that keeps the embeddings of a whole conversation in host memory can store them this way
    instead of the fp32 rows the vision tower's merger emits (four bytes per value), which is what the
    picture cache spends its budget on: a quarter of the bytes. The rows go back to their working dtype when
    the model gathers them: `mm_rows` is the only place their values are read, and the scaling happens there,
    so the packed values cross the bus instead of fp32 and the algebra is one multiply per value.

    Setting `MMEmbedding.embeddings` to one of these does not touch the object's pseudo token ids, so a prompt
    spliced from a hit keeps matching the paged KV cache exactly as an unquantized one does. Only the values
    that reach the model change, by the rounding this class introduces: at most half a step, one part in 254
    of the row's peak.
    """

    __slots__ = ("q", "scale", "shape", "dtype", "nbytes")

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
        self.q = q                              # uint8, (rows, hidden), offset by 128
        self.scale = scale                      # fp16, (rows, 1)
        self.shape = tuple(q.shape)
        self.dtype = dtype                      # what the rows were quantized from
        self.nbytes = int(q.numel()) + int(scale.numel()) * int(scale.element_size())

    def numel(self) -> int:
        return self.shape[0] * self.shape[1]

    def element_size(self) -> int:
        return 1

    def rows(self, index: torch.Tensor, out_dtype: torch.dtype | None = None,
             device: torch.device | None = None) -> torch.Tensor:
        """Rows `index` in `out_dtype` (default: the dtype the rows had). `index` may live anywhere: the
        gather is on the host where the rows are, the scaling where the result is wanted. The scale is applied
        in place, which measured ~15 % faster than allocating the product: 6 ms against 5 ms for the same rows
        as fp32, for one 1568x1568 page (3,136 tokens), so re-prefilling one picture costs about half a
        millisecond more."""
        out_dtype = out_dtype or self.dtype
        index = index.to(self.q.device)
        values = self.q[index].to(out_dtype).sub_(128.0)     # stored offset by half the range, so uint8
        scale = self.scale[index].to(out_dtype)
        if device is not None and device.type != "cpu":
            values, scale = values.to(device, non_blocking = True), scale.to(device, non_blocking = True)
        return values.mul_(scale)

    def __repr__(self) -> str:  # keeps logs and errors readable
        rows, hidden = self.shape
        return f"QuantizedMMRows({rows}x{hidden} 8 bit, {self.nbytes / 2 ** 20:.1f} MiB)"


def quantize_rows(tensor: torch.Tensor, mode: str = "fp32"):
    """The stored form of one image's rows for `mode`: fp32 (unchanged), fp16 (half the bytes) or `q8` (8 bit
    with one fp16 scale per row, a quarter).

    fp16 is not a lossy choice for a tower that computes in fp16: its fp32 output is an fp16 value widened, so
    storing it back as fp16 rounds nothing — measured on this pack over four real pages, where every value
    survives `x.half().float() == x` unchanged.

    Symmetric, offset-binary 8 bit: a value costs the same byte whatever its row looks like, and it comes back
    within half a step of where it was, i.e. one part in 254 of the row's peak."""
    if mode in ("fp32", "none", ""):
        return tensor
    if mode == "fp16":
        return tensor.half()
    if mode not in MM_QUANT_MODES:
        raise ValueError(f"unknown image embedding format {mode!r} (fp32, fp16 or {', '.join(MM_QUANT_MODES)})")
    x = tensor.detach()
    if x.dim() != 2:
        x = x.reshape(-1, x.shape[-1])
    scale = (x.abs().amax(dim = 1, keepdim = True) / 127.0).clamp_min(1e-8).to(torch.float16)
    code = torch.round(x / scale.to(x.dtype)).add_(128.0).clamp_(1, 255).to(torch.uint8)
    return QuantizedMMRows(code, scale, x.dtype)


def mm_rows(emb: MMEmbedding, index: torch.Tensor, out_dtype: torch.dtype | None = None,
            device: torch.device | None = None) -> torch.Tensor:
    """Rows `index` of one image embedding in `out_dtype`, dequantizing when the rows are stored packed.
    A plain tensor takes the path it always did."""
    src = emb.embeddings
    if isinstance(src, QuantizedMMRows):
        return src.rows(index, out_dtype, device)
    return src[index].to(out_dtype) if out_dtype is not None else src[index]


def send_embeddings(producer, ies: list[MMEmbedding]):
    return {
        "method": "list",
        "data": [
            {
                "metadata": ie.metadata,
                "full_length": ie.full_length,
                "mm_length": ie.mm_length,
                "first_index": ie.first_index,
                "last_index": ie.last_index,
                "text_alias": ie.text_alias,
                "grid_thw": ie.grid_thw,
                "mrope_merge_size": ie.mrope_merge_size,
                "align": ie.align,
                "align_phase": ie.align_phase,
                "align_lead": ie.align_lead,
                "embeddings": producer.send(ie.embeddings, cache_id = id(ie.embeddings)),
                "deepstack_embeddings": [
                    producer.send(dse, cache_id = id(dse))
                    for dse in ie.deepstack_embeddings
                ] if ie.deepstack_embeddings is not None else None
            }
            for ie in ies
        ]
    }


def recv_embeddings(consumer, recv) -> list[MMEmbedding]:
    result = []
    assert recv.get("method") == "list", "Consumer expected list"
    for imp in recv["data"]:
        imp["embeddings"] = consumer.recv(imp["embeddings"])
        imp["deepstack_embeddings"] = [
            consumer.recv(dse) for dse in imp["deepstack_embeddings"]
        ] if imp.get("deepstack_embeddings") else None
        result.append(MMEmbedding(imp = imp))
    return result
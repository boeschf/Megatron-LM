# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

import logging
import os
import socket
from typing_extensions import override
from torch.distributed import GroupMember, ProcessGroup, ReduceOp, Work
from typing import TypeVar, cast
from collections.abc import Iterator
from contextlib import contextmanager

from megatron.core.utils import internal_api

try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

try:
    from pplx_garden.kernels.p2p_all_to_all import P2PAllToAll
    from pplx_garden.distributed.parallel_group import ParallelGroup
    from pplx_garden.distributed.torch_group import TorchParallelGroup
    from pplx_garden.distributed.nccl_all_reduce import NcclAllReduce
    from pplx_garden.distributed.distributed_ops import Reducer
    from pplx_garden.utils.torch import profile_range

    HAVE_PPLX_GARDEN = True
except ImportError:
    P2PAllToAll = None
    ParallelGroup = object
    TorchParallelGroup = object
    NcclAllReduce = None
    profile_range = lambda name: (lambda x: x)  # no-op context manager
    Reducer = object
    HAVE_PPLX_GARDEN = False

import torch

_buffer = None
_process_group_cache = {}
_logger = logging.getLogger(__name__)

T = TypeVar("T")

def _pplx_debug_enabled() -> bool:
    return os.environ.get("MEGATRON_PPLX_DEBUG", "0") == "1"


def _pplx_debug_log(message: str) -> None:
    if _pplx_debug_enabled():
        _logger.warning(
            "[pplx-debug][rank=%s] %s", torch.distributed.get_rank(), message
        )


def _get_or_create_process_group(ranks, backend):
    key = (tuple(ranks), backend)
    _pplx_debug_log(f"Requesting process group for ranks {ranks} and backend {backend}")
    _pplx_debug_log(f"Current process group cache keys: {_process_group_cache.keys()}")
    _pplx_debug_log(f"Our key is {key}")
    if key not in _process_group_cache:
        _pplx_debug_log(f"Creating new process group for ranks {ranks} and backend {backend}")
        _process_group_cache[key] = torch.distributed.new_group(
            ranks=list(ranks), backend=backend
        )
        _pplx_debug_log(f"Created process group with ranks {torch.distributed.get_process_group_ranks(_process_group_cache[key])} and backend {backend}")

    return _process_group_cache[key]


class _TorchProcessGroupAdapter(ParallelGroup):
    """Wraps an existing Megatron device group as a pplx-garden TorchParallelGroup.

    Unlike TorchParallelGroup, which always creates a new NCCL group, this adapter
    accepts a pre-existing device_group (owned by Megatron) and only creates the
    Gloo cmd_group needed for CPU-side collectives (broadcast_object, etc.).
    is_inter_node is derived from actual node topology, not a size heuristic.
    """

    def __init__(
        self,
        device_group: torch.distributed.ProcessGroup,
        ranks: list,
        node_meta: dict,
    ):
        # Bypass TorchParallelGroup.__init__ — we must not create a new NCCL group.
        # Manually set all attributes that TorchParallelGroup.__init__ would set.
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
        self._global_rank = torch.distributed.get_rank()
        self._node_rank = node_meta["node_rank"]
        self._is_inter_node = node_meta["num_nodes"] > 1 # TODO: maybe change this

        self._ranks = list(ranks)
        self._size = len(self._ranks)
        self._rank = self._ranks.index(self._global_rank)

        # Reuse the existing Megatron NCCL group; only create the Gloo cmd_group.
        self._device_group = device_group
        # self._device_group = _get_or_create_process_group(self._ranks, backend="nccl")

        _pplx_debug_log(
            f"Creating TorchProcessGroupAdapter cmd_group with device_group ranks {torch.distributed.get_process_group_ranks(self._device_group)} and node_meta {node_meta} and size {self._size}"
        )
        # new_group() requires ALL world ranks to call it collectively, even for subgroups
        # they are not part of. Gather self._ranks from every world rank so that every rank
        # creates Gloo groups for all partitions, not just its own.
        world_size = torch.distributed.get_world_size()
        all_ranks: list[list[int] | None] = [None] * world_size
        torch.distributed.all_gather_object(all_ranks, self._ranks)
        seen: set[tuple[int, ...]] = set()
        for gathered_ranks in all_ranks:
            key = tuple(gathered_ranks)
            if key in seen:
                continue
            seen.add(key)
            pg = _get_or_create_process_group(list(key), backend="gloo")
            if gathered_ranks == self._ranks:
                self._cmd_group = pg

        assert hasattr(self, "_cmd_group"), (
            f"Failed to find own ranks {self._ranks} in all_gather_object result"
        )

        # Wire up the reducer against the existing device group.
        self._reducer = NcclAllReduce(group=self._device_group)

    @property
    def is_inter_node(self) -> bool:
        # Use actual node topology rather than TorchParallelGroup's size > 8 heuristic.
        return self._is_inter_node
    
    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @property
    @override
    def rank(self) -> int:
        """Local index within the parallel group."""
        return self._rank

    @property
    @override
    def node_rank(self) -> int:
        """The rank of the node within the parallel group."""
        return self._node_rank

    @property
    @override
    def global_rank(self) -> int:
        """Global index within the global group."""
        return self._global_rank

    @property
    @override
    def local_rank(self) -> int:
        """Local index within the current node."""
        return self._local_rank

    @property
    @override
    def size(self) -> int:
        """The size of the parallel group."""
        return self._size

    @property
    @override
    def is_inter_node(self) -> bool:
        """Returns true if the group spans multiple nodes."""
        return self._size > 8

    @override
    @profile_range("reducer")
    def reducer(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
    ) -> Reducer:
        return self._reducer.reducer(shape, dtype, op)

    @override
    @profile_range("all_reduce")
    def all_reduce(
        self,
        x: torch.Tensor,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
    ) -> torch.Tensor:
        return self._reducer.all_reduce(x, op)

    @override
    @profile_range("all_reduce_cpu_async")
    def all_reduce_cpu_async(
        self,
        x: torch.Tensor,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
    ) -> Work:
        assert x.device.type == "cpu"
        work = torch.distributed.all_reduce(
            x,
            op=op,
            group=self._cmd_group,
            async_op=True,
        )
        return cast(Work, work)

    @override
    @profile_range("all_gather")
    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Implementation adapted from vLLM
        assert -x.dim() <= dim < x.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {x.size()}"
        )
        assert x.device == self._device

        if dim < 0:
            dim += x.dim()

        input_size = x.size()
        output_tensor = torch.empty(
            (self._size,) + input_size,
            dtype=x.dtype,
            device=x.device,
        )
        torch.distributed.all_gather_into_tensor(
            output_tensor,
            x,
            group=self._device_group,
        )

        output_tensor = output_tensor.movedim(0, dim)
        return output_tensor.reshape(
            input_size[:dim] + (self._size * input_size[dim],) + input_size[dim + 1 :]
        )

    @override
    def all_gather_object(self, obj: T) -> list[T]:
        object_list = [None] * self._size
        torch.distributed.all_gather_object(object_list, obj, group=self._cmd_group)
        return cast(list[T], object_list)

    @override
    def broadcast_object(self, obj: T | None, root: int) -> T:
        assert 0 <= root < self._size, (
            f"Invalid rank {root} for group of size {self._size}"
        )
        objs = [obj]
        torch.distributed.broadcast_object_list(
            objs,
            src=self._ranks[root],
            group=self._cmd_group,
        )
        return cast(T, objs[0])

    @override
    def broadcast_cpu_tensor_async(self, tensor: torch.Tensor, root: int) -> Work:
        assert tensor.device.type == "cpu"
        work = torch.distributed.broadcast(
            tensor,
            src=self._ranks[root],
            group=self._cmd_group,
            async_op=True,
        )
        return cast(Work, work)

    @override
    def broadcast(self, tensor: torch.Tensor, root: int) -> torch.Tensor:
        torch.distributed.broadcast(
            tensor,
            src=self._ranks[root],
            group=self._device_group,
        )
        return tensor

    @override
    def all_to_all(self, tensor: torch.Tensor) -> torch.Tensor:
        m, *_ = tensor.shape
        if m != self._size:
            msg = f"Expected leading dim {m} to match group size {self._size}"
            raise ValueError(msg)

        output = torch.empty_like(tensor)
        torch.distributed.all_to_all_single(output, tensor, group=self._device_group)
        return output

    @override
    def barrier(self) -> None:
        torch.distributed.barrier(
            group=self._device_group,
            device_ids=[self._device.index],
        )

    @override
    @contextmanager
    def capture(self) -> Iterator[None]:
        with self._reducer.capture():
            yield

    def destroy(self) -> None:
        self._reducer.destroy()
        _pplx_debug_log("Destroyed reducer")

        # key = (tuple(self._ranks), "gloo")
        # if _process_group_cache.pop(key, None) is not None:
        #     torch.distributed.destroy_process_group(self._cmd_group)
        #     _pplx_debug_log("Destroyed cmd_group")
        
        for pg_key, pg in list(_process_group_cache.items()):
            if pg == self._cmd_group:
                _pplx_debug_log(f"Destroying cmd_group with key {pg_key} and ranks {torch.distributed.get_process_group_ranks(pg)}")
            if pg_key[1] == "gloo":
                _pplx_debug_log(f"Destroying external cmd_groups with key {pg_key} and ranks {torch.distributed.get_process_group_ranks(pg)}")
                torch.distributed.destroy_process_group(pg)
                _process_group_cache.pop(pg_key)
    
        # NOTE: _device_group is owned by Megatron, do not destroy it here.

    @override
    def slice_by_count(self, slice_count: int) -> ParallelGroup:
        assert self._size % slice_count == 0
        slice_size = self._size // slice_count
        return self.slice_by_lens([slice_size] * slice_count)

    @override
    def slice_by_lens(self, slice_lens: list[int]) -> ParallelGroup:
        # new_group() requires all ranks to call it with the same ranks,
        # even if the current rank is not in the subgroup.
        # And this needs to happen in the same order.
        # So we need to loop over all subgroups.

        assert sum(slice_lens) == self._size
        slice_ranks: list[list[int]] = []
        cumsum = 0
        for sl in slice_lens:
            slice_ranks.append(self._ranks[cumsum : cumsum + sl])
            cumsum += sl

        ret: TorchParallelGroup | None = None
        for ranks in slice_ranks:
            if self._global_rank in ranks:
                ret = TorchParallelGroup(
                    self._device,
                    self._node_rank,
                    self._local_rank,
                    self._global_rank,
                    ranks,
                )
            else:
                new_groups = self._create_new_groups(ranks)
                assert new_groups is None
        assert ret is not None

        # A barrier on the new group is required
        torch.distributed.barrier(
            group=ret._device_group,
            device_ids=[self._device.index],
        )
        return ret

    def _slice_ranks(self, slice_rank: int, slice_count: int) -> list[int]:
        """Slice the ranks assigned to this group."""

        assert 0 < slice_count <= len(self._ranks)
        assert 0 <= slice_rank < slice_count
        assert len(self._ranks) % slice_count == 0
        slice_size = len(self._ranks) // slice_count

        return self._ranks[slice_rank * slice_size : (slice_rank + 1) * slice_size]



def _build_node_rank_groups(ranks, gathered_rank_info):
    info_by_rank = {item["global_rank"]: item for item in gathered_rank_info}
    ordered_hostnames = []
    node_rank_groups = []
    for rank in ranks:
        hostname = info_by_rank[rank]["hostname"]
        if hostname in ordered_hostnames:
            continue
        ordered_hostnames.append(hostname)
        host_ranks = [
            candidate_rank
            for candidate_rank in ranks
            if info_by_rank[candidate_rank]["hostname"] == hostname
        ]
        local_ranks = [
            info_by_rank[candidate_rank]["local_rank"] for candidate_rank in host_ranks
        ]
        if len(local_ranks) != len(set(local_ranks)):
            raise RuntimeError(
                f"Detected duplicate LOCAL_RANK values for hostname {hostname} in pplx group"
            )
        host_ranks.sort(
            key=lambda candidate_rank: info_by_rank[candidate_rank]["local_rank"]
        )
        node_rank_groups.append(host_ranks)
    return node_rank_groups, ordered_hostnames


def _get_pplx_group_metadata(group: torch.distributed.ProcessGroup):
    ranks = torch.distributed.get_process_group_ranks(group)
    local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
    rank_info = {
        "global_rank": torch.distributed.get_rank(),
        "hostname": socket.gethostname(),
        "local_rank": local_rank,
    }
    gathered = [None] * len(ranks)
    torch.distributed.all_gather_object(gathered, rank_info, group=group)

    node_rank_groups, ordered_hostnames = _build_node_rank_groups(ranks, gathered)
    current_rank = torch.distributed.get_rank()
    current_hostname = rank_info["hostname"]
    node_ranks = next(
        node_ranks for node_ranks in node_rank_groups if current_rank in node_ranks
    )

    node_meta = {
        "node_rank": ordered_hostnames.index(current_hostname),
        "num_nodes": len(ordered_hostnames),
    }
    return ranks, node_ranks, node_rank_groups, node_meta


@internal_api
def make_pplx_process_group_adapters(
    global_group: torch.distributed.ProcessGroup,
    dp_group: torch.distributed.ProcessGroup,
):
    """Create pplx-garden adapters wrapping Megatron's flex global group and token-sharing group.

    global_group is the flex global expert communication group (EPxETP, pg_collection.tp_ep).
    dp_group is the ETP token-sharing subgroup where replicas live (pg_collection.expt_tp).
    Both are passed in from Megatron's pg_collection; we do not re-create their NCCL groups.
    Only Gloo cmd_groups and node-local subgroups are created here.
    """

    if not HAVE_PPLX_GARDEN:
        raise ImportError(
            "pplx-garden is not installed. Please install pplx_garden to use the "
            "pplx_garden flex MoE backend."
        )

    global_ranks, node_ranks, node_rank_groups, node_meta = _get_pplx_group_metadata(
        global_group
    )
    # dp_ranks = torch.distributed.get_process_group_ranks(dp_group)
    dp_ranks, node_ranks_dp, _, node_meta_dp = _get_pplx_group_metadata(dp_group)

    _pplx_debug_log(
        f"Constructed global_group_adapter with ranks {global_ranks} and node_meta {node_meta}"
    )
    _pplx_debug_log(
        f"Constructed dp_group_adapter with ranks {dp_ranks} and node_meta {node_meta_dp}"
    )

    global_group_adapter = _TorchProcessGroupAdapter(
        device_group=global_group,
        ranks=global_ranks,
        node_meta=node_meta,
    )
    dp_group_adapter = _TorchProcessGroupAdapter(
        device_group=dp_group,
        ranks=dp_ranks,
        node_meta=node_meta_dp,
    )

    # NOTE: For EDP, the inner groups are ETP, EPP and PP, so ETP should always have a part be intranode (and also EP if no ETP)
    # ep_is_inter_node = dp_group.size() >= gpus_per_node # EP pure internode
    # if not ep_is_inter_node:

    node_group_adapter = None
    # if node_meta["num_nodes"] > 1:
    #     node_device_group = None
    #     current_rank = torch.distributed.get_rank()
    #     for candidate_ranks in node_rank_groups:
    #         # All ranks must participate in new_group; only keep ours.
    #         candidate_device_group = _get_or_create_process_group(
    #             candidate_ranks, backend="nccl"
    #         )
    #         if current_rank in candidate_ranks:
    #             node_device_group = candidate_device_group

    #     assert node_device_group is not None, (
    #         "Failed to construct node-local pplx process group"
    #     )
    #     node_group_adapter = _TorchProcessGroupAdapter(
    #         device_group=node_device_group,
    #         ranks=node_ranks,
    #         node_meta=node_meta,
    #     )
    # else:
    #     # NOTE: because the slice goes in order the ranks should be in the same node
    #     gpus_per_node = torch.cuda.device_count()
    #     # TODO: How do we do this reliably?
    #     assert 0 < gpus_per_node <= min(8, global_group.size(), 8)
    #     node_group_adapter = global_group_adapter.slice_by_count(
    #         global_group_adapter._size // gpus_per_node
    #     )

    return global_group_adapter, dp_group_adapter, node_group_adapter


class PPLXDispatch(torch.autograd.Function):
    """Autograd wrapper for pplx-garden dispatch used as pure data movement."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        dispatch_weights,
        kernel,
        num_local_experts,
        max_recv_tokens,
        return_prob,
    ):
        _pplx_debug_log(
            "dispatch forward start "
            f"x_shape={tuple(x.shape)} x_dtype={x.dtype} x_stride={tuple(x.stride())} "
            f"indices_shape={tuple(token_indices.shape)} indices_dtype={token_indices.dtype} "
            f"weights_shape={tuple(dispatch_weights.shape)} weights_dtype={dispatch_weights.dtype} "
            f"max_recv_tokens={max_recv_tokens} num_local_experts={num_local_experts}"
        )
        out_num_tokens = torch.empty(
            (num_local_experts,), dtype=torch.int32, device=x.device
        )
        out_x = torch.empty(
            (max_recv_tokens, x.shape[1]), dtype=x.dtype, device=x.device
        )
        out_prob = None
        if return_prob:
            out_prob = torch.empty(
                (max_recv_tokens,), dtype=torch.float32, device=x.device
            )

        # NOTE: out_expert_x will be contiguous in order of tokens of expert 0, then tokens of expert 1, etc.
        _pplx_debug_log(f"dispatch x before dispatch:\n{x[:, 0].tolist()}")
        kernel.dispatch(
            out_expert_num_tokens=out_num_tokens,
            out_expert_x=out_x,
            out_expert_x_scale=None,
            dp_x=x,
            dp_x_scale=None,
            indices=token_indices,
            weights=dispatch_weights,
            out_expert_prob=out_prob,
            bound_m=None,
            do_send=True,
            do_recv=False,
        )
        torch.cuda.synchronize()
        kernel.dispatch(
            out_expert_num_tokens=out_num_tokens,
            out_expert_x=out_x,
            out_expert_x_scale=None,
            dp_x=x,
            dp_x_scale=None,
            indices=token_indices,
            weights=dispatch_weights,
            out_expert_prob=out_prob,
            bound_m=None,
            do_send=False,
            do_recv=True,
        )
        torch.cuda.synchronize()
        num_recv_tokens = int(out_num_tokens.sum().item())
        _pplx_debug_log(
            "dispatch forward end "
            f"num_recv_tokens={num_recv_tokens} tokens_per_expert={out_num_tokens.tolist()} out_x.shape={out_x.shape}"
        )
        _pplx_debug_log(f"dispatch forward end out_x tokens:\n{out_x[:num_recv_tokens, 0].tolist()}")

        # Debugging
        tokens_mine_to_recv = (token_indices.reshape(-1).to(torch.int64) < num_local_experts).sum().item()
        tokens_mine_to_recv_no_topk = (token_indices.to(torch.int64) < num_local_experts).any(dim=1).sum().item()
        # recv_tokens_mine = 0
        # if num_recv_tokens > 0 and x.shape[0] > 0:
        #     x_first = x[:, 0].cpu()
        #     out_x_first = out_x[:num_recv_tokens, 0].cpu()
        #     for i in x_first.tolist():
        #         for j in out_x_first.tolist():
        #             if i == j:
        #                 recv_tokens_mine += 1
        _pplx_debug_log(f"tokens_mine_to_recv={tokens_mine_to_recv}")
        _pplx_debug_log(f"tokens_mine_to_recv_no_topk={tokens_mine_to_recv_no_topk}")

        ctx.kernel = kernel
        ctx.num_input_tokens = x.shape[0]
        ctx.num_local_experts = num_local_experts
        ctx.save_for_backward(token_indices, torch.ones_like(dispatch_weights))
        # if out_prob is None:
        #     return out_x[:num_recv_tokens], out_num_tokens, None
        # return out_x[:num_recv_tokens], out_num_tokens, out_prob[:num_recv_tokens]
        if out_prob is None:
            return out_x, out_num_tokens, None
        return out_x, out_num_tokens, out_prob

    @staticmethod
    def backward(ctx, grad_output, grad_num_tokens, grad_prob):
        del grad_num_tokens
        del grad_prob
        token_indices, dispatch_weights = ctx.saved_tensors
        _pplx_debug_log(
            "dispatch backward start "
            f"grad_output_shape={tuple(grad_output.shape)} grad_output_dtype={grad_output.dtype}"
        )
        grad_x = torch.empty(
            (ctx.num_input_tokens, grad_output.shape[1]),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        ctx.kernel.combine(
            out_tokens=grad_x,
            indices=token_indices,
            weights=torch.ones_like(dispatch_weights), # NOTE: we need to pass ones here as the combine kernel always multiply the hidden states with the weights.
            expert_y=grad_output,
        )
        torch.cuda.synchronize()
    
        _pplx_debug_log("dispatch backward end")
        return grad_x, None, None, None, None, None, None


class PPLXCombine(torch.autograd.Function):
    """Autograd wrapper for pplx-garden combine used as pure data movement."""

    @staticmethod
    def forward(
        ctx, x, token_indices, combine_weights, kernel, num_tokens, num_local_experts
    ):
        _pplx_debug_log(
            "combine forward start "
            f"x_shape={tuple(x.shape)} x_dtype={x.dtype} x_stride={tuple(x.stride())} "
            f"indices_shape={tuple(token_indices.shape)} indices_dtype={token_indices.dtype} indices={token_indices.tolist()} "
            f"weights_shape={tuple(combine_weights.shape)} weights_dtype={combine_weights.dtype} "
            f"num_tokens={num_tokens} num_local_experts={num_local_experts} "
        )
        out_tokens = torch.empty(
            (num_tokens, x.shape[1]), dtype=x.dtype, device=x.device
        )
        rank = torch.distributed.get_rank()

        _pplx_debug_log(f"combine expert_y before combine:\n{x[:8, :8].tolist()}")
        kernel.combine(
            out_tokens=out_tokens,
            indices=token_indices,
            weights=torch.ones_like(combine_weights), # NOTE: we need to pass ones here as the combine kernel always multiply the hidden states with the weights.
            expert_y=x,
            accumulate=False, # NOTE: accumulate seems to be for when our out_tokens tensor already contains values and we want to add the expert results on top of it.
            do_send=True,
            do_recv=False,
        )
        torch.cuda.synchronize()
        recv_buf = kernel._recv_buffer_mapping.to_tensor(
            (kernel._recv_buffer_mapping.size,), torch.uint8
        ).cpu()
        # view as the output dtype to see actual values
        hidden_dim = kernel._hidden_dim
        out_dtype = kernel._out_dtype
        token_dim = ((hidden_dim * out_dtype.itemsize + 15) // 16) * 16
        num_slots = kernel._recv_buffer_mapping.size // token_dim
        data = recv_buf[:num_slots * token_dim].view(out_dtype).reshape(num_slots, -1)
        _pplx_debug_log(f"combine recv buffer first 8 slots:\n{data[:8, :8].tolist()}")

        kernel.combine(
            out_tokens=out_tokens,
            indices=token_indices,
            weights=torch.ones_like(
                combine_weights
            ),  # NOTE: we need to pass ones here as the combine kernel always multiply the hidden states with the weights.
            expert_y=x,
            accumulate=False,  # NOTE: accumulate seems to be for when our out_tokens tensor already contains values and we want to add the expert results on top of it.
            do_send=False,
            do_recv=True,
        )
        torch.cuda.synchronize() 

        _pplx_debug_log("combine forward end")
        ctx.kernel = kernel
        ctx.num_local_experts = num_local_experts
        ctx.num_expert_tokens = x.shape[0]
        ctx.save_for_backward(token_indices, combine_weights) # NOTE: in reality we don't need the combine weights saved for the backward
        return out_tokens

    @staticmethod
    def backward(ctx, grad_output):
        token_indices, combine_weights = ctx.saved_tensors
        _pplx_debug_log(
            "combine backward start "
            f"grad_output_shape={tuple(grad_output.shape)} grad_output_dtype={grad_output.dtype}"
        )
        out_num_tokens = torch.empty(
            (ctx.num_local_experts,), dtype=torch.int32, device=grad_output.device
        )
        out_x = torch.empty(
            (ctx.num_expert_tokens, grad_output.shape[1]),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        ctx.kernel.dispatch(
            out_expert_num_tokens=out_num_tokens,
            out_expert_x=out_x,
            out_expert_x_scale=None,
            dp_x=grad_output,
            dp_x_scale=None,
            indices=token_indices,
            weights=combine_weights, # NOTE: it will not be dispatched as we aren't allocating for the out_expert_probs
        )
        num_recv_tokens = int(out_num_tokens.sum().item())
        _pplx_debug_log(
            "combine backward end "
            f"num_recv_tokens={num_recv_tokens} tokens_per_expert={out_num_tokens.tolist()}"
        )
        return out_x, None, None, None, None, None


if HAVE_PPLX_GARDEN:

    @internal_api
    def pplx_dispatch(
        x,
        token_indices,
        dispatch_weights,
        kernel,
        num_local_experts,
        max_recv_tokens,
        return_prob=False,
    ):
        return PPLXDispatch.apply(
            x,
            token_indices,
            dispatch_weights,
            kernel,
            num_local_experts,
            max_recv_tokens,
            return_prob,
        )

    @internal_api
    def pplx_combine(
        x, token_indices, combine_weights, kernel, num_tokens, num_local_experts
    ):
        return PPLXCombine.apply(
            x, token_indices, combine_weights, kernel, num_tokens, num_local_experts
        )

else:

    def pplx_dispatch(*args, **kwargs):
        raise ImportError(
            "pplx-garden is not installed. Please install pplx_garden to use the "
            "pplx_garden flex MoE backend."
        )

    def pplx_combine(*args, **kwargs):
        raise ImportError(
            "pplx-garden is not installed. Please install pplx_garden to use the "
            "pplx_garden flex MoE backend."
        )


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused dispatch."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,  # wait in deepep::intra/inter_dispatch
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx,
        grad_output,
        grad_token_indices,
        grad_token_probs,
        grad_tokens_per_expert,
        grad_handle,
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(
        ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False
    ):
        """Forward pass of fused combine."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None


if HAVE_DEEP_EP:

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
        )

    def fused_combine(
        x, group, handle, async_finish=False, allocate_on_comm_stream=False
    ):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        return FusedCombine.apply(
            x, group, handle, async_finish, allocate_on_comm_stream
        )

    def set_deepep_num_sms(num_sms):
        """Sets the number of SMs to use for DeepEP"""
        Buffer.set_num_sms(num_sms)

else:
    fused_dispatch = None
    fused_combine = None
    set_deepep_num_sms = None


try:
    from deep_ep import HybridEPBuffer

    HAVE_HYBRIDEP = True
except ImportError:
    HAVE_HYBRIDEP = False

_hybrid_ep_buffer = None


def init_hybrid_ep_buffer(
    group: torch.distributed.ProcessGroup,
    hidden_dim: int,
    seq_len: int,
    num_local_experts: int,
    num_sms_dispatch_api: int,
    num_sms_combine_api: int,
    fp8_dispatch: bool,
) -> None:
    """
    Initialize the HybridEP buffer, including buffer allocation and metadata
    initialization.

    If a runtime dispatch/combine requires a larger buffer than the one
    initialized, the buffer will be reallocated at runtime,
    incuring extra run-time overhead.

    Args:
        group (torch.distributed.ProcessGroup):
            Process group for HybridEP all-to-all communication.
        hidden_dim (int):
            Hidden dimension of the input tensor.
        seq_len (int):
            Maximum sequence length of the input tensor.
        num_local_experts (int):
            Number of local experts.
        num_sms_dispatch_api (int):
            Number of SMs used by the dispatch API.
        num_sms_combine_api (int):
            Number of SMs used by the combine API.
        fp8_dispatch (bool):
            Whether to use FP8 communication during the dispatch phase.
    """
    assert not fp8_dispatch, "HybridEP dispatcher does not support fp8 dispatch now"
    global _hybrid_ep_buffer
    _hybrid_ep_buffer = HybridEPBuffer(
        group=group,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=seq_len,
        num_local_experts=num_local_experts,
        use_fp8=fp8_dispatch,
        num_sms_dispatch_api=num_sms_dispatch_api,
        num_sms_combine_api=num_sms_combine_api,
    )


def reset_hybrid_ep_buffer():
    """
    Reset the HybridEP buffer
    """
    global _hybrid_ep_buffer
    _hybrid_ep_buffer = None


class HybridEPDispatch(torch.autograd.Function):
    """
    Fused dispatch operation for permute + dispatch a2a + permute using the HybridEP backend
    """

    @staticmethod
    def forward(
        ctx,
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_permuted_tokens=None,
        pad_multiple=None,
    ):
        """
        Forward pass of fused dispatch of the HybridEP backend
        """
        if _hybrid_ep_buffer is None:
            seq_len, hidden_dim = x.shape[-2:]
            fp8_dispatch = False  # Currently, we do not support fp8 dispatch
            init_hybrid_ep_buffer(
                group,
                hidden_dim,
                seq_len,
                num_local_experts,
                num_sms_dispatch_api,
                num_sms_combine_api,
                fp8_dispatch,
            )
        # If we provide the num_permuted_tokens, we do not need to use sync to
        # wait for the data in pinned memory ready
        non_blocking = num_permuted_tokens is not None
        # Process the dispatch
        (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        ) = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=x,
            routing_map=routing_map,
            probs=probs,
            scaling_factor=None,
            num_of_experts_per_rank=num_local_experts,
            pad_multiple=pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=non_blocking,
        )

        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        return (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        )

    @staticmethod
    def backward(
        ctx,
        grad_x,
        grad_probs,
        grad_scaling_factor,
        grad_tokens_per_expert,
        grad_handle,
    ):
        """
        Backward pass of fused dispatch of the HybridEP backend
        """
        handle = ctx.handle
        combined_hidden, combined_probs = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=grad_x,
            probs=grad_probs,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
        )
        return (
            combined_hidden,
            None,
            combined_probs,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@internal_api
class HybridEPCombine(torch.autograd.Function):
    """
    Fused combine operation for permute + combine a2a + permute using the HybridEP backend
    """

    @staticmethod
    def forward(ctx, x, handle, num_permuted_tokens=None, pad_multiple=None):
        """
        Forward pass of fused combine of the HybridEP backend
        """
        combined_hidden, _ = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=x, handle=handle, pad_multiple=pad_multiple
        )
        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.num_permuted_tokens = num_permuted_tokens
        return combined_hidden

    @staticmethod
    def backward(ctx, grad_x):
        """
        Backward pass of fused combine of the HybridEP backend
        """
        handle = ctx.handle
        dispatched_hidden, _, _, _, _ = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=grad_x,
            scaling_factor=None,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
            num_permuted_tokens=ctx.num_permuted_tokens,
        )
        return dispatched_hidden, None, None, None, None


if HAVE_HYBRIDEP:

    @internal_api
    def hybrid_ep_dispatch(
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_permuted_tokens=None,
        pad_multiple=None,
    ):
        """
        Perform fused dispatch for "permute + dispatch a2a + permute" using the
        HybridEP backend.

        Args:
            x (torch.Tensor):
                Input hidden states to dispatch.
            routing_map (torch.Tensor):
                Map indicating which expert each token is routed to.
            probs (torch.Tensor):
                Routing probabilities for each token-expert pair.
            group (torch.distributed.ProcessGroup):
                Process group used for communication.
            num_local_experts (int):
                Number of local experts.
            num_sms_dispatch_api (int):
                Number of SMs used by the dispatch API.
            num_sms_combine_api (int):
                Number of SMs used by the combine API.
            num_permuted_tokens (int):
                Number of tokens after permute. HybridEP uses this to allocate buffers.
                If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                Alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
        """
        return HybridEPDispatch.apply(
            x,
            routing_map,
            probs,
            group,
            num_local_experts,
            num_sms_dispatch_api,
            num_sms_combine_api,
            num_permuted_tokens,
            pad_multiple,
        )

    @internal_api
    def hybrid_ep_combine(x, handle, num_permuted_tokens, pad_multiple):
        """
        Perform fused combine operation for unpermute + combine a2a + unpermute
        using the HybridEP backend

        args:
            x (torch.Tensor):
                Input hidden states to combine
            handle (EventHandle):
                Communication handle from dispatch operation
            num_permuted_tokens (int): The number of tokens before unpermute. HybridEP uses this
                to allocate buffers. If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                The alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
        """
        return HybridEPCombine.apply(x, handle, num_permuted_tokens, pad_multiple)

else:
    hybrid_ep_dispatch = None
    hybrid_ep_combine = None

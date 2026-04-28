# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import token

import copy
import dataclasses
import os

import pytest
import torch

from megatron.core import config, parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import get_capacity
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def get_runtime_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


PPLX_SINGLE_NODE_FWD_BWD_CASES = [(1, 4), (1, 2)]
PPLX_MULTI_NODE_FWD_BWD_CASES = [(1, 8), (4, 2)]
PPLX_SINGLE_NODE_INFERENCE_CASES = [(1, 4)]
PPLX_MULTI_NODE_INFERENCE_CASES = [(1, 8)]


def token_permutation(token_dispatcher, hidden_states, probs, indices):
    hidden_states, probs = token_dispatcher.dispatch_preprocess(
        hidden_states, indices, probs
    )
    hidden_states, probs = token_dispatcher.token_dispatch(hidden_states, probs)
    hidden_states, tokens_per_expert, permuted_probs = (
        token_dispatcher.dispatch_postprocess(hidden_states, probs)
    )
    return hidden_states, tokens_per_expert, permuted_probs


def token_unpermutation(token_dispatcher, hidden_states):
    hidden_states = token_dispatcher.combine_preprocess(hidden_states)
    hidden_states = token_dispatcher.token_combine(hidden_states)
    hidden_states = token_dispatcher.combine_postprocess(hidden_states)
    return hidden_states, None


class MoEModelTestContainer:
    def __init__(
        self,
        tp_size,
        ep_size,
        pp_size,
        cp_size=1,
        moe_tp_size=None,
        data_parallel_random_init=False,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="alltoall",
        moe_expert_capacity_factor=None,
        moe_pad_expert_input_to_capacity=False,
        moe_aux_loss_coeff=0.1,
        test_dtype=torch.float32,
        **kwargs,
    ):
        self.num_local_experts = num_moe_experts // ep_size
        self.test_dtype = test_dtype
        if moe_tp_size is None:
            moe_tp_size = tp_size
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            expert_model_parallel_size=ep_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=data_parallel_random_init)
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        self.config = TransformerConfig(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            pipeline_model_parallel_size=pp_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
            moe_router_topk=moe_router_topk,
            num_moe_experts=num_moe_experts,
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_expert_capacity_factor=moe_expert_capacity_factor,
            moe_pad_expert_input_to_capacity=moe_pad_expert_input_to_capacity,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            num_layers=1,
            moe_router_dtype="fp32",
            moe_grouped_gemm=kwargs.get("moe_grouped_gemm", False),
            hidden_size=kwargs.get("hidden_size", 16),
            num_attention_heads=kwargs.get("num_attention_heads", 8),
            use_cpu_initialization=kwargs.get("use_cpu_initialization", True),
            sequence_parallel=tp_size > 1,
            add_bias_linear=kwargs.get("add_bias_linear", False),
            moe_permute_fusion=kwargs.get("moe_permute_fusion", False),
            moe_flex_dispatcher_backend=kwargs.get("moe_flex_dispatcher_backend", None),
        )

        # init moe layer
        self.moe_layer = self.new_moe_layer()

    def new_moe_layer(self, **kargs):
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=self.config.num_moe_experts,
            moe_grouped_gemm=self.config.moe_grouped_gemm,
        )
        new_config = dataclasses.replace(self.config, **kargs)
        moe_layer = (
            MoELayer(new_config, transformer_layer_spec.submodules.mlp.submodules)
            .cuda()
            .to(dtype=self.test_dtype)
        )
        moe_layer.set_layer_number(0)
        return moe_layer

    def __del__(self):
        try:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def dispatcher_dropless_test(self):
        moe_layer = self.moe_layer
        bs = 8
        seql = 8

        def _dump_pplx_kernel_debug_state(label: str, max_token_offsets: int = 64, max_recv_entries: int = 128) -> None:
            comm_manager = getattr(moe_layer.token_dispatcher, "_comm_manager", None)
            kernel = getattr(comm_manager, "_hidden_kernel", None)
            if kernel is None or not hasattr(kernel, "debug_state"):
                print(
                    f"rank {torch.distributed.get_rank()} {label} kernel debug_state unavailable"
                )
                return

            state = kernel.debug_state(
                max_token_offsets=max_token_offsets,
                max_recv_entries=max_recv_entries,
            )
            print(
                f"rank {torch.distributed.get_rank()} {label} kernel debug_state "
                f"num_recv_tokens={state['num_recv_tokens']} "
                f"expert_offsets={state['expert_offsets']} "
                f"token_offset={state['token_offset']} "
                f"padded_index={state['padded_index']} "
                f"combine_send_offset={state['combine_send_offset']} "
                f"source_dispatch_offset={state['source_dispatch_offset']} "
                f"source_rank={state['source_rank']}"
                f"tokens_per_expert={state['tokens_per_expert']} "
            )

        # TODO: Find why setting manual seed can cause the test to fail
        # Manual seed to differentiate input data for each rank
        # rank = torch.distributed.get_rank()
        # torch.manual_seed(1000 + rank)
        hidden_states = torch.randn(
            (bs, seql, moe_layer.config.hidden_size), dtype=self.test_dtype
        )
        hidden_states = hidden_states.cuda()
        # Permute and then unpermute data are supposed to restore original data
        ans = hidden_states.clone()
        hidden_states.requires_grad = True
        probs, indices = apply_module(moe_layer.router)(hidden_states)
        probs = torch.ones_like(probs) / moe_layer.router.topk

        # # print("probs:", probs)
        # token_indices = torch.arange(
        #     moe_layer.config.num_moe_experts, device=hidden_states.device
        # ).expand(bs * seql, -1)

        # print("shape of indices:", indices.shape)
        # print("shape of token_indices:", token_indices.shape)
        # print(f"indices: {indices}")

        # print("indices:", token_indices[indices].reshape(bs * seql, moe_layer.router.topk))

        (permuted_local_hidden_states, tokens_per_expert, permuted_probs) = (
            token_permutation(moe_layer.token_dispatcher, hidden_states, probs, indices)
        )
        valid_recv_tokens = int(tokens_per_expert.sum().item())
        assert 0 <= valid_recv_tokens <= permuted_local_hidden_states.shape[0], (
            f"Invalid valid_recv_tokens={valid_recv_tokens} for "
            f"permuted_local_hidden_states.shape={permuted_local_hidden_states.shape}"
        )

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        all_tp_permuted_local_hidden_states = [ torch.empty_like(permuted_local_hidden_states) for _ in range(tp_size) ]
        torch.distributed.all_gather(all_tp_permuted_local_hidden_states, permuted_local_hidden_states, group=tp_group)

        for i in range(1, tp_size):
            (
                torch.testing.assert_close(
                    all_tp_permuted_local_hidden_states[i], permuted_local_hidden_states,
                ),
                f"Permuted local hidden states are not the same across TP ranks, rank 0 and rank {i}",
            )

        print(f"rank {torch.distributed.get_rank()} probs.shape {probs.shape}, permuted_probs.shape {permuted_probs.shape}, permuted_local_hidden_states.shape {permuted_local_hidden_states.shape}")
        print(f"rank {torch.distributed.get_rank()} probs max: {probs.max()}, min: {probs.min()}, mean: {probs.mean()}, mode: {probs.mode()} \n rank {torch.distributed.get_rank()} permuted_probs max: {permuted_probs.max()}, min: {permuted_probs.min()}, mean: {permuted_probs.mean()}, mode: {permuted_probs.mode()}")
        print(
            f"rank {torch.distributed.get_rank()} valid_recv_tokens={valid_recv_tokens}, "
            f"tokens_per_expert={tokens_per_expert.tolist()}, "
            f"tail_probs_nonzero={int(torch.count_nonzero(permuted_probs[valid_recv_tokens:]).item())}"
        )
        _dump_pplx_kernel_debug_state("baseline")

        print(f"rank {torch.distributed.get_rank()} permuted_local_hidden_states max: {permuted_local_hidden_states.max()}, min: {permuted_local_hidden_states.min()}, mean: {permuted_local_hidden_states.mean()}, mode: {permuted_local_hidden_states.mode()}\n rank {torch.distributed.get_rank()} ans max: {ans.max()}, min: {ans.min()}, mean: {ans.mean()}, mode: {ans.mode()}")

        weighted_permuted_local_hidden_states = (
            permuted_local_hidden_states * permuted_probs.unsqueeze(-1).to(dtype=permuted_local_hidden_states.dtype)
        ) # NOTE: This is the grouped gemm op, as in megatron they multiply the probs during the grouped gemm (like in after the up projection).

        weighted_permuted_local_hidden_states = weighted_permuted_local_hidden_states.to(
            dtype=self.test_dtype
        )
        print(
            f"rank {torch.distributed.get_rank()} permuted_local_hidden_states dtype after mul: "
            f"{weighted_permuted_local_hidden_states.dtype}, ans dtype: {ans.dtype}"
        )
        # print(
        #     f"rank {torch.distributed.get_rank()} after mul probs, permuted_local_hidden_states max: {permuted_local_hidden_states.max()}, min: {permuted_local_hidden_states.min()}, mean: {permuted_local_hidden_states.mean()}, mode: {permuted_local_hidden_states.mode()}\n rank {torch.distributed.get_rank()} ans max: {ans.max()}, min: {ans.min()}, mean: {ans.mean()}, mode: {ans.mode()}"
        # )

        restored_hidden_states, restored_bias = token_unpermutation(
            moe_layer.token_dispatcher, weighted_permuted_local_hidden_states
        )

        # reduce across TP rank equals to multiply data by a scale of ETP because the token will be duplicated ETP times
        scale = moe_layer.config.expert_tensor_parallel_size
        restored_hidden_states = restored_hidden_states / scale

        if valid_recv_tokens < weighted_permuted_local_hidden_states.shape[0]:
            (
                poisoned_permuted_local_hidden_states,
                poisoned_tokens_per_expert,
                poisoned_permuted_probs,
            ) = token_permutation(moe_layer.token_dispatcher, hidden_states, probs, indices)
            poisoned_valid_recv_tokens = int(poisoned_tokens_per_expert.sum().item())
            assert poisoned_valid_recv_tokens == valid_recv_tokens, (
                f"Fresh dispatch valid_recv_tokens mismatch: "
                f"{poisoned_valid_recv_tokens} != {valid_recv_tokens}"
            )
            _dump_pplx_kernel_debug_state("poisoned_fresh_dispatch")

            poisoned_weighted_hidden_states = (
                poisoned_permuted_local_hidden_states
                * poisoned_permuted_probs.unsqueeze(-1).to(
                    dtype=poisoned_permuted_local_hidden_states.dtype
                )
            ).to(dtype=self.test_dtype)
            poison_value = torch.tensor(
                1024.0, dtype=self.test_dtype, device=hidden_states.device
            )
            poisoned_weighted_hidden_states[valid_recv_tokens:] = poison_value
            restored_hidden_states_poisoned, _ = token_unpermutation(
                moe_layer.token_dispatcher, poisoned_weighted_hidden_states
            )
            restored_hidden_states_poisoned = restored_hidden_states_poisoned / scale

            poison_diff = (restored_hidden_states_poisoned - restored_hidden_states).abs()
            max_poison_diff = float(poison_diff.max().item())
            changed_positions = int(torch.count_nonzero(poison_diff).item())
            print(
                f"rank {torch.distributed.get_rank()} tail poison check "
                f"max_diff={max_poison_diff}, changed_positions={changed_positions}, "
                f"poisoned_tail_rows={poisoned_weighted_hidden_states.shape[0] - valid_recv_tokens}, "
                f"fresh_tail_probs_nonzero={int(torch.count_nonzero(poisoned_permuted_probs[valid_recv_tokens:]).item())}"
            )
            (
                torch.testing.assert_close(
                    restored_hidden_states_poisoned,
                    restored_hidden_states,
                    atol=0.0,
                    rtol=0.0,
                ),
                "Combine output changed after poisoning only the padded dispatched tail",
            )

        # print("restored_hidden_states:", restored_hidden_states[0, 0, 0:4])
        # print("ans:", ans[0, 0, 0:4])
        # print(f"ans.shape: {ans.shape}, restored_hidden_states.shape: {restored_hidden_states.shape}")

        print(f"rank {torch.distributed.get_rank()} restored_hidden_states max: {restored_hidden_states.max()}, min: {restored_hidden_states.min()}, mean: {restored_hidden_states.mean()}, mode: {restored_hidden_states.mode()}\n ans max: {ans.max()}, min: {ans.min()}, mean: {ans.mean()}, mode: {ans.mode()}")

        print(f"rank {torch.distributed.get_rank()} restored_hidden_states max(dim=-1): {len(restored_hidden_states.max(dim=-1)[0].unique())}\n ans max(dim=-1): {len(ans.max(dim=-1)[0].unique())}")

        print(f"rank {torch.distributed.get_rank()} restored_hidden_states: {restored_hidden_states[:, :, 0].tolist()} \n ans: {ans[:, :, 0].tolist()}")

        (
            torch.testing.assert_close(restored_hidden_states, ans),
            "Restored hidden states do not match original hidden states",
        )

        # check if the grad of the hidden states is same as the hidden states
        # TODO: Reactivate this
        # torch.autograd.backward(restored_hidden_states, hidden_states)
        # (
        #     torch.testing.assert_close(hidden_states.grad, ans),
        #     "Restored hidden states do not match original hidden states",
        # )

    @pytest.mark.internal
    def dispatcher_capacity_test(self):
        moe_layer = self.moe_layer
        num_tokens = 16
        hidden_states = torch.randn(
            (num_tokens, moe_layer.config.hidden_size), dtype=self.test_dtype
        )
        hidden_states = hidden_states.cuda()
        hidden_states.requires_grad = True
        probs, indices = apply_module(moe_layer.router)(hidden_states)

        # Create the answer.
        prob_mask = probs != 0
        probs = torch.ones_like(probs) * prob_mask / moe_layer.router.topk
        local_probss = probs
        restored_hidden_states_answer = hidden_states * local_probss.sum(
            dim=1
        ).unsqueeze(1)
        restored_hidden_states_answer = restored_hidden_states_answer.to(
            dtype=self.test_dtype
        )

        (permuted_local_hidden_states, tokens_per_expert, permuted_probs) = (
            token_permutation(moe_layer.token_dispatcher, hidden_states, probs, indices)
        )

        # Check tokens per expert not exceed the capacity.
        capacity = get_capacity(
            num_tokens * self.config.moe_router_topk,
            self.config.num_moe_experts,
            self.config.moe_expert_capacity_factor,
        )
        assert torch.all(
            tokens_per_expert
            <= capacity
            * self.config.expert_model_parallel_size
            * self.config.tensor_model_parallel_size
        ), "Tokens per expert exceed the capacity"

        permuted_local_hidden_states = (
            permuted_local_hidden_states * permuted_probs.unsqueeze(-1)
        )

        permuted_local_hidden_states /= moe_layer.config.tensor_model_parallel_size
        permuted_local_hidden_states = permuted_local_hidden_states.to(
            dtype=self.test_dtype
        )

        restored_hidden_states, restored_bias = token_unpermutation(
            moe_layer.token_dispatcher, permuted_local_hidden_states
        )
        (
            torch.testing.assert_close(
                restored_hidden_states, restored_hidden_states_answer
            ),
            "Restored hidden states does not match",
        )

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, hidden_states)
        (
            torch.testing.assert_close(
                hidden_states.grad, restored_hidden_states_answer
            ),
            "Gradient of hidden states should be same as hidden states",
        )

    @pytest.mark.internal
    def dispatcher_drop_and_pad_test(self):
        """Test if the tokens are dropped and padded correctly.

        Since the probs of padded tokens are 0, the combined results for
        dispatching with or without padding should be the same.
        """
        moe_layer = self.new_moe_layer(moe_pad_expert_input_to_capacity=False)

        num_tokens = 16
        hidden_states = torch.randn(
            (num_tokens, moe_layer.config.hidden_size), dtype=self.test_dtype
        ).cuda()
        hidden_states.requires_grad = True

        probs_1, indices_1 = apply_module(moe_layer.router)(hidden_states)
        (permuted_input_1, tokens_per_expert, permuted_probs_1) = token_permutation(
            moe_layer.token_dispatcher, hidden_states, probs_1, indices_1
        )
        permuted_input_1 = permuted_input_1 * permuted_probs_1.unsqueeze(-1)
        permuted_input_1 = permuted_input_1.to(dtype=self.test_dtype)
        forward_answer, restored_bias = token_unpermutation(
            moe_layer.token_dispatcher, permuted_input_1
        )
        torch.autograd.backward(forward_answer, forward_answer)
        backward_answer = hidden_states.grad.clone()
        hidden_states.grad = None
        torch.cuda.synchronize()
        # End

        moe_layer_2 = self.new_moe_layer(moe_pad_expert_input_to_capacity=True)
        moe_layer_2.load_state_dict(moe_layer.state_dict())

        probs_2, indices_2 = apply_module(moe_layer_2.router)(hidden_states)
        (permuted_input_2, tokens_per_expert, permuted_probs_2) = token_permutation(
            moe_layer_2.token_dispatcher, hidden_states, probs_2, indices_2
        )
        permuted_input_2 = permuted_input_2 * permuted_probs_2.unsqueeze(-1)
        permuted_input_2 = permuted_input_2.to(dtype=self.test_dtype)
        restored_hidden_states, restored_bias = token_unpermutation(
            moe_layer_2.token_dispatcher, permuted_input_2
        )

        # # Check tokens per expert equals to the capacity.
        capacity = get_capacity(
            num_tokens * self.config.moe_router_topk,
            self.config.num_moe_experts,
            self.config.moe_expert_capacity_factor,
        )
        assert torch.all(
            tokens_per_expert
            == capacity
            * self.config.expert_model_parallel_size
            * self.config.tensor_model_parallel_size
        ), "Tokens per expert should be the same as the capacity"
        (
            torch.testing.assert_close(restored_hidden_states, forward_answer),
            "Restored hidden states does not match",
        )

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, restored_hidden_states)
        (
            torch.testing.assert_close(hidden_states.grad, backward_answer),
            "Gradient of hidden states should be same as hidden states",
        )

    @pytest.mark.internal
    def dispatcher_router_padding_for_fp8_test(self):
        """Test if the routing map is padded correctly for FP8 training.

        The test runs the forward flow twice:
        1. First with moe_router_padding_for_quantization=False
        2. Then with moe_router_padding_for_quantization=True

        We verify that:
        1. The results are the same in both cases
        2. The number of tokens received by each expert is padded to a multiple of 16
        """
        # First run with moe_router_padding_for_quantization = False
        moe_layer = self.new_moe_layer(moe_router_padding_for_quantization=False)

        num_tokens = 32
        hidden_states = torch.randn(
            (num_tokens, moe_layer.config.hidden_size), dtype=self.test_dtype
        ).cuda()
        hidden_states.requires_grad = True

        probs_1, indices_1 = apply_module(moe_layer.router)(hidden_states)
        (permuted_input_1, tokens_per_expert_1, permuted_probs_1) = token_permutation(
            moe_layer.token_dispatcher, hidden_states, probs_1, indices_1
        )
        permuted_input_1 = permuted_input_1 * permuted_probs_1.unsqueeze(-1)
        permuted_input_1 = permuted_input_1.to(dtype=self.test_dtype)
        restored_hidden_states_1, _ = token_unpermutation(
            moe_layer.token_dispatcher, permuted_input_1
        )
        torch.autograd.backward(restored_hidden_states_1, restored_hidden_states_1)
        grad_1 = hidden_states.grad.clone()
        hidden_states.grad = None

        # Run with moe_router_padding_for_quantization = True
        moe_layer_2 = self.new_moe_layer(
            moe_router_padding_for_quantization=True, fp8="hybrid"
        )
        moe_layer_2.load_state_dict(moe_layer.state_dict())

        probs_2, indices_2 = apply_module(moe_layer_2.router)(hidden_states)
        (permuted_input_2, tokens_per_expert_2, permuted_probs_2) = token_permutation(
            moe_layer_2.token_dispatcher, hidden_states, probs_2, indices_2
        )
        assert sum(tokens_per_expert_2) == permuted_input_2.shape[0], (
            f"number of tokens is not the same, {sum(tokens_per_expert_2)} != {permuted_input_2.shape[0]}"
        )
        # when there is only one expert, the tokens is not enough for router padding
        if moe_layer_2.num_local_experts > 1:
            assert torch.all(tokens_per_expert_2 % 16 == 0), (
                "number of tokens for expert is not a multiple of 16"
            )

        permuted_input_2 = permuted_input_2 * permuted_probs_2.unsqueeze(-1)
        permuted_input_2 = permuted_input_2.to(dtype=self.test_dtype)
        restored_hidden_states_2, _ = token_unpermutation(
            moe_layer_2.token_dispatcher, permuted_input_2
        )

        # Check that the results are the same
        (
            torch.testing.assert_close(
                restored_hidden_states_1, restored_hidden_states_2
            ),
            "Restored hidden states do not match between padded and non-padded versions",
        )

        # Check gradients
        torch.autograd.backward(restored_hidden_states_2, restored_hidden_states_2)
        (
            torch.testing.assert_close(grad_1, hidden_states.grad),
            "Gradients do not match between padded and non-padded versions",
        )

    def set_params(self):
        # TODO: Set consistent parameters for various parallelisms.
        raise NotImplementedError

    def destroy(self):
        Utils.destroy_model_parallel()


permute_fusion_params = [False]
if is_te_min_version("2.1.0"):
    permute_fusion_params.append(True)


class TestAllgatherDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("tp_size,ep_size", [(8, 1), (1, 8), (2, 4), (1, 1)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    def test_forward_backward(self, tp_size, ep_size, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="allgather",
            moe_permute_fusion=permute_fusion,
        )

        container.dispatcher_dropless_test()

    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize(
        "tp_size,ep_size,moe_tp_size", [(1, 1, 8), (1, 2, 4), (1, 4, 2), (2, 2, 4)]
    )
    def test_moe_tp_forward_backward(
        self, tp_size, ep_size, moe_tp_size, permute_fusion
    ):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            moe_tp_size=moe_tp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="allgather",
            sequence_parallel=True,
            moe_permute_fusion=permute_fusion,
            use_cpu_initialization=False,
        )

        container.dispatcher_dropless_test()


def is_deep_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP

    return HAVE_DEEP_EP


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP

    return HAVE_HYBRIDEP


def is_pplx_garden_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_PPLX_GARDEN

    return HAVE_PPLX_GARDEN


def test_pplx_garden_node_group_builder_orders_local_ranks():
    from megatron.core.transformer.moe.fused_a2a import _build_node_rank_groups

    node_rank_groups, ordered_hostnames = _build_node_rank_groups(
        [0, 1, 2, 3, 4, 5],
        [
            {"global_rank": 0, "hostname": "node-a", "local_rank": 2},
            {"global_rank": 1, "hostname": "node-a", "local_rank": 0},
            {"global_rank": 2, "hostname": "node-b", "local_rank": 1},
            {"global_rank": 3, "hostname": "node-b", "local_rank": 0},
            {"global_rank": 4, "hostname": "node-a", "local_rank": 1},
            {"global_rank": 5, "hostname": "node-b", "local_rank": 2},
        ],
    )

    assert ordered_hostnames == ["node-a", "node-b"]
    assert node_rank_groups == [[1, 4, 0], [3, 2, 5]]


def test_pplx_garden_node_group_builder_rejects_duplicate_local_ranks():
    from megatron.core.transformer.moe.fused_a2a import _build_node_rank_groups

    with pytest.raises(RuntimeError, match="duplicate LOCAL_RANK"):
        _build_node_rank_groups(
            [0, 1],
            [
                {"global_rank": 0, "hostname": "node-a", "local_rank": 0},
                {"global_rank": 1, "hostname": "node-a", "local_rank": 0},
            ],
        )


@pytest.mark.skipif(
    not is_deep_ep_available() and not is_hybrid_ep_available(),
    reason="Deep EP and Hybrid EP are not available",
)
class TestFlexDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize("moe_flex_dispatcher_backend", ["deepep", "hybridep"])
    def test_forward_backward(
        self, tp_size, ep_size, permute_fusion, moe_flex_dispatcher_backend
    ):
        if moe_flex_dispatcher_backend == "deepep" and not is_deep_ep_available():
            pytest.skip("Deep EP is not available")
        if moe_flex_dispatcher_backend == "hybridep" and not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")
        if permute_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_permute_fusion=permute_fusion,
            hidden_size=1024,
            moe_flex_dispatcher_backend=moe_flex_dispatcher_backend,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_dropless_test()
        # reset experimental flag to False
        config.ENABLE_EXPERIMENTAL = False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize("moe_flex_dispatcher_backend", ["deepep", "hybridep"])
    def test_capacity_forward_backward(
        self, tp_size, ep_size, permute_fusion, moe_flex_dispatcher_backend
    ):
        if moe_flex_dispatcher_backend == "deepep" and not is_deep_ep_available():
            pytest.skip("Deep EP is not available")
        if moe_flex_dispatcher_backend == "hybridep" and not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")
        if permute_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_token_drop_policy="probs",
            moe_expert_capacity_factor=0.5,
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=permute_fusion,
            hidden_size=1024,
            moe_flex_dispatcher_backend=moe_flex_dispatcher_backend,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_capacity_test()
        config.ENABLE_EXPERIMENTAL = False

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0"), reason="TE 1.7.0 is required for MoE with FP8."
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", [True])
    @pytest.mark.parametrize("moe_flex_dispatcher_backend", ["deepep", "hybridep"])
    def test_router_padding_for_fp8_forward_backward(
        self, tp_size, ep_size, permute_fusion, moe_flex_dispatcher_backend
    ):
        if moe_flex_dispatcher_backend == "deepep" and not is_deep_ep_available():
            pytest.skip("Deep EP is not available")
        if moe_flex_dispatcher_backend == "hybridep" and not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")
        if permute_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=32,
            moe_router_topk=4,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=permute_fusion,
            hidden_size=1024,
            moe_flex_dispatcher_backend=moe_flex_dispatcher_backend,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_router_padding_for_fp8_test()
        config.ENABLE_EXPERIMENTAL = False


@pytest.mark.skipif(
    not is_pplx_garden_available(), reason="pplx_garden is not available"
)
class TestPplxGardenFlexDispatcher:
    def setup_method(self, method):
        self.container = None

    def teardown_method(self, method):
        if self.container is not None:
            moe_layer = self.container.moe_layer
            if hasattr(moe_layer, 'token_dispatcher') and hasattr(moe_layer.token_dispatcher, '_comm_manager'):
                moe_layer.token_dispatcher._comm_manager.destroy()
            self.container = None
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    def test_single_node_metadata_layout_tp1_ep4(self):
        if get_runtime_world_size() != 4:
            pytest.skip("single-node pplx test requires WORLD_SIZE=4")

        self.container = MoEModelTestContainer(
            tp_size=1,
            ep_size=4,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="pplx_garden",
            moe_permute_fusion=False,
            test_dtype=torch.bfloat16,
        )
        moe_layer = self.container.moe_layer
        hidden_states = torch.randn(
            (8, 4, moe_layer.config.hidden_size), dtype=self.container.test_dtype
        )
        hidden_states = hidden_states.cuda()

        probs, indices = apply_module(moe_layer.router)(hidden_states)
        probs = torch.ones_like(probs) / moe_layer.router.topk

        token_dispatcher = moe_layer.token_dispatcher
        token_dispatcher.dispatch_preprocess(hidden_states, indices, probs)
        token_indices = token_dispatcher._comm_manager.token_indices

        assert token_indices is not None
        assert token_indices.shape[-1] == moe_layer.router.topk

        # routing_map = indices.reshape(-1, container.config.num_moe_experts)
        # flat_token_indices = torch.arange(
        #     container.config.num_moe_experts,
        #     device=routing_map.device,
        #     dtype=torch.int64,
        # ).expand(routing_map.shape[0], -1)
        # expected_token_indices = flat_token_indices[routing_map].reshape(
        #     routing_map.shape[0], moe_layer.router.topk
        # )
        _, expected_token_indices = torch.topk(probs, k=moe_layer.router.topk, dim=-1)

        torch.testing.assert_close(token_indices.long(), expected_token_indices)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", PPLX_SINGLE_NODE_FWD_BWD_CASES)
    def test_single_node_forward_backward(self, tp_size, ep_size):
        if get_runtime_world_size() > 4:
            pytest.skip("single-node pplx test requires WORLD_SIZE<=4")

        if tp_size * ep_size > get_runtime_world_size():
            pytest.skip(
                f"tp_size * ep_size should be less than or equal to WORLD_SIZE, but got tp_size={tp_size}, ep_size={ep_size}, WORLD_SIZE={get_runtime_world_size()}"
            )
    
        self.container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="pplx_garden",
            moe_permute_fusion=False,
            test_dtype=torch.bfloat16,
            moe_router_force_load_balancing=True,
            hidden_size=16,
        )

        # TODO: TP>1 fails for now
        self.container.dispatcher_dropless_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", PPLX_SINGLE_NODE_INFERENCE_CASES)
    def test_single_node_inference_no_grad(self, tp_size, ep_size):
        if get_runtime_world_size() != 4:
            pytest.skip("single-node pplx test requires WORLD_SIZE=4")
        self.container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=ep_size,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="pplx_garden",
            moe_permute_fusion=False,
            test_dtype=torch.bfloat16,
        )

        moe_layer = self.container.moe_layer
        hidden_states = torch.randn(
            (16, 4, moe_layer.config.hidden_size), dtype=self.container.test_dtype
        )
        hidden_states = hidden_states.cuda()

        with torch.no_grad():
            output, _ = moe_layer(hidden_states)

        assert output.shape == hidden_states.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", PPLX_MULTI_NODE_FWD_BWD_CASES)
    def test_multi_node_forward_backward(self, tp_size, ep_size):
        if get_runtime_world_size() != 8:
            pytest.skip("multi-node pplx test requires WORLD_SIZE=8")
        self.container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="pplx_garden",
            moe_permute_fusion=False,
            test_dtype=torch.bfloat16,
        )
        self.container.dispatcher_dropless_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", PPLX_MULTI_NODE_INFERENCE_CASES)
    def test_multi_node_inference_no_grad(self, tp_size, ep_size):
        if get_runtime_world_size() != 8:
            pytest.skip("multi-node pplx test requires WORLD_SIZE=8")
        self.container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=ep_size,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="pplx_garden",
            moe_permute_fusion=False,
            test_dtype=torch.bfloat16,
        )

        moe_layer = self.container.moe_layer
        hidden_states = torch.randn(
            (16, 4, moe_layer.config.hidden_size), dtype=self.container.test_dtype
        )
        hidden_states = hidden_states.cuda()

        with torch.no_grad():
            output, _ = moe_layer(hidden_states)

        assert output.shape == hidden_states.shape


def test_pplx_garden_config_rejects_capacity_mode():
    with pytest.raises(
        ValueError,
        match="pplx_garden backend does not support moe_expert_capacity_factor",
    ):
        TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=8,
            num_moe_experts=8,
            moe_router_topk=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="pplx_garden",
            moe_expert_capacity_factor=1.0,
            use_cpu_initialization=True,
        )


def test_pplx_garden_config_rejects_fp8():
    with pytest.raises(
        ValueError, match="pplx_garden backend does not yet support fp8/fp4"
    ):
        TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=8,
            num_moe_experts=8,
            moe_router_topk=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="pplx_garden",
            fp8="hybrid",
            use_cpu_initialization=True,
        )


def test_pplx_garden_config_rejects_invalid_nets_per_gpu():
    with pytest.raises(ValueError, match="moe_pplx_garden_nets_per_gpu > 0"):
        TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=8,
            num_moe_experts=8,
            moe_router_topk=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="pplx_garden",
            moe_pplx_garden_nets_per_gpu=0,
            use_cpu_initialization=True,
        )


def test_pplx_garden_config_rejects_cuda_graphs():
    with pytest.raises(
        ValueError, match="pplx_garden backend does not yet support cuda_graph_impl"
    ):
        TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=8,
            num_moe_experts=8,
            moe_router_topk=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="pplx_garden",
            cuda_graph_impl="local",
            use_cpu_initialization=True,
        )


def test_pplx_garden_config_rejects_shared_expert_overlap():
    with pytest.raises(
        ValueError,
        match="pplx_garden backend does not support moe_shared_expert_overlap",
    ):
        TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=8,
            num_moe_experts=8,
            moe_router_topk=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="pplx_garden",
            moe_shared_expert_overlap=True,
            use_cpu_initialization=True,
        )


def test_pplx_garden_config_rejects_ep_overlap():
    with pytest.raises(
        ValueError,
        match="pplx_garden backend does not support overlap_moe_expert_parallel_comm",
    ):
        TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=8,
            num_moe_experts=8,
            moe_router_topk=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="pplx_garden",
            overlap_moe_expert_parallel_comm=True,
            bf16=True,
            use_cpu_initialization=True,
        )

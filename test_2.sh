set -euo pipefail

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9001
export FI_CXI_ENABLE_WRITEDATA=1
export MEGATRON_PPLX_DEBUG=1
export MEGATRON_PPLX_DEBUG_STRICT_METADATA=1
export MEGATRON_PPLX_DEBUG_STRICT_TOPK=1
export CUDA_LAUNCH_BLOCKING=1
export RUST_BACKTRACE=full
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1 # for torch .item() calls
export RUST_LOG=debug
export PPLX_DEBUG_SYNC_DISPATCH_SEND=1
export PPLX_DEBUG_LAUNCHER_DISPATCH_SEND=1
export PPLX_DEBUG_SKIP_PADDED_INDEX_COPY=0
# export PPLX_DEBUG_PADDED_INDEX_FILL=1
unset PPLX_DEBUG_PADDED_INDEX_FILL

# export NCCL_DEBUG=TRACE
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# export OMP_NUM_THREADS=64

# python -m torch.distributed.run \
#     --nnodes="$SLURM_NNODES" \
#     --node-rank="$SLURM_NODEID" \
#     --nproc-per-node=4 \
#     --rdzv-id="$SLURM_JOB_ID" \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
#     -m pytest \
#     tests/unit_tests/transformer/moe/test_token_dispatcher.py \
#     -k "TestPplxGardenFlexDispatcher" \
#     -v

# export CUDA_VISIBLE_DEVICES=0,1
NUMBER_OF_GPUS_PER_NODE=4

if [ "${DEBUG:-0}" = "1" ]; then
    echo "Running in debug mode with debugpy"
    python -m debugpy --listen 4567 --wait-for-client \
        -m torch.distributed.run \
        --nnodes="$SLURM_NNODES" \
        --node-rank="$SLURM_NODEID" \
        --nproc-per-node=$NUMBER_OF_GPUS_PER_NODE \
        --rdzv-id="$SLURM_JOB_ID" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
        -m pytest \
        tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestPplxGardenFlexDispatcher::test_single_node_forward_backward \
        -v
else
    echo "Running in normal mode"
    python -m torch.distributed.run \
        --nnodes="$SLURM_NNODES" \
        --node-rank="$SLURM_NODEID" \
        --nproc-per-node=$NUMBER_OF_GPUS_PER_NODE \
        --rdzv-id="$SLURM_JOB_ID" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
        -m pytest \
        tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestPplxGardenFlexDispatcher::test_single_node_forward_backward \
        -v \
        # -k "test_single_node_forward_backward[128-8-1-4]"
fi

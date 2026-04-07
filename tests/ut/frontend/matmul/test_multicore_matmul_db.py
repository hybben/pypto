"""Multi-core matmul double-buffer test for the @fe.kernel decorator.

This test validates matrix multiplication with double-buffer optimization:
- C[M, N] = A[M, K] @ B[K, N]
- Uses ping-pong buffers to overlap compute and memory transfer
- While computing on ping buffers, load data into pong buffers

Double-buffer optimization:
- Two sets of buffers (ping and pong) for A and B matrices
- Alternates between ping and pong for each K-tile iteration
- Hides memory latency behind computation

Multi-core implementation:
- Distributes M-tiles across Cube cores via strided loop
- Each core processes: core_id, core_id+num_cores, core_id+2*num_cores, ...
"""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.op.manual as plm


M = pl.DynVar('M')
K = pl.DynVar('K')
N = pl.DynVar('N')

TILE = 128
TILE_SIZE_FP16 = 32768
TILE_SIZE_FP32 = 65536


@fe.kernel
def multicore_matmul_db_kernel(
    a: pl.Tensor[[M, K], pl.FP16],
    b: pl.Tensor[[K, N], pl.FP16],
    c: pl.Tensor[[M, N], pl.FP32]
) -> pl.Tensor[[M, N], pl.FP32]:
    tile_type_a_load = plm.TileType(
        shape=[TILE, TILE],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Mat,
        blayout=2,
        slayout=1,
    )
    tile_a_load_0 = plm.make_tile(tile_type_a_load, addr=0x00000, size=TILE_SIZE_FP16)
    tile_a_load_1 = plm.make_tile(tile_type_a_load, addr=0x10000, size=TILE_SIZE_FP16)
    tile_a_load_buf = (tile_a_load_0, tile_a_load_1)

    tile_type_b_load = plm.TileType(
        shape=[TILE, TILE],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Mat,
        blayout=2,
        slayout=1,
    )
    tile_b_load_0 = plm.make_tile(tile_type_b_load, addr=0x08000, size=TILE_SIZE_FP16)
    tile_b_load_1 = plm.make_tile(tile_type_b_load, addr=0x18000, size=TILE_SIZE_FP16)
    tile_b_load_buf = (tile_b_load_0, tile_b_load_1)

    tile_type_a = plm.TileType(
        shape=[TILE, TILE],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Left,
        blayout=1,
        slayout=1,
    )
    tile_a_0 = plm.make_tile(tile_type_a, addr=0x00000, size=TILE_SIZE_FP16)
    tile_a_1 = plm.make_tile(tile_type_a, addr=0x08000, size=TILE_SIZE_FP16)
    tile_a_buf = (tile_a_0, tile_a_1)

    tile_type_b = plm.TileType(
        shape=[TILE, TILE],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Right,
        blayout=1,
        slayout=2,
    )
    tile_b_0 = plm.make_tile(tile_type_b, addr=0x00000, size=TILE_SIZE_FP16)
    tile_b_1 = plm.make_tile(tile_type_b, addr=0x08000, size=TILE_SIZE_FP16)
    tile_b_buf = (tile_b_0, tile_b_1)

    tile_type_c = plm.TileType(
        shape=[TILE, TILE],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Acc,
        blayout=2,
        slayout=1,
        fractal=1024,
    )
    tile_c = plm.make_tile(tile_type_c, addr=0x00000, size=TILE_SIZE_FP32)
    
    with pl.section_cube():
        M_dim = pl.tensor.dim(a, 0)
        K_dim = pl.tensor.dim(a, 1)
        N_dim = pl.tensor.dim(b, 1)
        
        block_idx = pl.block.get_block_idx()
        core_id = pl.block.index_cast(block_idx)
        block_num = pl.block.get_block_num()
        num_cores = pl.block.index_cast(block_num)
        
        m_tiles = (M_dim + (TILE - 1)) // TILE
        n_tiles = (N_dim + (TILE - 1)) // TILE
        k_tiles = (K_dim + (TILE - 1)) // TILE

        event_ids = (0, 1)

        for i in pl.range(core_id, m_tiles, num_cores):
            for j in pl.range(n_tiles):
                for k in pl.range(k_tiles):
                    buf_idx = k % 2

                    plm.load(tile_a_load_buf[buf_idx], a, [i * TILE, k * TILE])
                    plm.load(tile_b_load_buf[buf_idx], b, [k * TILE, j * TILE])

                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=event_ids[buf_idx])
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=event_ids[buf_idx])

                    plm.move(tile_a_buf[buf_idx], tile_a_load_buf[buf_idx])
                    plm.move(tile_b_buf[buf_idx], tile_b_load_buf[buf_idx])

                    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=event_ids[buf_idx])
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=event_ids[buf_idx])

                    if k == 0:
                        plm.matmul(tile_c, tile_a_buf[buf_idx], tile_b_buf[buf_idx])
                    else:
                        plm.matmul_acc(tile_c, tile_c, tile_a_buf[buf_idx], tile_b_buf[buf_idx])

                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=event_ids[buf_idx])
                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=event_ids[buf_idx])

                pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                plm.store(c, tile_c, [i * TILE, j * TILE])

                pl.system.bar_all()

    return c


@fe.jit()
def test_multicore_matmul_db():
    compiled_lib = fe.compile(multicore_matmul_db_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:0"
    torch.npu.set_device(device)

    shapes = [
        [1024, 1024, 1024],
        [2048, 1024, 2048],
        [8192, 2048, 8192],
    ]
    torch.manual_seed(0)
    BLOCK_DIM = 24

    for M_val, K_val, N_val in shapes:
        print(f"\nTesting shape: ({M_val}, {K_val}) x ({K_val}, {N_val}) = ({M_val}, {N_val})")
        
        a = torch.randn(M_val, K_val, dtype=torch.float16, device=device)
        b = torch.randn(K_val, N_val, dtype=torch.float16, device=device)
        c = torch.zeros(M_val, N_val, dtype=torch.float32, device=device)
        
        fe.launch(None, BLOCK_DIM, compiled_lib, a, b, c)
        torch.npu.synchronize()
        
        print("***********npu output***********")
        print(c.shape, c.dtype)
        print(c)
        c_ref = torch.matmul(a.float(), b.float())
        print("***********golden output***********")
        print(c_ref.shape, c_ref.dtype)
        print(c_ref)
        
        torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)
        print("result equal!")


if __name__ == "__main__":
    test_multicore_matmul_db()
    print("\nAll tests passed!")

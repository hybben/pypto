"""Multi-core matmul basic test for the @fe.kernel decorator.

This test validates matrix multiplication with multi-core distribution:
- C[M, N] = A[M, K] @ B[K, N]
- Each Cube core processes a strided subset of M-tile-rows
- Uses get_block_idx() and get_block_num() for core indexing

Multi-core implementation:
- Distributes M-tiles across Cube cores via strided loop
- Core i processes tiles: i, i+num_cores, i+2*num_cores, ...
- Larger shapes to better utilize multi-core parallelism
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
def multicore_matmul_basic_kernel(
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
    tile_a_load = plm.make_tile(tile_type_a_load, addr=0x00000, size=TILE_SIZE_FP16)
    
    tile_type_b_load = plm.TileType(
        shape=[TILE, TILE],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Mat,
        blayout=2,
        slayout=1,
    )
    tile_b_load = plm.make_tile(tile_type_b_load, addr=0x08000, size=TILE_SIZE_FP16)
    
    tile_type_a = plm.TileType(
        shape=[TILE, TILE],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Left,
        blayout=1,
        slayout=1,
    )
    tile_a = plm.make_tile(tile_type_a, addr=0x00000, size=TILE_SIZE_FP16)
    
    tile_type_b = plm.TileType(
        shape=[TILE, TILE],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Right,
        blayout=1,
        slayout=2,
    )
    tile_b = plm.make_tile(tile_type_b, addr=0x00000, size=TILE_SIZE_FP16)
    
    tile_type_c = plm.TileType(
        shape=[TILE, TILE],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Acc,
        blayout=2,
        slayout=1,
        fractal=1024,
        valid_shape=[-1, -1],
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
        
        for i in pl.range(core_id, m_tiles, num_cores):
            for j in pl.range(n_tiles):
                for k in pl.range(k_tiles):
                    m_off = i * TILE
                    n_off = j * TILE
                    k_off = k * TILE
                    
                    plm.load(tile_a_load, a, [m_off, k_off])
                    plm.load(tile_b_load, b, [k_off, n_off])
                    
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                    
                    plm.move(tile_a, tile_a_load, target_memory=pl.MemorySpace.Left)
                    plm.move(tile_b, tile_b_load, target_memory=pl.MemorySpace.Right)
                    
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
                    
                    if k == 0:
                        plm.matmul(tile_c, tile_a, tile_b)
                    else:
                        plm.matmul_acc(tile_c, tile_c, tile_a, tile_b)
                    
                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=2)
                
                pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                plm.l0c_store(tile_c, [i * TILE, j * TILE], [TILE, TILE], c)
                
                pl.system.bar_all()

    return c


@fe.jit()
def test_multicore_matmul_basic():
    compiled_lib = fe.compile(multicore_matmul_basic_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:1"
    torch.npu.set_device(device)

    shapes = [
        [1024, 1024, 1024],
        [2048, 1024, 2048],
        [4096, 2048, 4096],
    ]
    torch.manual_seed(0)
    BLOCK_DIM = 4

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
    test_multicore_matmul_basic()
    print("\nAll tests passed!")

"""Multi-core softmax basic test for the @fe.kernel decorator.

This test validates softmax operation with multi-core distribution:
1. row_max: find max per row for numerical stability
2. row_expand_sub: subtract max from each element (broadcast)
3. exp: compute exponential
4. row_sum: sum of exp values per row
5. row_expand_div: divide by sum for normalization (broadcast)

Multi-core implementation:
- Uses get_block_idx() and get_block_num() to distribute work across cores
- Each core processes a strided subset of M-tile-rows
"""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.op.manual as plm


M = pl.DynVar('M')
N = pl.DynVar('N')

TILE_M = 64
TILE_N = 128
TILE_SIZE_FP32 = 32768
TILE_ROW_SIZE = 256


@fe.kernel
def multicore_softmax_kernel(
    a: pl.Tensor[[M, N], pl.FP32],
    out: pl.Tensor[[M, N], pl.FP32]
) -> pl.Tensor[[M, N], pl.FP32]:
    tile_type_a = plm.TileType(
        shape=[TILE_M, TILE_N],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_a = plm.make_tile(tile_type_a, addr=0x00000, size=TILE_SIZE_FP32)
    
    tile_type_tmp = plm.TileType(
        shape=[TILE_M, TILE_N],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_tmp = plm.make_tile(tile_type_tmp, addr=0x08000, size=TILE_SIZE_FP32)
    
    tile_type_row_max = plm.TileType(
        shape=[TILE_M, 1],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Vec,
        blayout=2,
    )
    tile_row_max = plm.make_tile(tile_type_row_max, addr=0x10000, size=TILE_ROW_SIZE)
    
    tile_type_row_sum = plm.TileType(
        shape=[TILE_M, 1],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Vec,
        blayout=2,
    )
    tile_row_sum = plm.make_tile(tile_type_row_sum, addr=0x10100, size=TILE_ROW_SIZE)
    
    tile_type_out = plm.TileType(
        shape=[TILE_M, TILE_N],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_out = plm.make_tile(tile_type_out, addr=0x18000, size=TILE_SIZE_FP32)
    
    with pl.section_vector():
        M_dim = pl.tensor.dim(a, 0)
        N_dim = pl.tensor.dim(a, 1)
        
        block_idx = pl.block.get_block_idx()
        core_id = pl.block.index_cast(block_idx)
        block_num = pl.block.get_block_num()
        num_cores = pl.block.index_cast(block_num)
        
        m_tiles = (M_dim + (TILE_M - 1)) // TILE_M
        
        for i in pl.range(core_id, m_tiles, num_cores):
            m_off = i * TILE_M
            
            pl.system.bar_all()
            plm.load(tile_a, a, [m_off, 0])
            
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            
            plm.row_max(tile_row_max, tile_a, tile_tmp)
            
            plm.row_expand_sub(tile_tmp, tile_a, tile_row_max)
            
            plm.exp(tile_a, tile_tmp)
            
            plm.row_sum(tile_row_sum, tile_a, tile_tmp)
            
            plm.row_expand_div(tile_out, tile_a, tile_row_sum)
            
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            
            plm.store(out, tile_out, [m_off, 0])
    
    return out


@fe.jit()
def test_multicore_softmax_basic():
    compiled_lib = fe.compile(multicore_softmax_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:1"
    torch.npu.set_device(device)

    shapes = [
        [64, 128],
        [128, 128],
        [256, 128],
        [512, 128],
        [1024, 128],
    ]
    torch.manual_seed(0)
    dtype = torch.float32
    BLOCK_DIM = 8

    for M_val, N_val in shapes:
        print(f"\nTesting shape: ({M_val}, {N_val})")
        
        a = torch.randn(M_val, N_val, dtype=dtype, device=device)
        out = torch.zeros(M_val, N_val, dtype=dtype, device=device)
        
        fe.launch(None, BLOCK_DIM, compiled_lib, a, out)
        torch.npu.synchronize()
        
        print("***********npu output***********")
        print(out.shape, out.dtype)
        print(out)
        out_ref = torch.softmax(a, dim=1)
        print("***********golden output***********")
        print(out_ref.shape, out_ref.dtype)
        print(out_ref)
        
        torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
        print("result equal!")


if __name__ == "__main__":
    test_multicore_softmax_basic()
    print("\nAll tests passed!")

"""Dynamic divs (scalar division) tests for the @fe.kernel decorator.

Tests the divs operation: out = tile / scalar
Used in FlashAttention for softmax normalization (dividing by sum of exp values).
"""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


M = pl.DynVar('M')
N = pl.DynVar('N')
DIVISOR = 4.0

@fe.kernel
def dynamic_divs_kernel(
    a: pl.Tensor[[M, N], pl.FP32],
    out: pl.Tensor[[M, N], pl.FP32]
) -> pl.Tensor[[M, N], pl.FP32]:
    tile_type_a = plm.TileType(
        shape=[64, 64],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    
    tile_type_out = plm.TileType(
        shape=[64, 64],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_out = plm.make_tile(tile_type_out, addr=0x4000, size=16384)
    
    with pl.section_vector():
        M_dim = pl.tensor.dim(a, 0)
        N_dim = pl.tensor.dim(a, 1)
        
        for i in pl.range(0, M_dim, 64):
            for j in pl.range(0, N_dim, 64):
                pl.system.bar_all()
                plm.load(tile_a, a, [i, j])
                
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                
                plm.divs(tile_out, tile_a, DIVISOR)
                
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                
                plm.store(out, tile_out, [i, j])
    
    return out


@fe.jit()
def test_dynamic_divs():
    compiled_lib = fe.compile(dynamic_divs_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:1"
    torch.npu.set_device(device)

    shapes = [
        [64, 64],
        [128, 128],
        [256, 256],
    ]
    torch.manual_seed(0)
    dtype = torch.float32

    for M_val, N_val in shapes:
        print(f"\nTesting shape: ({M_val}, {N_val})")
        
        a = torch.randn(M_val, N_val, dtype=dtype, device=device)
        out = torch.zeros(M_val, N_val, dtype=dtype, device=device)
        
        fe.launch(None, 1, compiled_lib, a, out)
        torch.npu.synchronize()
        
        print("***********npu output***********")
        print(out.shape, out.dtype)
        print(out)
        
        out_ref = a / DIVISOR
        print("***********golden output***********")
        print(out_ref.shape, out_ref.dtype)
        print(out_ref)
        
        torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
        print("result equal!")


if __name__ == "__main__":
    test_dynamic_divs()
    print("\nAll tests passed!")

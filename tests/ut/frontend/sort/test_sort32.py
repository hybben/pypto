# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend tests for plm.sort32 function."""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.op.manual as plm


@fe.kernel
def sort32_kernel(
    a: pl.Tensor[[1, 32], pl.FP16],
    idx_in: pl.Tensor[[1, 32], pl.UINT32],
    sorted_out: pl.Tensor[[1, 128], pl.FP16],
) -> pl.Tensor[[1, 128], pl.FP16]:
    """Sort 32 elements using sort32.
    
    For FP16, dst has 4x columns of src because:
    - TYPE_COEF = sizeof(float)/sizeof(half) = 2
    - dst stores value-index pairs, so cols = src_cols * TYPE_COEF * 2
    - For 32 FP16 elements: dst_cols = 32 * 2 * 2 = 128
    """
    pl.system.bar_all()
    
    tile_src = plm.make_tile(
        plm.TileType(shape=[1, 32], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0000, size=64
    )
    tile_dst = plm.make_tile(
        plm.TileType(shape=[1, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0040, size=256
    )
    tile_idx = plm.make_tile(
        plm.TileType(shape=[1, 32], dtype=pl.UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0140, size=128
    )
    
    with pl.section_vector():
        plm.load(tile_src, a, [0, 0])
        plm.load(tile_idx, idx_in, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.sort32(tile_dst, tile_src, tile_idx)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(sorted_out, tile_dst, [0, 0])
    
    return sorted_out


@fe.jit()
def test_sort32():
    device = "npu:7"
    torch.npu.set_device(device)

    torch.manual_seed(0)
    dtype = torch.float16
    
    a = torch.rand([1, 32], device=device, dtype=dtype)
    idx_in = torch.arange(32, device=device, dtype=torch.int32).unsqueeze(0)
    sorted_out = torch.zeros([1, 128], device=device, dtype=dtype)
    
    print("Input tensor:")
    print("a =", a)
    print("idx_in =", idx_in)
    
    expected_sorted, expected_idx = torch.sort(a, dim=1, descending=True)
    print("\nExpected sorted (descending, torch.sort):")
    print(expected_sorted)
    print("Expected indices:")
    print(expected_idx)

    compiled_lib = fe.compile(sort32_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)
    fe.launch(None, 1, compiled_lib, a, idx_in, sorted_out)
    
    torch.npu.synchronize()
    
    print("\n***********npu output***********")
    print("sorted_out shape:", sorted_out.shape)
    
    values = sorted_out[0, 0::4]
    print("\nExtracted values (descending order):", values)
    
    sorted_out_cpu = sorted_out.cpu()
    indices = torch.zeros(32, dtype=torch.int32)
    for i in range(32):
        idx_low = sorted_out_cpu[0, i*4 + 2].view(torch.uint16).item()
        idx_high = sorted_out_cpu[0, i*4 + 3].view(torch.uint16).item()
        indices[i] = idx_low | (idx_high << 16)
    print("Extracted indices:", indices)
    
    sorted_diff = torch.abs(values - expected_sorted).max().item()
    print(f"\nMax diff between sorted values and expected: {sorted_diff}")
    
    assert sorted_diff < 1e-3, f"sort32 failed: max diff {sorted_diff}"
    print("sort32 test passed!")


if __name__ == "__main__":
    test_sort32()
    print("\nAll tests passed!")

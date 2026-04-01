# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""NPU test: verify SoC specs are dynamically obtained from CANN runtime.

Run prerequisite:
    source ~/setup.sh

Run:
    pytest tests/ut/frontend/test_soc_platform.py -v
"""

import pytest
from pypto import ir
from pypto.backend import Backend910B_PTO


def test_soc_core_counts():
    """cube_core_cnt and vector_core_cnt must be positive integers."""
    backend = Backend910B_PTO.instance()
    soc = backend.soc
    # TODO: assert exact values per chip type once mapping is available
    assert soc.total_core_count() > 0
    assert soc.total_die_count() == 1


def test_soc_mem_sizes():
    """All memory fields must be positive values."""
    backend = Backend910B_PTO.instance()
    mem_types = [
        ir.MemorySpace.Vec,    # ub_size
        ir.MemorySpace.Mat,    # l1_size
        ir.MemorySpace.Left,   # l0_a_size
        ir.MemorySpace.Right,  # l0_b_size
        ir.MemorySpace.Acc,    # l0_c_size
    ]
    for mem_type in mem_types:
        size = backend.get_mem_size(mem_type)
        # TODO: assert exact values per chip type once mapping is available
        assert size > 0, f"{mem_type} size should be positive, got {size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

__global__ AICORE void fa_perf_kernel(__gm__ half* q, __gm__ half* k, __gm__ half* v, __gm__ half* o, __gm__ float* qk_buf, __gm__ half* p_buf, __gm__ float* pv_buf, int32_t Sq, int32_t D, int32_t Skv, int32_t SqFifo, __gm__ int64_t* ffts_addr)
{
    const int32_t sq_dim = Sq;
    const int32_t skv_dim = Skv;
    set_ffts_base_addr((unsigned long)ffts_addr);

    using qGlobalShapeDim5 = Shape<1, 1, 1, 64, 128>;
    using qGlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using qGlobalType = GlobalTensor<half, qGlobalShapeDim5, qGlobalStrideDim5>;
    qGlobalType qGlobal(q);

    using kGlobalShapeDim5 = Shape<1, 1, 1, 128, 128>;
    using kGlobalStrideDim5 = Stride<128, 128, 128, 1, 128>;
    using kGlobalType = GlobalTensor<half, kGlobalShapeDim5, kGlobalStrideDim5, Layout::DN>;
    kGlobalType kGlobal(k);

    using vGlobalShapeDim5 = Shape<1, 1, 1, 128, 128>;
    using vGlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using vGlobalType = GlobalTensor<half, vGlobalShapeDim5, vGlobalStrideDim5>;
    vGlobalType vGlobal(v);

    using oGlobalShapeDim5 = Shape<1, 1, 1, 64, 128>;
    using oGlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using oGlobalType = GlobalTensor<half, oGlobalShapeDim5, oGlobalStrideDim5>;
    oGlobalType oGlobal(o);

    using qk_bufGlobalShapeDim5 = Shape<1, 1, 1, 64, 128>;
    using qk_bufGlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using qk_bufGlobalType = GlobalTensor<float, qk_bufGlobalShapeDim5, qk_bufGlobalStrideDim5>;
    qk_bufGlobalType qk_bufGlobal(qk_buf);

    using p_bufGlobalShapeDim5 = Shape<1, 1, 1, 64, 128>;
    using p_bufGlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using p_bufGlobalType = GlobalTensor<half, p_bufGlobalShapeDim5, p_bufGlobalStrideDim5>;
    p_bufGlobalType p_bufGlobal(p_buf);

    using pv_bufGlobalShapeDim5 = Shape<1, 1, 1, 64, 128>;
    using pv_bufGlobalStrideDim5 = Stride<1, 1, 1, 128, 1>;
    using pv_bufGlobalType = GlobalTensor<float, pv_bufGlobalShapeDim5, pv_bufGlobalStrideDim5>;
    pv_bufGlobalType pv_bufGlobal(pv_buf);

    const int32_t sq_tiles = ((sq_dim + 127) >> 7);
    const int32_t skv_tiles = ((skv_dim + 127) >> 7);
    const auto num_cores = (int32_t)(get_block_num());
    const auto core_id = (int32_t)(get_block_idx());

    #if defined(__DAV_CUBE__)
    using q_mat_buf_Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    q_mat_buf_Type q_mat_buf_0(64, 128); TASSIGN(q_mat_buf_0, 0x0);
    q_mat_buf_Type q_mat_buf_1(64, 128); TASSIGN(q_mat_buf_1, 0x4000);
    using k_mat_buf_Type = Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    k_mat_buf_Type k_mat_buf_0(128, 128); TASSIGN(k_mat_buf_0, 0x8000);
    k_mat_buf_Type k_mat_buf_1(128, 128); TASSIGN(k_mat_buf_1, 0x10000);
    using p_mat_buf_Type = Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    p_mat_buf_Type p_mat_buf_0(64, 128); TASSIGN(p_mat_buf_0, 0x18000);
    p_mat_buf_Type p_mat_buf_1(64, 128); TASSIGN(p_mat_buf_1, 0x1c000);
    using v_mat_buf_Type = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    v_mat_buf_Type v_mat_buf_0(128, 128); TASSIGN(v_mat_buf_0, 0x20000);
    v_mat_buf_Type v_mat_buf_1(128, 128); TASSIGN(v_mat_buf_1, 0x28000);
    using left_buf_Type = Tile<TileType::Left, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
    left_buf_Type left_buf_0(64, 128); TASSIGN(left_buf_0, 0x0);
    left_buf_Type left_buf_1(64, 128); TASSIGN(left_buf_1, 0x4000);
    using right_buf_Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    right_buf_Type right_buf_0(128, 128); TASSIGN(right_buf_0, 0x0);
    right_buf_Type right_buf_1(128, 128); TASSIGN(right_buf_1, 0x8000);
    using acc_buf_Type = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    acc_buf_Type acc_buf_0(64, 128); TASSIGN(acc_buf_0, 0x0);
    acc_buf_Type acc_buf_1(64, 128); TASSIGN(acc_buf_1, 0x8000);
    #endif

    #if defined(__DAV_VEC__)
    using qk_vecType = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    qk_vecType qk_vec(64, 128); TASSIGN(qk_vec, 0x0);
    using tmp_vecType = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    tmp_vecType tmp_vec(64, 128); TASSIGN(tmp_vec, 0x8000);
    using p_f16Type = Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    p_f16Type p_f16(64, 128); TASSIGN(p_f16, 0x10000);
    using reduce_dstType = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    reduce_dstType reduce_dst(64, 1); TASSIGN(reduce_dst, 0x14000);
    using reduce_dst_rmType = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    reduce_dst_rmType reduce_dst_rm(1, 64); TASSIGN(reduce_dst_rm, 0x14000);
    using gmax_rmType = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    gmax_rmType gmax_rm_0(1, 64); TASSIGN(gmax_rm_0, 0x14100);
    gmax_rmType gmax_rm_1(1, 64); TASSIGN(gmax_rm_1, 0x14200);
    using gsum_colType = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    gsum_colType gsum_col_0(64, 1); TASSIGN(gsum_col_0, 0x14300);
    gsum_colType gsum_col_1(64, 1); TASSIGN(gsum_col_1, 0x14400);
    using gsum_rmType = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    gsum_rmType gsum_rm_0(1, 64); TASSIGN(gsum_rm_0, 0x14300);
    gsum_rmType gsum_rm_1(1, 64); TASSIGN(gsum_rm_1, 0x14400);
    using ecorr_colType = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512>;
    ecorr_colType ecorr_col_0(64, 1); TASSIGN(ecorr_col_0, 0x14500);
    ecorr_colType ecorr_col_1(64, 1); TASSIGN(ecorr_col_1, 0x14600);
    using ecorr_rmType = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    ecorr_rmType ecorr_rm_0(1, 64); TASSIGN(ecorr_rm_0, 0x14500);
    ecorr_rmType ecorr_rm_1(1, 64); TASSIGN(ecorr_rm_1, 0x14600);
    using running_oType = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    running_oType running_o(64, 128); TASSIGN(running_o, 0x14700);
    using pv_vecType = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    pv_vecType pv_vec(64, 128); TASSIGN(pv_vec, 0x1c700);
    using o_f16Type = Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>;
    o_f16Type o_f16(64, 128); TASSIGN(o_f16, 0x24700);
    #endif


    #if defined(__DAV_CUBE__)
    // --- CUBE ---
    const event_t E0 = (event_t)0, E1 = (event_t)1, E2 = (event_t)2, E3 = (event_t)3;
    const event_t eid01[] = {E0, E1};
    const event_t eid23[] = {E2, E3};

    q_mat_buf_Type  q_mat_buf[] = {q_mat_buf_0, q_mat_buf_1};
    k_mat_buf_Type  k_mat_buf[] = {k_mat_buf_0, k_mat_buf_1};
    left_buf_Type   left_buf[]  = {left_buf_0, left_buf_1};
    right_buf_Type  right_buf[] = {right_buf_0, right_buf_1};
    acc_buf_Type    acc_buf[]   = {acc_buf_0, acc_buf_1};
    v_mat_buf_Type  v_mat_buf[] = {v_mat_buf_0, v_mat_buf_1};
    p_mat_buf_Type  p_mat_buf[] = {p_mat_buf_0, p_mat_buf_1};

    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0); set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2); set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);     set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

    int32_t l0ab = 0, l0c = 0, q_count = 0;
    // Pre-compute core's pv_buf base (constant across all iterations)
    const int64_t pv_core_base = (int64_t)core_id * 512;

    for (int32_t qi = core_id; qi < sq_tiles; qi += num_cores) {
        const int64_t sq_off = (int64_t)qi << 7;  // qi * 128

        for (int32_t row_idx = 0; row_idx < 2; row_idx++) {
            const int64_t row_off = (int64_t)row_idx << 6;  // row_idx * 64
            const int32_t qmi = q_count & 1;  // q_mat_idx
            // Pre-compute row base for qk/p buffer addressing
            const int64_t row_base = sq_off + row_off;
            // Pre-compute Q global address (shared by QK prologue + all QK in this row_off)
            __gm__ half* q_addr = q + (row_base * D);
            // Pre-compute pv_buf row bases
            const int64_t pv_q_base = pv_core_base + ((int64_t)(qmi << 1) << 7); // qmi*2*128

            // ---- Prologue: QK task_id=0 ----
            {
                const int32_t bi = (q_count * skv_tiles) & 1;

                wait_flag(PIPE_MTE1, PIPE_MTE2, eid01[bi]);
                TASSIGN(qGlobal, q_addr);
                TLOAD(q_mat_buf[qmi], qGlobal);
                TASSIGN(kGlobal, k);
                TLOAD(k_mat_buf[bi], kGlobal);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

                wait_flag(PIPE_M, PIPE_MTE1, eid01[l0ab]);
                TMOV(left_buf[l0ab], q_mat_buf[qmi]);
                TMOV(right_buf[l0ab], k_mat_buf[bi]);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                set_flag(PIPE_MTE1, PIPE_MTE2, eid01[bi]);

                wait_flag(PIPE_FIX, PIPE_M, eid01[l0c]);
                TMATMUL(acc_buf[l0c], left_buf[l0ab], right_buf[l0ab]);
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                set_flag(PIPE_M, PIPE_MTE1, eid01[l0ab]);

                TASSIGN(qk_bufGlobal, qk_buf + (row_base * Skv));
                TSTORE(qk_bufGlobal, acc_buf[l0c]);
                set_flag(PIPE_FIX, PIPE_M, eid01[l0c]);

                l0ab ^= 1; l0c ^= 1;
                ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, E0));
            }

            // ---- Main loop ----
            for (int32_t ki = 0; ki < skv_tiles; ki++) {
                const int32_t nk = ki + 1;

                // -- QK pre-compute for nk --
                if (nk < skv_tiles) {
                    const int32_t bi_qk = ((q_count * skv_tiles + nk) & 1);
                    const int64_t skv_off = (int64_t)nk << 7;
                    const int32_t fifo = nk & 1;

                    wait_flag(PIPE_MTE1, PIPE_MTE2, eid01[bi_qk]);
                    TASSIGN(kGlobal, k + (skv_off * D));
                    TLOAD(k_mat_buf[bi_qk], kGlobal);
                    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

                    wait_flag(PIPE_M, PIPE_MTE1, eid01[l0ab]);
                    TMOV(left_buf[l0ab], q_mat_buf[qmi]);
                    TMOV(right_buf[l0ab], k_mat_buf[bi_qk]);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    set_flag(PIPE_MTE1, PIPE_MTE2, eid01[bi_qk]);

                    wait_flag(PIPE_FIX, PIPE_M, eid01[l0c]);
                    TMATMUL(acc_buf[l0c], left_buf[l0ab], right_buf[l0ab]);
                    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                    set_flag(PIPE_M, PIPE_MTE1, eid01[l0ab]);

                    TASSIGN(qk_bufGlobal, qk_buf + (((int64_t)fifo * sq_dim + row_base) * Skv + skv_off));
                    TSTORE(qk_bufGlobal, acc_buf[l0c]);
                    set_flag(PIPE_FIX, PIPE_M, eid01[l0c]);

                    l0ab ^= 1; l0c ^= 1;
                    ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, eid01[fifo]));
                }

                // -- PV for ki --
                {
                    const int32_t bi_pv = ((q_count * skv_tiles + ki) & 1);
                    const int64_t sv_off = (int64_t)ki << 7;
                    const int32_t pv_slot = ki & 1;

                    wait_flag(PIPE_MTE1, PIPE_MTE2, eid23[bi_pv]);
                    TASSIGN(vGlobal, v + (sv_off * D));
                    TLOAD(v_mat_buf[bi_pv], vGlobal);

                    wait_flag_dev(eid23[pv_slot]);
                    TASSIGN(p_bufGlobal, p_buf + (((int64_t)pv_slot * sq_dim + row_base) * Skv + sv_off));
                    TLOAD(p_mat_buf[bi_pv], p_bufGlobal);
                    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

                    wait_flag(PIPE_M, PIPE_MTE1, eid01[l0ab]);
                    TMOV(left_buf[l0ab], p_mat_buf[bi_pv]);
                    TMOV(right_buf[l0ab], v_mat_buf[bi_pv]);
                    set_flag(PIPE_MTE1, PIPE_MTE2, eid23[bi_pv]);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

                    wait_flag(PIPE_FIX, PIPE_M, eid01[l0c]);
                    TMATMUL(acc_buf[l0c], left_buf[l0ab], right_buf[l0ab]);
                    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                    set_flag(PIPE_M, PIPE_MTE1, eid01[l0ab]);

                    TASSIGN(pv_bufGlobal, pv_buf + ((pv_q_base + ((int64_t)pv_slot << 7) + row_off) * D));
                    TSTORE(pv_bufGlobal, acc_buf[l0c]);
                    set_flag(PIPE_FIX, PIPE_M, eid01[l0c]);

                    l0ab ^= 1; l0c ^= 1;
                    ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, (event_t)(4 + pv_slot)));
                }
            }
            q_count++;
        }
    }

    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0); wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2); wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);     wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    #endif

    #if defined(__DAV_VEC__)
    // --- VECTOR ---
    const event_t E0 = (event_t)0, E1 = (event_t)1;
    const event_t eid01[] = {E0, E1};
    const event_t eid23[] = {(event_t)2, (event_t)3};
    const event_t eid45[] = {(event_t)4, (event_t)5};

    gmax_rmType     gmax_rm[]  = {gmax_rm_0, gmax_rm_1};
    gsum_colType    gsum_col[] = {gsum_col_0, gsum_col_1};
    gsum_rmType     gsum_rm[]  = {gsum_rm_0, gsum_rm_1};
    ecorr_rmType    ecorr_rm[] = {ecorr_rm_0, ecorr_rm_1};
    ecorr_colType   ecorr_col[]= {ecorr_col_0, ecorr_col_1};

    int32_t q_count = 0;
    const int64_t pv_core_base = (int64_t)core_id * 512;

    for (int32_t qi = core_id; qi < sq_tiles; qi += num_cores) {
        const int64_t sq_off = (int64_t)qi << 7;

        for (int32_t row_idx = 0; row_idx < 2; row_idx++) {
            const int64_t row_off = (int64_t)row_idx << 6;
            const int32_t qx = q_count & 1;
            const int64_t row_base = sq_off + row_off;
            const int64_t pv_q_base = pv_core_base + ((int64_t)(qx << 1) << 7);

            // ---- Prologue: softmax task_id=0 ----
            {
                wait_flag_dev(E0);
                TASSIGN(qk_bufGlobal, qk_buf + (row_base * Skv));
                TLOAD(qk_vec, qk_bufGlobal);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

                TROWMAX(reduce_dst, qk_vec, tmp_vec);
                pipe_barrier(PIPE_V);
                TROWEXPANDSUB(tmp_vec, qk_vec, reduce_dst);
                TMULS(gmax_rm[qx], reduce_dst_rm, 1.0f);
                TMULS(tmp_vec, tmp_vec, 0.088388f);
                TEXP(qk_vec, tmp_vec);
                pipe_barrier(PIPE_V);
                TROWSUM(reduce_dst, qk_vec, tmp_vec);
                pipe_barrier(PIPE_V);
                TMULS(gsum_rm[qx], reduce_dst_rm, 1.0f);
                TCVT(p_f16, qk_vec, RoundMode::CAST_ROUND);

                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TASSIGN(p_bufGlobal, p_buf + (row_base * Skv));
                TSTORE(p_bufGlobal, p_f16);
                ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_MODE_VAL, eid23[0]));
            }

            // ---- Main loop ----
            for (int32_t ki = 0; ki < skv_tiles; ki++) {
                const int32_t nk = ki + 1;

                // -- Softmax for nk --
                if (nk < skv_tiles) {
                    const int32_t fifo = nk & 1;
                    const int64_t skv_off = (int64_t)nk << 7;

                    wait_flag_dev(eid01[fifo]);
                    TASSIGN(qk_bufGlobal, qk_buf + (((int64_t)fifo * sq_dim + row_base) * Skv + skv_off));
                    TLOAD(qk_vec, qk_bufGlobal);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

                    TROWMAX(reduce_dst, qk_vec, tmp_vec);
                    pipe_barrier(PIPE_V);
                    TMAX(reduce_dst_rm, reduce_dst_rm, gmax_rm[qx]);
                    pipe_barrier(PIPE_V);
                    TSUB(ecorr_rm[fifo], gmax_rm[qx], reduce_dst_rm);
                    pipe_barrier(PIPE_V);
                    TMULS(gmax_rm[qx], reduce_dst_rm, 1.0f);
                    pipe_barrier(PIPE_V);
                    TROWEXPANDSUB(tmp_vec, qk_vec, reduce_dst);
                    TMULS(ecorr_rm[fifo], ecorr_rm[fifo], 0.088388f);
                    TMULS(tmp_vec, tmp_vec, 0.088388f);
                    TEXP(ecorr_rm[fifo], ecorr_rm[fifo]);
                    TEXP(qk_vec, tmp_vec);
                    TCVT(p_f16, qk_vec, RoundMode::CAST_ROUND);
                    pipe_barrier(PIPE_V);
                    TMUL(gsum_rm[qx], gsum_rm[qx], ecorr_rm[fifo]);
                    TROWSUM(reduce_dst, qk_vec, tmp_vec);
                    pipe_barrier(PIPE_V);
                    TADD(gsum_rm[qx], gsum_rm[qx], reduce_dst_rm);

                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    TASSIGN(p_bufGlobal, p_buf + (((int64_t)fifo * sq_dim + row_base) * Skv + skv_off));
                    TSTORE(p_bufGlobal, p_f16);
                    ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_MODE_VAL, eid23[fifo]));
                }

                // -- GU for ki --
                {
                    const int32_t pvs = ki & 1;
                    wait_flag_dev(eid45[pvs]);

                    __gm__ float* pv_addr = pv_buf + ((pv_q_base + ((int64_t)pvs << 7) + row_off) * D);
                    if (ki == 0) {
                        TASSIGN(pv_bufGlobal, pv_addr);
                        TLOAD(running_o, pv_bufGlobal);
                        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    } else {
                        TASSIGN(pv_bufGlobal, pv_addr);
                        TLOAD(pv_vec, pv_bufGlobal);
                        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                        TROWEXPANDMUL(running_o, running_o, ecorr_col[pvs]);
                        TADD(running_o, running_o, pv_vec);
                    }
                }
            }

            TROWEXPANDDIV(running_o, running_o, gsum_col[qx]);
            TCVT(o_f16, running_o, RoundMode::CAST_ROUND);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TASSIGN(oGlobal, o + (row_base * D));
            TSTORE(oGlobal, o_f16);
            q_count++;
        }
    }
    #endif
}

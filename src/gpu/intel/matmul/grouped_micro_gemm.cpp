/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/intel/matmul/grouped_micro_gemm.hpp"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include "common/memory_tracking.hpp"
#include "gemmstone/microkernel/shim.hpp"
#include "gemmstone/microkernel_selector.hpp"
#include "gemmstone/strategy_parser.hpp"
#include "gpu/intel/compute/ukernels.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/gemm/jit/gen_kernel.hpp"

#define VCHECK_MATMUL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, matmul, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

status_t grouped_micro_gemm_t::pd_t::init_microkernels(impl::engine_t *engine) {
    using namespace jit;
    using namespace gemmstone;
    using namespace gemmstone::microkernel;
    using gemm::jit::convert_dnnl_to_kernel_type;

    assert(engine->kind() == engine_kind::gpu);
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    auto *dev_info = intel_engine->device_info();
    bool use_systolic_ukernel = dev_info->mayiuse_systolic();

    /* Get device information */
    HWInformation hw_info;
    hw_info.euCount = dev_info->eu_count();
    hw_info.gmdid = dev_info->ip_version();
    hw_info.systolicAvailable = use_systolic_ukernel;

    if (hw_info.gmdid == 0) return status::unimplemented;

    memory_desc_wrapper src_mdw(src_md(0));
    memory_desc_wrapper wei_mdw(weights_md());
    memory_desc_wrapper dst_mdw(dst_md());

    auto convert_dnnl_to_kernel_layout = [](const memory_desc_t *md) {
        return (gemm_desc_t::get_trans(*md) == dnnl_trans) ? MatrixLayout::T
                                                           : MatrixLayout::N;
    };

    GEMMProblem problem;
    problem.Ta_ext = convert_dnnl_to_kernel_type(wei_mdw.data_type());
    problem.Tb_ext = convert_dnnl_to_kernel_type(src_mdw.data_type());
    problem.Tc_ext = problem.Ts = problem.Tc = Type::f32;

    problem.Ta = problem.Ta_ext;
    problem.Tb = problem.Tb_ext;

    problem.A.setAlignment(
            alignmentForLD(static_cast<int>(gemm_desc_t::get_ld(*wei_mdw.md_))
                    * problem.Ta_ext));
    problem.B.setAlignment(
            alignmentForLD(static_cast<int>(K()) * problem.Tb_ext));
    problem.C.setAlignment(problem.Tc.size());

    problem.A.layout = convert_dnnl_to_kernel_layout(wei_mdw.md_);
    problem.B.layout = MatrixLayout::N;
    problem.C.layout = MatrixLayout::N;

    GEMMOptions opts;
    opts.scaleA = wei_quant_.with_scale() && wei_group_sizes_[1] < K();
    opts.offsetA = wei_quant_.with_zp();
    opts.scaleB = src_quant_.with_scale() && src_group_sizes_[1] < K();
    opts.offsetB = src_quant_.with_zp();
    opts.slmPtr = true;
    opts.kParallelLocal = is_gemv_;

    if (opts.scaleA) {
        data_type_t wei_scale_dt = wei_quant_.scale_dt();
        problem.Ta_scale = convert_dnnl_to_kernel_type(wei_scale_dt);
        problem.A_scale.setAlignment(alignmentForLD(
                static_cast<int>(types::data_type_size(wei_scale_dt))));
        problem.A_scale.layout = MatrixLayout::N;
        problem.asPtrDims = 2;
    }

    if (opts.offsetA) {
        data_type_t wei_zp_dt = wei_quant_.zp_dt();
        problem.Tao = convert_dnnl_to_kernel_type(wei_zp_dt);
        problem.AO.setAlignment(
                static_cast<int>(types::data_type_size(wei_zp_dt)));
        problem.AO.layout = MatrixLayout::N;
        problem.aoPtrDims = 2;
        problem.aOffset = ABOffset::Calc;
    }

    if (opts.scaleB) {
        data_type_t src_scale_dt = src_quant_.scale_dt();
        problem.Tb_scale = convert_dnnl_to_kernel_type(src_scale_dt);
        problem.B_scale.setAlignment(
                static_cast<int>(types::data_type_size(src_scale_dt)));
        problem.bsPtrDims
                = (src_quant_.scale_mask() == (src_qmask_M() | src_qmask_K()))
                ? 2
                : 1;
        problem.B_scale.layout
                = problem.bsPtrDims > 1 ? MatrixLayout::N : MatrixLayout::T;
    }
    if (opts.offsetB) {
        data_type_t src_zp_dt = src_quant_.zp_dt();
        problem.Tbo = convert_dnnl_to_kernel_type(src_zp_dt);
        problem.BO.setAlignment(
                static_cast<int>(types::data_type_size(src_zp_dt)));
        problem.boPtrDims
                = (src_quant_.zp_mask() == (src_qmask_M() | src_qmask_K())) ? 2
                                                                            : 1;
        problem.bOffset = ABOffset::Calc;
        problem.BO.layout
                = problem.boPtrDims > 1 ? MatrixLayout::N : MatrixLayout::T;
    }

    if (opts.scaleA || opts.offsetA) {
        problem.aqGroupM = wei_group_sizes_[2];
        problem.aqGroupK = utils::rnd_up_pow2(wei_group_sizes_[1]);
    }

    if (opts.scaleB || opts.offsetB) {
        problem.bqGroupN = src_group_sizes_[0];
        problem.bqGroupK
                = static_cast<int>(utils::rnd_up_pow2(src_group_sizes_[1]));
    }

    // When both A and B are integers and group sums are needed, we
    // can avoid using group sums by converting one of the inputs to
    // f16/bf16.
    if (problem.Ta.isInteger() && problem.Tb.isInteger()
            && ((problem.needsAGroupSums() || problem.needsBGroupSums())
                    || ((opts.scaleA || opts.scaleB)
                            && dev_info->gpu_arch()
                                    < compute::gpu_arch_t::xe_hpc))) {
        Type ctype = Type::f16;
        if (utils::one_of(Type::bf16, problem.Ta_scale, problem.Tb_scale,
                    convert_dnnl_to_kernel_type(dst_mdw.data_type())))
            ctype = Type::bf16;
        if (problem.Ta_ext.bits() < problem.Tb_ext.bits()) {
            problem.Ta = ctype;
        } else {
            problem.Tb = ctype;
        }
    }

    SizeParams sizes;
    sizes.m = static_cast<uint16_t>(N());
    sizes.n = is_gemv_ ? 1 : 32;
    sizes.k = static_cast<uint16_t>(K());

    auto strat_override = [&](gemmstone::GEMMStrategy &strat) {
        std::string newStrat;
        using namespace gemmstone;
        strat.dpasw |= strat.fused;
        newStrat = gpu_utils::dev_getenv("GRPGEMM_USTRATEGY", newStrat);
        if (!newStrat.empty()) {
            // Example: 16 16 1 0 aT32 aM32 aB wg 2x4 sys
            printf("GRPGEMM_USTRATEGY: %s\n", newStrat.c_str());
            auto product = ngen::npack::decodeHWIPVersion(hw_info.gmdid);
            auto hw = getCore(product.family);
            auto stepping = hw_info.gmdid & 0xFF;
            strat = GEMMStrategy(hw, stepping);
            std::stringstream ss(newStrat);
            ss >> strat.unroll[0];
            ss >> strat.unroll[1];
            float a, b;
            ss >> a;
            ss >> b;
            Scalar alpha((int)a), beta((int)b);
            std::string strategyString;
            std::getline(ss >> std::ws, strategyString);
            parseStrategy(strategyString, hw, problem, strat);
            adjustStrategy(hw, problem, strat);
        } else if (dev_info->gpu_arch() == compute::gpu_arch_t::xe2
                && !is_gemv_
                && (float(M()) / ngroups_) >= 64.0f) {
            auto product = ngen::npack::decodeHWIPVersion(hw_info.gmdid);
            auto hw = getCore(product.family);
            auto stepping = hw_info.gmdid & 0xFF;
            strat = GEMMStrategy(hw, stepping);
            strat.unroll[0] = 32;
            strat.unroll[1] = 32;
            // Per-shape dispatch tuned via grid sweep on production MoE shapes.
            // sizes.m = matrix-N (col-major), sizes.k = matrix-K.
            //
            // EU-scaled heuristic: the right wg_m depends on whether we have
            // "enough" WGs to fill the GPU. Larger wg_m amortizes per-WG
            // overhead but produces fewer total WGs per launch. We pick wg_m
            // such that total dispatched WGs >= ~3x concurrent_wgs (3 waves
            // worth) so the scheduler has work to hide tail effects.
            //
            // For Xe2: hw_threads(large_grf=true) gives the count of
            // GRF256-mode threads (4 per EU). With wg_m subgroups of
            // SUBGROUP_SIZE=16 work-items per WG, concurrent_wgs is
            // hw_threads / wg_m. So wg_m=8 -> concurrent ~= eu_count/2,
            // wg_m=4 -> ~= eu_count, wg_m=2 -> ~= 2*eu_count.
            //
            // Reference points (BMG B60, 160 EUs, 456 GB/s, ~80 grf256 threads):
            //   TP8G1 (N=768,  K=2944): wg 8x1  - few N-tiles, large K, plenty of WGs from K-loop
            //   TP4G1 (N=1536, K=2944): wg 4x1  - balance
            //   TP4G2 (N=2880, K=768):  wg 2x1  - many N-tiles, large per-WG cost
            //   TP8G2 (N=2880, K=384):  wg 4x1  - many N-tiles, K too short for wg_m=2 efficiency
            //
            // Scaling: a 32-Xe-core variant (256 EUs, 608 GB/s) has 1.6x more
            // EUs but only 1.33x more BW, so per-EU BW drops 17%. To utilize
            // the extra EUs, prefer smaller wg_m (more concurrent WGs).
            // Threshold sizes.m for wg_m=2 selection scales inversely with
            // eu_count (smaller sizes.m can use wg_m=2 on bigger GPUs).
            const int eu_count = hw_info.euCount;
            // Reference: B60 has 160 EUs. Use ratio to scale thresholds.
            // (eu_ref / eu_count) > 1 means we have FEWER EUs (smaller GPU) -
            // prefer larger wg_m. < 1 means more EUs - prefer smaller wg_m.
            const float eu_scale = 160.0f / float(std::max(eu_count, 1));

            // Thresholds scaled by eu_scale (=160/eu_count, B60=1.0):
            //   small_n_thresh: below this, sizes.m is 'small' -> wg_m=8 OK
            //   large_n_thresh: above this, sizes.m is 'large' -> wg_m=2
            // On larger GPUs (smaller eu_scale), thresholds shrink so more
            // shapes land in the wg_m=2 bucket (more parallel WGs).
            const float small_n_thresh = 1024.0f * eu_scale; // 1024 @ B60
            const float large_n_thresh = 2500.0f * eu_scale; // 2500 @ B60
            // For 'big-enough' GPUs, also use wg_m=2 even at short K, since
            // we have enough EUs to absorb the extra concurrent WGs without
            // serialization. eu_count >= 200 (between B60 and 256-EU SKU).
            const bool big_gpu = eu_count >= 200;

            std::string body;
            if (sizes.m < small_n_thresh && sizes.k >= 1500) {
                // Small N + large K: wg 8x1
                // K-loop fills the latency pipeline; few N-tiles fit GPU.
                body = "aT64/0{cc} aM64/0{cc}+M64,64@128{cc} rB wg 8x1 sys xaf k1 grf256 vav di sr pk32 np";
            } else if (sizes.m >= large_n_thresh
                    && (sizes.k >= 512 || big_gpu)) {
                // Large N: wg 2x1.
                // - On B60 we require k>=512 (TP4G2) because at short K the
                //   pipeline tail dominates and wg_m=4 is better (TP8G2).
                // - On bigger GPUs we use wg_m=2 even at short K because
                //   more EUs give more concurrent WGs to hide tail effects.
                body = "aT64/0{cc} aM64/0{cc}+M64,64@128{cc} rB wg 2x1 sys xaf k1 grf256 vav di sr pk32 np";
            } else {
                // Default: wg 4x1 (balanced for medium N or short K on B60)
                body = "aT64/0{cc} aM64/0{cc}+M64,64@128{cc} rB wg 4x1 sys xaf k1 grf256 vav di sr pk32 np";
            }
            parseStrategy(body, hw, problem, strat);
            adjustStrategy(hw, problem, strat);
        }
        strategyGRFs_ = strat.GRFs;
        if (gpu_utils::dev_getenv("GRPGEMM_DUMP_STRATEGY", 0)) {
            auto product = ngen::npack::decodeHWIPVersion(hw_info.gmdid);
            auto hw = getCore(product.family);
            printf("GRPGEMM_STRATEGY_DUMP: unroll=%dx%d M=%d N=%d K=%d ngroups=%d str=\"%s\"\n",
                strat.unroll[0], strat.unroll[1],
                (int)sizes.m, (int)sizes.n, (int)sizes.k, (int)ngroups_,
                unparseStrategy(hw, problem, strat).c_str());
        }
    };

    try {
        gemm_ = selectGEMM(opts, hw_info, sizes, problem, {}, strat_override);
    } catch (const std::runtime_error &) {
        std::vector<StrategyRequirement> reqs;

        // TODO: These values should be based on the eu_count
        dim_t m_unroll = sg_size_;
        float avg_m = float(M()) / ngroups_;
        dim_t n_unroll = std::max<dim_t>(2, utils::rnd_up_pow2(dim_t(avg_m)));
        dim_t max_n_unroll = 0;
        dim_t max_wg_n = 4;
        dim_t min_wg_n = 1;

        switch (dev_info->gpu_arch()) {
            case compute::gpu_arch_t::xe_lp:
            case compute::gpu_arch_t::xe_hp:
                max_n_unroll = (problem.aqGroupK > 64
                                       && problem.A.layout == MatrixLayout::T)
                                || (problem.Ta_ext.isF8() && opts.scaleA
                                        && opts.scaleB)
                                || (problem.Ta_scale == Type::f8_e8m0
                                        || problem.Tb_scale == Type::f8_e8m0)
                        ? 8
                        : 16;
                break;
            case compute::gpu_arch_t::xe_hpg:
                if (!dev_info->mayiuse_systolic()) max_wg_n = 2;
                max_n_unroll = (problem.Ta_ext.bits() < 8)
                        ? sg_size_ * problem.Ta_ext
                        : 16;
                if (problem.Ta_ext.bits() <= 8) min_wg_n = 2;
                break;
            case compute::gpu_arch_t::xe_hpc: max_n_unroll = 32; break;
            case compute::gpu_arch_t::xe2: {
                m_unroll = 32;
                n_unroll = 32;
                max_n_unroll = 32;
                max_wg_n = 2;
                min_wg_n = 1;
                break;
            }
            default:
                m_unroll = sg_size_ / problem.Ta_ext;
                max_n_unroll
                        = problem.Ta.isInt4() ? sg_size_ * problem.Ta_ext : 32;
        }

        reqs.push_back(StrategyRequirement::UnrollM == m_unroll);
        reqs.push_back(StrategyRequirement::UnrollN
                == std::min(n_unroll, max_n_unroll));
        reqs.push_back(StrategyRequirement::WGM == 2);
        reqs.push_back(StrategyRequirement::WGN
                == utils::rnd_up_pow2(std::max(min_wg_n,
                        std::min((dim_t)(avg_m / reqs[1].value), max_wg_n))));
        try {
            gemm_ = selectGEMM(
                    opts, hw_info, sizes, problem, reqs, strat_override);
        } catch (const std::runtime_error &ex) {
            VDISPATCH_MATMUL_IC(false,
                    "gemm microkernel generation failure with message: %s",
                    ex.what());
        }
    }

    /* Generate microkernel shims */
    ShimOptions shimOptions;
    shimOptions.subgroupSize = sg_size_;
    shimOptions.useTileOps = true;
    shimOptions.decorator = "grouped";

    kernel_ctx_.define_int("SUBGROUP_SIZE", sg_size_);
    kernel_ctx_.add_custom_header("gemm_grouped.h",
            generateShim(gemm_, HostLanguage::OpenCL_C, shimOptions));

    return status::success;
}

template <size_t N>
void calc_group_sizes(std::array<int, N> &dims, const quant_entry_t &entry,
        const memory_desc_wrapper &desc) {
    memory_desc_t md;
    entry.get_md(md, *desc.md_);
    std::transform(desc.dims(), desc.dims() + dims.size(), md.dims, begin(dims),
            [](dim_t d, dim_t d2) -> int {
        return static_cast<int>(d2 == 0 ? 1 : d / d2);
    });
}

status_t grouped_micro_gemm_t::pd_t::init(impl::engine_t *engine) {
    using namespace data_type;

    memory_desc_wrapper src_d(src_md());
    memory_desc_wrapper wei_d(weights_md(0));
    memory_desc_wrapper dst_d(dst_md());

    data_type_t src_dt = src_d.data_type();
    data_type_t wei_dt = wei_d.data_type();
    data_type_t dst_dt = dst_d.data_type();
    src_quant_ = quantization_t(attr(), src_d, DNNL_ARG_SRC);
    wei_quant_ = quantization_t(attr(), wei_d, DNNL_ARG_WEIGHTS);

    // Check for grouped encoding on src and dst
    VDISPATCH_MATMUL(src_d.is_grouped_desc() && dst_d.is_grouped_desc(),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // Weights should be dense
    VDISPATCH_MATMUL(!wei_d.is_sparse_desc() && !wei_d.is_grouped_desc(),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // Extract grouped encoding
    const sparse_desc_t::grouped_desc_t &src_grouped
            = src_d.sparse_desc().grouped_desc;
    const sparse_desc_t::grouped_desc_t &dst_grouped
            = dst_d.sparse_desc().grouped_desc;

    VDISPATCH_MATMUL(wei_d.matches_one_of_tag(format_tag::ab, format_tag::ba,
                             format_tag::abc, format_tag::acb),
            VERBOSE_UNSUPPORTED_TAG_S, "weights");

    // Validate matching number of groups
    VDISPATCH_MATMUL(src_grouped.group_count == dst_grouped.group_count,
            VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "src ngroups", "dst ngroups",
            (int)src_grouped.group_count, (int)dst_grouped.group_count);

    ngroups_ = src_grouped.group_count;
    is_gemv_ = M() < ngroups_;

    // Token-centric dispatch (Phase 1.6): replaces expert-major dispatch
    // for MoE shapes with many experts. A precompute kernel writes
    // tile_starts[ngroups+1] once per call; the main GEMM dispatches
    // gws[2] = ceil(M/wg_tile_n) + ngroups upper-bound tile-WGs.
    //
    // Heuristic: enable when ngroups > 32 AND M/ngroups >= 8 (i.e., the
    // current expert-major dispatch wastes >>50% of WGs on early-exit).
    // Override via GRPGEMM_TOKEN_CENTRIC: 0=off, 1=on, default=auto.
    {
        int env_override = gpu_utils::dev_getenv("GRPGEMM_TOKEN_CENTRIC", -1);
        if (env_override == 1) {
            use_token_centric_ = !is_gemv_;
        } else if (env_override == 0) {
            use_token_centric_ = false;
        } else {
            use_token_centric_ = !is_gemv_ && ngroups_ > 32
                    && M() >= ngroups_ * 8;
        }
    }

    // only supported dt for now
    VDISPATCH_MATMUL(utils::one_of(src_dt, f32, f16, bf16, u8, s8, s4, u4,
                             f8_e5m2, f8_e4m3, e8m0, f4_e2m1, f4_e3m0),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(utils::one_of(wei_dt, f32, f16, bf16, u8, s8, s4, u4,
                             f8_e5m2, f8_e4m3, e8m0, f4_e2m1, f4_e3m0),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(
            utils::one_of(dst_dt, f32, f16, bf16), VERBOSE_UNSUPPORTED_DT_CFG);

    const bool src_subbyte = utils::one_of(src_dt, s4, u4);
    const bool wei_subbyte = utils::one_of(wei_dt, s4, u4);
    VDISPATCH_MATMUL(IMPLICATION(src_subbyte, (K() % 2) == 0), VERBOSE_BAD_DIM,
            "src", 1);
    VDISPATCH_MATMUL(IMPLICATION(wei_subbyte, (K() % 2) == 0), VERBOSE_BAD_DIM,
            "weights", 1);
    VDISPATCH_MATMUL(IMPLICATION(wei_subbyte, (N() % 2) == 0), VERBOSE_BAD_DIM,
            "weights", 2);

    // Check offsets are int32
    VDISPATCH_MATMUL(utils::everyone_is(s32, src_d.metadata_type(0),
                             dst_d.metadata_type(0)),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // Check for limited Bias support
    if (with_bias()) {
        memory_desc_wrapper bia_d(weights_md(1));
        VDISPATCH_MATMUL(!bia_d.is_sparse_desc() && !bia_d.is_grouped_desc(),
                VERBOSE_UNSUPPORTED_BIAS_CFG);
        VDISPATCH_MATMUL(bia_d.ndims() == 2, VERBOSE_UNSUPPORTED_BIAS_CFG);
        // Bias shape should be [num_experts, N]
        VDISPATCH_MATMUL(bia_d.dims()[0] == src_grouped.group_count,
                VERBOSE_INCONSISTENT_DIM, "bia_d", 0, "src_grouped.group_count",
                -1);
        VDISPATCH_MATMUL(bia_d.dims()[1] == wei_d.dims()[2],
                VERBOSE_INCONSISTENT_DIM, "bia_d", 1, "wei_d", 2);
    }

    assert(engine->kind() == engine_kind::gpu);
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    auto *dev_info = intel_engine->device_info();
    VDISPATCH_MATMUL(compute::mayiuse_microkernels(intel_engine),
            VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "microkernels");

    // Check for supported quantization schemes
    const scales_t &attr_scales = attr()->scales_;
    if (src_quant_.with_scale()) {
        VDISPATCH_MATMUL(utils::one_of(src_quant_.scale_dt(), f32, f16, bf16,
                                 f8_e5m2, f8_e4m3, e8m0, f4_e2m1, f4_e3m0),
                VERBOSE_UNSUPPORTED_SCALES_CFG ": src scales dt(%s)",
                dnnl_dt2str(src_quant_.scale_dt()));
    }

    if (src_quant_.with_zp()) {
        const int src_zp_mask = src_quant_.zp_mask();
        // Only per-row or per-column zero points supported for src
        VDISPATCH_MATMUL(utils::one_of(src_zp_mask, src_qmask_M(),
                                 src_qmask_M() | src_qmask_K(), 0),
                VERBOSE_UNSUPPORTED_ZP_CFG ": src zero points mask(%d)",
                src_zp_mask);
        VDISPATCH_MATMUL(utils::one_of(src_quant_.zp_dt(), u8, s8),
                VERBOSE_UNSUPPORTED_ZP_CFG ": src zero points dt(%s)",
                dnnl_dt2str(src_quant_.zp_dt()));
    }

    if (src_quant_.with_scale() && src_quant_.with_zp()) {
        const int src_scale_mask = src_quant_.scale_mask();
        const int src_zp_mask = src_quant_.zp_mask();
        VDISPATCH_MATMUL(src_scale_mask == src_zp_mask,
                VERBOSE_UNSUPPORTED_SCALES_CFG
                ": src scale(%d) and zp(%d) mask must match",
                src_scale_mask, src_zp_mask);
    }

    if (wei_quant_.with_scale()) {
        const int wei_mask = wei_quant_.scale_mask();
        VDISPATCH_MATMUL(
                utils::one_of(wei_mask, 7, 5), VERBOSE_UNSUPPORTED_SCALES_CFG);
        VDISPATCH_MATMUL(utils::one_of(wei_quant_.scale_dt(), f32, f16, bf16,
                                 f8_e5m2, f8_e4m3, e8m0, f4_e2m1, f4_e3m0),
                VERBOSE_UNSUPPORTED_SCALES_CFG ": wei scales dt(%s)",
                dnnl_dt2str(wei_quant_.scale_dt()));
    }

    if (wei_quant_.with_zp()) {
        const int wei_zp_mask = wei_quant_.zp_mask();
        // Only per-column zero points supported for weights
        VDISPATCH_MATMUL(utils::one_of(wei_zp_mask, 7, 5),
                VERBOSE_UNSUPPORTED_ZP_CFG ": wei zero points mask(%d)",
                wei_zp_mask);
        VDISPATCH_MATMUL(utils::one_of(wei_quant_.zp_dt(), u8, s8, u4, s4),
                VERBOSE_UNSUPPORTED_ZP_CFG ": wei zero points dt(%s)",
                dnnl_dt2str(wei_quant_.zp_dt()));
    }

    if (wei_quant_.with_scale() && wei_quant_.with_zp()) {
        const int wei_scale_mask = wei_quant_.scale_mask();
        const int wei_zp_mask = wei_quant_.zp_mask();
        VDISPATCH_MATMUL(wei_scale_mask == wei_zp_mask,
                VERBOSE_UNSUPPORTED_SCALES_CFG
                ": wei scale(%d) and zp(%d) mask must match",
                wei_scale_mask, wei_zp_mask);
    }

    VDISPATCH_MATMUL(attr_scales.has_default_values(DNNL_ARG_DST),
            VERBOSE_UNSUPPORTED_SCALES_CFG);

    // No post-ops for now
    VDISPATCH_MATMUL(
            attr()->post_ops_.has_default_values(), VERBOSE_UNSUPPORTED_POSTOP);

    if (src_quant_.with_scale()) {
        calc_group_sizes(
                src_group_sizes_, attr()->scales_.get(DNNL_ARG_SRC), src_d);
    } else if (src_quant_.with_zp()) {
        calc_group_sizes(src_group_sizes_,
                attr()->zero_points_.get(DNNL_ARG_SRC), src_d);
    }
    if (wei_quant_.with_scale()) {
        calc_group_sizes(
                wei_group_sizes_, attr()->scales_.get(DNNL_ARG_WEIGHTS), wei_d);
    } else if (wei_quant_.with_zp()) {
        calc_group_sizes(wei_group_sizes_,
                attr()->zero_points_.get(DNNL_ARG_WEIGHTS), wei_d);
    }
    sg_size_ = dev_info->min_subgroup_size();

    CHECK(init_microkernels(engine));

    src_quant_.define_macros(kernel_ctx_, "SRC");
    wei_quant_.define_macros(kernel_ctx_, "WEI");

    kernel_ctx_.set_data_type(dst_dt);

    if (gemm_.grfMin > 128 || strategyGRFs_ > 128)
        kernel_ctx_.add_option("-cl-intel-256-GRF-per-thread");

    def_data_type(kernel_ctx_, src_dt, "SRC");
    def_data_type(kernel_ctx_, wei_dt, "WEI");
    def_data_type(kernel_ctx_, dst_dt, "DST");

    kernel_ctx_.define_int("WITH_SRC_SCALES", src_quant_.with_scale());
    kernel_ctx_.define_int("WITH_WEI_SCALES", wei_quant_.with_scale());
    kernel_ctx_.define_int("WITH_SRC_ZP", src_quant_.with_zp());
    kernel_ctx_.define_int("WITH_WEI_ZP", wei_quant_.with_zp());
    if (src_quant_.with_scale() || src_quant_.with_zp()) {
        kernel_ctx_.define_int("SRC_GROUP_SIZE", src_group_sizes_[1]);
    }
    if (wei_quant_.with_scale() || wei_quant_.with_zp()) {
        kernel_ctx_.define_int("WEI_GROUP_SIZE", wei_group_sizes_[1]);
    }

    kernel_ctx_.define_int("SRC_SCALES_GROUPED",
            src_quant_.with_scale() && src_group_sizes_[1] < K());
    kernel_ctx_.define_int("WEI_SCALES_GROUPED",
            wei_quant_.with_scale() && wei_group_sizes_[1] < K());
    kernel_ctx_.define_int(
            "SRC_ELEMS_PER_BYTE", types::bytes_to_elements(src_dt, 1));
    kernel_ctx_.define_int(
            "WEI_ELEMS_PER_BYTE", types::bytes_to_elements(wei_dt, 1));

    if (src_quant_.with_zp()) {
        kernel_ctx_.define_int("SRC_ZP_ELEMS_PER_BYTE",
                types::bytes_to_elements(src_quant_.zp_dt(), 1));
    }
    if (wei_quant_.with_zp()) {
        kernel_ctx_.define_int("WEI_ZP_ELEMS_PER_BYTE",
                types::bytes_to_elements(wei_quant_.zp_dt(), 1));
    }

    auto bia_dt = weights_md(1)->data_type;
    def_data_type(kernel_ctx_, bia_dt, "BIA");
    kernel_ctx_.define_int("WITH_BIAS", with_bias());
    kernel_ctx_.define_int("K_PARALLEL_LOCAL", is_gemv_);
    // WITH_SPARSE_GROUPS: token-flat dispatch (gws[2]=m_all), used by GEMV.
    // WITH_TILE_INDEX: per-expert tile dispatch (gws[2]=total_tiles upper
    //   bound), used by token-centric mode for non-GEMV. Each WG computes
    //   tile_starts[] in SLM via single-WI prefix sum, then binary searches
    //   to find owning expert.
    kernel_ctx_.define_int("WITH_SPARSE_GROUPS", is_gemv_);
    kernel_ctx_.define_int("WITH_TILE_INDEX", use_token_centric_);
    kernel_ctx_.define_int("WITH_SLM", gemm_.getSetting("slm_size") > 0);
    kernel_ctx_.define_int("NUM_GROUPS", ngroups_);
    // WG_TILE_N is the M-direction tile size used by the main GEMM.
    // Needed by the precompute kernel which doesn't include gemm_grouped.h.
    kernel_ctx_.define_int("WG_TILE_N", gemm_.getSetting("wg_tile_n"));
    kernel_ctx_.add_option("-cl-std=CL3.0");

    // Reserve scratchpad for tile_starts buffer when WITH_TILE_INDEX is on.
    // Layout: int32[ngroups_ + 1]; index ngroups_ holds total_tiles count.
    if (use_token_centric_) {
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_matmul_pack_space,
                ngroups_ + 1, sizeof(int32_t), OCL_BUFFER_ALIGNMENT);
    }

    return status::success;
}

status_t grouped_micro_gemm_t::init(impl::engine_t *engine) {
    // Create both kernels - main GEMM and the precompute kernel for
    // tile_starts. Precompute is only dispatched when use_token_centric_.
    std::vector<compute::kernel_t> kernels;
    static const std::vector<const char *> kernel_names
            = {"grouped_micro_gemm", "grouped_compute_tile_starts"};
    CHECK(create_kernels(engine, &kernels, kernel_names, pd()->kernel_ctx_));
    kernel_ = kernels[0];
    precompute_kernel_ = kernels[1];
    return status::success;
}

status_t grouped_micro_gemm_t::execute(const exec_ctx_t &ctx) const {
    // buffer 0: values, buffer 1: offsets
    const auto &src_data = CTX_IN_STORAGE(DNNL_ARG_SRC, 0);
    const auto &src_offsets = CTX_IN_STORAGE(DNNL_ARG_SRC, 1);
    const auto &wei_data = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &dst_data = CTX_OUT_STORAGE(DNNL_ARG_DST, 0);
    const auto &dst_offsets = CTX_OUT_STORAGE(DNNL_ARG_DST, 1);

    const auto &src_scales = pd()->src_quant_.scales(ctx);
    const auto &src_zero_points = pd()->src_quant_.zero_points(ctx);
    const auto &wei_scales = pd()->wei_quant_.scales(ctx);
    const auto &wei_zero_points = pd()->wei_quant_.zero_points(ctx);

    const auto &bias_data = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    const memory_desc_t *src_md = ctx.input(DNNL_ARG_SRC)->md();
    const memory_desc_t *wei_md = pd()->weights_md();
    const memory_desc_t *dst_md = ctx.output(DNNL_ARG_DST)->md();

    const bool with_src_scales = pd()->src_quant_.with_scale();
    const bool with_src_zero_points = pd()->src_quant_.with_zp();
    const bool with_wei_scales = pd()->wei_quant_.with_scale();
    const bool with_wei_zero_points = pd()->wei_quant_.with_zp();

    dim_t ldsrcq = 0;
    dim_t ldweiq = 0;

    if (with_src_scales || with_src_zero_points) {
        const memory_desc_t *src_quant_md = with_src_scales
                ? ctx.input(DNNL_ARG_SRC | DNNL_ARG_ATTR_SCALES)->md()
                : ctx.input(DNNL_ARG_SRC | DNNL_ARG_ATTR_ZERO_POINTS)->md();
        ldsrcq = static_cast<int>(
                src_quant_md->format_desc.blocking.strides[0]);
    }
    if (with_wei_scales || with_wei_zero_points) {
        const memory_desc_t *wei_quant_md = with_wei_scales
                ? ctx.input(DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_SCALES)->md()
                : ctx.input(DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_ZERO_POINTS)->md();
        ldweiq = static_cast<int>(
                wei_quant_md->format_desc.blocking.strides[1]);
    }
    dim_t m_all = dst_md->dims[dst_md->ndims - 2];
    dim_t n = dst_md->dims[dst_md->ndims - 1];
    dim_t k = src_md->dims[src_md->ndims - 1];

    dim_t ldsrc = src_md->dims[src_md->ndims - 1];
    dim_t lddst = dst_md->dims[dst_md->ndims - 1];
    const dims_t &wei_strides_ = wei_md->format_desc.blocking.strides;
    compute::int64x4_t wei_strides
            = {static_cast<int64_t>(wei_strides_[wei_md->ndims - 3]),
                    static_cast<int64_t>(wei_strides_[wei_md->ndims - 2]),
                    static_cast<int64_t>(wei_strides_[wei_md->ndims - 1]),
                    static_cast<int64_t>(wei_strides_[wei_md->ndims - 0])};

    compute::kernel_arg_list_t arg_list;
    arg_list.append(src_data);
    arg_list.append(ldsrc);
    arg_list.append(wei_data);
    arg_list.append(wei_strides);
    arg_list.append(dst_data);
    arg_list.append(lddst);
    arg_list.append(src_offsets);
    arg_list.append(dst_offsets);
    arg_list.append(src_scales);
    arg_list.append(src_zero_points);
    arg_list.append(ldsrcq);
    arg_list.append(wei_scales);
    arg_list.append(wei_zero_points);
    arg_list.append(ldweiq);
    arg_list.append(n);
    arg_list.append(k);

    arg_list.append(bias_data);

    // For WITH_TILE_INDEX mode: launch precompute kernel for tile_starts[]
    // before the main GEMM. The scratchpad buffer holds [NUM_GROUPS+1] int32.
    // When token-centric is off, pass src_offsets as a dummy (kernel won't
    // dereference tile_starts in that case).
    std::unique_ptr<memory_storage_t> tile_starts_storage;
    if (pd()->use_token_centric_) {
        tile_starts_storage
                = ctx.get_scratchpad_grantor().get_memory_storage(
                        memory_tracking::names::key_matmul_pack_space);

        compute::kernel_arg_list_t pre_args;
        pre_args.append(src_offsets);
        pre_args.append(*tile_starts_storage);

        // Single WG of 1 work-item (kernel uses sequential scan, ngroups~128).
        compute::range_t pre_lws = compute::range_t {1, 1, 1};
        compute::range_t pre_gws = compute::range_t {(size_t)pd()->sg_size_, 1, 1};
        // Subgroup-size attribute requires gws[0] >= sg_size; kernel itself
        // only uses WI 0.
        CHECK(parallel_for(ctx, compute::nd_range_t(pre_gws, pre_lws),
                precompute_kernel_, pre_args));
        arg_list.append(*tile_starts_storage);
    } else {
        // Pass src_offsets as a placeholder; main kernel won't read it
        // unless WITH_TILE_INDEX (which is gated on use_token_centric_).
        arg_list.append(src_offsets);
    }

    size_t sg_per_wg_m = pd()->gemm_.getSetting("sg_per_wg_m");
    size_t sg_per_wg_n = pd()->gemm_.getSetting("sg_per_wg_n");
    size_t sg_per_wg_k = pd()->gemm_.getSetting("sg_per_wg_k");
    size_t wg_tile_m = pd()->gemm_.getSetting("wg_tile_m");
    size_t wg_tile_n = pd()->gemm_.getSetting("wg_tile_n");

    // Use total_tokens as upper bound for M dimension
    compute::range_t lws = compute::range_t::one(3);
    lws[0] *= pd()->sg_size_;

    lws[0] *= sg_per_wg_m;
    lws[1] *= sg_per_wg_n;
    lws[2] *= sg_per_wg_k;

    dim_t m_dispatch = m_all;
    const int32_t *max_var_dim
            = CTX_IN_MEM(const int32_t *, DNNL_ARG_HINT_MAX_GROUP_SIZE);
    if (max_var_dim && *max_var_dim > 0 && *max_var_dim <= m_all)
        m_dispatch = *max_var_dim;

    compute::range_t gws = lws;
    // Swap wg_tile_[mn]_ for col-major vs row-major representations
    gws[0] *= utils::div_up(n, wg_tile_m);
    // Dispatch shape:
    //   GEMV: gws[2]=m_all, one WG per flat_token
    //   token-centric: gws[2]=ceil(m_all/wg_tile_n)+ngroups (upper bound),
    //     gws[1]=1, each WG owns one (expert, tile_within_expert)
    //   default: gws[2]=ngroups, gws[1]=ceil(m_dispatch/wg_tile_n)
    if (pd()->is_gemv_) {
        gws[1] *= 1;
        gws[2] *= m_all;
    } else if (pd()->use_token_centric_) {
        // Upper bound: each expert contributes at most ceil(t/wg_tile_n)
        // tiles, total <= m_all/wg_tile_n + ngroups (boundary slack).
        const dim_t total_tiles_upper
                = utils::div_up(m_all, wg_tile_n) + pd()->ngroups_;
        gws[1] *= 1;
        gws[2] *= total_tiles_upper;
    } else {
        gws[1] *= utils::div_up(m_dispatch, wg_tile_n);
        gws[2] *= pd()->ngroups_;
    }

    return parallel_for(ctx, compute::nd_range_t(gws, lws), kernel_, arg_list);
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY

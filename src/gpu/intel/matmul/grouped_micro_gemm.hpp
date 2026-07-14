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

#ifndef GPU_INTEL_MATMUL_GROUPED_MICRO_GEMM_HPP
#define GPU_INTEL_MATMUL_GROUPED_MICRO_GEMM_HPP

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gemmstone/microkernel/package.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/matmul/config.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

struct grouped_micro_params_t
    : trivially_serializable_t<grouped_micro_params_t> {

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> kernel_names
                = {"grouped_micro_gemm"};
        return kernel_names;
    }

    status_t create_generator(const intel::engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        compute::kernel_ctx_t kernel_ctx;
        CHECK(get_kernel_ctx(kernel_ctx));
        auto status = engine.create_kernel_bundle(
                bundle, get_kernel_names(), kernel_ctx);
        return status;
    }

    status_t get_kernel_ctx(compute::kernel_ctx_t &) const;
};

struct grouped_micro_gemm_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public matmul::pd_t {
        using matmul::pd_t::pd_t;

        DECLARE_COMMON_PD_T("grouped_gemm:micro", grouped_micro_gemm_t);

        status_t init(impl::engine_t *engine);
        status_t init_microkernels(impl::engine_t *engine);

        bool is_gemv_ = false;
        bool k_parallel_local_ = false;
        bool use_active_tile_list_ = false;
        int sg_size_ = 0;
        int strategyGRFs_ = 0;
        dim_t ngroups_ = 0;
        std::array<int, 2> src_group_sizes_ = {0, 0};
        std::array<int, 3> wei_group_sizes_ = {0, 0, 0};
        quantization_t src_quant_;
        quantization_t wei_quant_;
        gemmstone::microkernel::Package gemm_;
        compute::kernel_ctx_t kernel_ctx_;
    };
    status_t init(impl::engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;
    status_t execute_fast(const exec_ctx_t &ctx) const override {
        return execute_fast_cached(ctx);
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
    compute::kernel_t precompute_kernel_;

    struct dispatch_t {
        compute::kernel_arg_list_t args;
        compute::range_t lws = compute::range_t::one(3);
        compute::range_t gws = compute::range_t::one(3);
        size_t wg_tile_n = 0;
        dim_t m_all = 0;
        std::unique_ptr<memory_storage_t> tile_starts;
    };

    struct fast_cache_t {
        bool populated = false;
        dispatch_t dispatch;

#ifndef __SYCL_DEVICE_ONLY__
        void *l0_kernel = nullptr;
        void *l0_cmdlist = nullptr;
        struct l0_ptr_arg_t {
            int idx;
            void *ptr;
        };
        std::vector<l0_ptr_arg_t> l0_ptr_args;
        bool l0_ready = false;
#endif
    };
    mutable fast_cache_t fast_cache_;

    status_t prepare_dispatch(
            const exec_ctx_t &ctx, dispatch_t &dispatch) const;
    compute::range_t resolve_dispatch_range(
            const exec_ctx_t &ctx, const dispatch_t &dispatch) const;
    status_t populate_fast_cache(const exec_ctx_t &ctx) const;
    status_t execute_fast_cached(const exec_ctx_t &ctx) const;
};

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
#endif // GPU_INTEL_MATMUL_GROUPED_MICRO_GEMM_HPP

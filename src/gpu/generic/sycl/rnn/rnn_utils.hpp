/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_RNN_RNN_UTILS_HPP
#define GPU_GENERIC_SYCL_RNN_RNN_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive_desc.hpp"
#include "common/stream.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/gpu_rnn_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

template <int N>
inline int calc_offset(std::array<dim_t, N> Ds, std::array<dim_t, N + 1> Is) {
    int offset = Is[0];
#pragma unroll
    for (int i = 0; i < N; i++) {
        offset = offset * Ds[i] + Is[i + 1];
    }
    return offset;
}

using strides_t = std::array<dim_t, 4>;

namespace rnn_utils {

enum ws_part_t { gates, states, cell, grid, bias };

namespace kernel_id {
constexpr size_t copy_init_layer = 0;
constexpr size_t copy_init_iter = 1;
constexpr size_t copy_res_layer = 2;
constexpr size_t copy_res_iter = 3;
constexpr size_t bias_fwd = 4;
constexpr size_t cell_fwd = 5;
} // namespace kernel_id

struct conf_t {
    dim_t n_layer, n_iter, n_dir, n_gates, n_states;
    dim_t mb;
    dim_t slc, sic, dhc, dlc;

    dim_t gates_ld, gates_ws_ld;

    dim_t n_bias, n_parts_bias, parts_bias[DNNL_RNN_MAX_N_PARTS];

    dim_t iter_loop;

    dim_t states_ws_ld, scratch_diff_states_ld;
    bool is_fwd, is_training;
    bool use_workspace;

    // Size of workspace for each tensor in bytes
    dim_t ws_states_cell_size, ws_gates_cell_size;
    dim_t ws_gates_size, ws_states_size, scratch_cell_size, ws_per_cell,
            scratch_diff_states_size, ws_bias_size;

    dim_t ws_gates_offset;
    dim_t ws_states_offset;
    dim_t ws_bias_offset;

    // Element size of each workspace part in bytes
    dim_t ws_gates_elsz, ws_states_elsz, ws_bias_elsz;

    dim_t n_iter_scratch_gates;
    dim_t scratch_gates_size, scratch_gates_elsz, scratch_gates_ld;
    dim_t scratch_diff_gates_size, scratch_diff_gates_elsz,
            scratch_diff_gates_ld;
    dims_t local_ranges;

    data_type_t acc_data_type;
    data_type_t aux_data_type;
    data_type_t input_data_type;
    data_type_t output_data_type;
    data_type_t ws_data_type;
    data_type_t src_data_type;
    data_type_t dst_data_type;
    data_type_t diff_data_type;
    data_type_t wei_layer_type;
    data_type_t wei_iter_type;
    data_type_t bias_data_type;
};

dim_t get_good_ld(
        dim_t arch_ld, dim_t dim, dim_t sizeof_dt, bool ignore_assoc = false);
void init_rnn_conf(
        conf_t &rnn, const rnn_pd_t *rnn_pd, data_type_t acc_data_type);
void set_rnn_conf(conf_t &rnn, const rnn_desc_t &rd);
dim_t set_workspace_offsets(
        const conf_t &rnn, dim_t &ws_gates_offset, dim_t &ws_h_state_offset);
dim_t get_workspace_size(const conf_t &rnn);
status_t set_weights_desc(memory_desc_t &weights_md, const conf_t &rnn);
status_t set_good_strides(memory_desc_t &weights_md, format_tag_t tag);
const memory_storage_t &get_storage(const memory_storage_t *storage);
const memory_storage_t &get_storage(
        const std::unique_ptr<memory_storage_t> &storage);

struct data_helper_t {
    static dim_t type_size(data_type_t d) {
        return static_cast<dim_t>(types::data_type_size(d));
    }
};

struct user_data_t : public data_helper_t {
    using mst = memory_storage_t;
    user_data_t()
        : wei_layer_(nullptr)
        , wei_layer_mdw_(memory_desc_t())
        , diff_wei_layer_(nullptr)
        , diff_wei_layer_mdw_(memory_desc_t())
        , wei_iter_(nullptr)
        , wei_iter_mdw_(memory_desc_t())
        , diff_wei_iter_(nullptr)
        , diff_wei_iter_mdw_(memory_desc_t())
        , bias_(nullptr)
        , bias_mdw_(memory_desc_t())
        , diff_bias_(nullptr)
        , diff_bias_mdw_(memory_desc_t()) {}
    user_data_t(mst &wei_layer, const memory_desc_t *wei_layer_mdw,
            mst &diff_wei_layer, const memory_desc_t *diff_wei_layer_mdw,
            mst &wei_iter, const memory_desc_t *wei_iter_mdw,
            mst &diff_wei_iter, const memory_desc_t *diff_wei_iter_mdw,
            mst &bias, const memory_desc_t *bias_mdw, mst &diff_bias,
            const memory_desc_t *diff_bias_mdw)
        : wei_layer_(&wei_layer)
        , wei_layer_mdw_(wei_layer_mdw)
        , diff_wei_layer_(&diff_wei_layer)
        , diff_wei_layer_mdw_(diff_wei_layer_mdw)
        , wei_iter_(&wei_iter)
        , wei_iter_mdw_(wei_iter_mdw)
        , diff_wei_iter_(&diff_wei_iter)
        , diff_wei_iter_mdw_(diff_wei_iter_mdw)
        , bias_(&bias)
        , bias_mdw_(bias_mdw)
        , diff_bias_(&diff_bias)
        , diff_bias_mdw_(diff_bias_mdw) {}

    const mst *wei_layer() const { return wei_layer_; }
    std::unique_ptr<mst> wei_layer(dim_t lay, dim_t dir) const {

        dim_t t = type_size(wei_layer_mdw_.data_type());
        // wei_layer dimension order: layer, dir, src c, gate, dst c
        dim_t offset = wei_layer_mdw_.off(lay, dir, 0, 0, 0) * t;

        return wei_layer_->clone_ptr_off(offset);
    }

    const mst *wei_iter() const { return wei_iter_; }
    std::unique_ptr<mst> wei_iter(dim_t lay, dim_t dir) const {
        dim_t t = type_size(wei_iter_mdw_.data_type());
        // wei_iter dimension order: layer, dir, src c, gate, dst c
        dim_t offset = wei_iter_mdw_.off(lay, dir, 0, 0, 0) * t;

        return wei_iter_->clone_ptr_off(offset);
    }

    const mst *bias() const { return bias_; }

    std::unique_ptr<mst> bias(dim_t lay, dim_t dir) const {
        if (bias()->data_handle() == nullptr) return {};
        auto t = type_size(bias_mdw_.data_type());
        // bia dimension order: lay, dir, gates, dhc
        auto offset = bias_mdw_.off(lay, dir, 0, 0) * t;

        return bias_->clone_ptr_off(offset);
    }

    const mst *diff_bias() const { return diff_bias_; }

    std::unique_ptr<mst> diff_bias(dim_t lay, dim_t dir) const {
        if (bias()->data_handle() == nullptr) return {};
        auto t = type_size(diff_bias_mdw_.data_type());
        // bia dimension order: lay, dir, gates, dhc
        auto offset = diff_bias_mdw_.off(lay, dir, 0, 0) * t;

        return diff_bias_->clone_ptr_off(offset);
    }

    const mst *diff_wei_layer() const { return diff_wei_layer_; }
    std::unique_ptr<mst> diff_wei_layer(dim_t lay, dim_t dir) const {

        // diff_wei_layer dimension order: layer, dir, src c, gate, dst c
        dim_t t = sizeof(float);
        dim_t offset = diff_wei_layer_mdw_.off(lay, dir, 0, 0, 0) * t;

        return diff_wei_layer_->clone_ptr_off(offset);
    }

    const mst *diff_wei_iter() const { return diff_wei_iter_; }
    std::unique_ptr<mst> diff_wei_iter(dim_t lay, dim_t dir) const {
        // diff_wei_iter dimension order: layer, dir, src c, gate, dst c
        dim_t t = sizeof(float);
        dim_t offset = diff_wei_iter_mdw_.off(lay, dir, 0, 0, 0) * t;

        return diff_wei_iter_->clone_ptr_off(offset);
    }

    mst *wei_layer_;
    memory_desc_wrapper wei_layer_mdw_;
    mst *diff_wei_layer_;
    memory_desc_wrapper diff_wei_layer_mdw_;
    mst *wei_iter_;
    memory_desc_wrapper wei_iter_mdw_;
    mst *diff_wei_iter_;
    memory_desc_wrapper diff_wei_iter_mdw_;
    mst *bias_;
    memory_desc_wrapper bias_mdw_;
    mst *diff_bias_;
    memory_desc_wrapper diff_bias_mdw_;
};

struct workspace_t : public data_helper_t {
    using mst = memory_storage_t;
    workspace_t(const mst &ws, const conf_t &conf)
        : ws_(ws)
        , conf_(conf)
        , gates_(conf.ws_gates_size > 0 ? ws.clone() : nullptr)
        , gates_strides_ {0}
        , states_(conf.ws_states_size > 0 ? ws.clone() : nullptr)
        , states_strides_ {0}
        , bias_(conf.ws_bias_size > 0 ? ws.clone() : nullptr) {
        if (gates_) {
            gates_->set_offset(gates_->offset() + conf.ws_gates_offset);
            const int n_b = conf_.mb;
            const int n_tb = (conf_.n_iter + 1) * n_b;
            const int n_dtb = conf_.n_dir * n_tb;
            gates_strides_
                    = {n_dtb * conf_.gates_ws_ld, n_tb * conf_.gates_ws_ld,
                            n_b * conf_.gates_ws_ld, conf_.gates_ws_ld};
        }
        if (states_) {
            states_->set_offset(states_->offset() + conf.ws_states_offset);
            const int n_b = conf_.mb;
            const int n_tb = (conf_.n_iter + 1) * n_b;
            const int n_dtb = conf_.n_dir * n_tb;
            states_strides_ = {n_dtb * conf_.states_ws_ld,
                    n_tb * conf_.states_ws_ld, n_b * conf_.states_ws_ld, 1};
        }
        bias_->set_offset(bias_->offset() + conf.ws_bias_offset);
    }

    static dim_t get_offset(
            const strides_t &strides, const std::array<dim_t, 4> &dims) {
        dim_t offset = 0;
        for (size_t i = 0; i < 4; i++) {
            offset += strides[i] * dims[i];
        }
        return offset;
    }

    dim_t calc_off_ws_state(
            dim_t i0, dim_t i1, dim_t i2, dim_t i3, dim_t i4) const {
        assert(i0 >= 0);
        //lay,dir,time
        return calc_offset<4>(
                {conf_.n_dir, conf_.n_iter + 1, conf_.mb, conf_.states_ws_ld},
                {i0, i1, i2, i3, i4});
    }

    const mst &ws() const { return ws_; }
    const mst &gates() const { return get_storage(gates_); }
    const mst &states() const { return get_storage(states_); }

    std::unique_ptr<mst> states(dim_t layer, dim_t dir, dim_t time) const {
        if (!states_) return {};
        auto off_ = get_offset(states_strides(), {layer, dir, time, 0})
                * conf_.ws_states_elsz;
        return states().clone_ptr_off(off_);
    }

    const strides_t &states_strides() const { return states_strides_; }

    std::unique_ptr<mst> states_range(dim_t layer_start, dim_t layer_end,
            dim_t dir_start, dim_t dir_end, dim_t time_start,
            dim_t time_end) const {
        auto off_start
                = calc_off_ws_state(layer_start, dir_start, time_start, 0, 0)
                * conf_.ws_states_elsz;
        return states().clone_ptr_off(off_start);
    }

    std::unique_ptr<mst> gates(
            dim_t layer, dim_t dir, dim_t time, dim_t mb = 0) const {
        auto off = get_offset(gates_strides(), {layer, dir, time, mb})
                * conf_.ws_gates_elsz;
        return gates().clone_ptr_off(off);
    }
    const strides_t &gates_strides() const { return gates_strides_; }

    const mst &bias() const { return get_storage(bias_); }

private:
    const mst &ws_;
    const conf_t &conf_;
    std::unique_ptr<mst> gates_;
    strides_t gates_strides_;
    std::unique_ptr<mst> states_;
    strides_t states_strides_;
    std::unique_ptr<mst> bias_;
    std::unique_ptr<mst> grid_comp_;
};

struct scratch_t : public data_helper_t {
    using mst = memory_storage_t;

    enum {
        key_gemm_iter_fwd = memory_tracking::names::key_nested_multiple,
        key_gemm_layer_fwd,
        key_gemm_iter_bwd,
        key_gemm_layer_bwd,
        key_gemm_diff_wei_layer,
        key_gemm_diff_wei_iter
    };

    scratch_t(const conf_t &conf, const memory_tracking::grantor_t &scratchpad)
        : conf_(conf) {
        using namespace memory_tracking::names;
        gates_ = scratchpad.get_memory_storage(key_rnn_gates);
        diff_gates_ = scratchpad.get_memory_storage(key_rnn_diff_gates);
        cell_ = scratchpad.get_memory_storage(key_rnn_cell);
        diff_states_ = scratchpad.get_memory_storage(key_rnn_diff_states);
    }

    struct fwd_matmul_pds {
        const primitive_desc_t *iter_fwd_pd;
        const primitive_desc_t *layer_fwd_pd;
    };
    struct bwd_matmul_pds {
        const primitive_desc_t *iter_bwd_pd;
        const primitive_desc_t *layer_bwd_pd;
        const primitive_desc_t *diff_wei_layer_pd;
        const primitive_desc_t *diff_wei_iter_pd;
    };

    static void book_fwd(memory_tracking::registrar_t &scratchpad,
            const conf_t &rnn_conf, const fwd_matmul_pds &matmuls) {
        using namespace memory_tracking::names;
        if (rnn_conf.scratch_gates_size > 0)
            scratchpad.book(key_rnn_gates, rnn_conf.scratch_gates_size, 1);
        scratchpad.book(key_rnn_cell, rnn_conf.scratch_cell_size, 1);
        // book scratchpad for nested primitives
        if (matmuls.layer_fwd_pd) {
            scratchpad.book(key_gemm_layer_fwd,
                    matmuls.layer_fwd_pd->scratchpad_registry());
        }
        if (matmuls.iter_fwd_pd) {
            scratchpad.book(key_gemm_iter_fwd,
                    matmuls.iter_fwd_pd->scratchpad_registry());
        }
    }

    static void book_bwd(memory_tracking::registrar_t &scratchpad,
            const conf_t &rnn_conf, const bwd_matmul_pds &matmuls) {
        using namespace memory_tracking::names;
        if (rnn_conf.scratch_gates_size > 0)
            scratchpad.book(key_rnn_gates, rnn_conf.scratch_gates_size, 1);
        scratchpad.book(key_rnn_cell, rnn_conf.scratch_cell_size, 1);
        scratchpad.book(
                key_rnn_diff_states, rnn_conf.scratch_diff_states_size, 1);
        // book scratchpad for nested primitives
        if (!rnn_conf.is_fwd) {
            scratchpad.book(
                    key_rnn_diff_gates, rnn_conf.scratch_diff_gates_size, 1);
            scratchpad.book(key_gemm_iter_bwd,
                    matmuls.iter_bwd_pd->scratchpad_registry());
            scratchpad.book(key_gemm_layer_bwd,
                    matmuls.layer_bwd_pd->scratchpad_registry());
            scratchpad.book(key_gemm_diff_wei_layer,
                    matmuls.diff_wei_layer_pd->scratchpad_registry());
            scratchpad.book(key_gemm_diff_wei_iter,
                    matmuls.diff_wei_iter_pd->scratchpad_registry());
        }
    }

    dim_t calc_off_gates(dim_t iter) const {
        return conf_.n_iter_scratch_gates != 1
                ? iter * conf_.mb * conf_.scratch_gates_ld * conf_.ws_gates_elsz
                : 0;
    };

    const mst *gates() const {
        assert(gates_);
        return gates_.get();
    }
    std::unique_ptr<mst> gates(dim_t iter) const {
        auto g = gates();
        if (g == nullptr) return {};

        auto off = calc_off_gates(iter);
        return g->clone_ptr_off(off);
    }

    dim_t calc_off_diff_gates(dim_t iter) const {
        return conf_.n_iter_scratch_gates != 1
                ? iter * conf_.mb * conf_.scratch_diff_gates_ld
                : 0;
    };

    const mst *diff_gates() const { return (diff_gates_.get()); }

    std::unique_ptr<mst> diff_gates(dim_t iter) const {
        auto g = gates();
        if (g == nullptr) return {};

        auto off = calc_off_diff_gates(iter) * conf_.scratch_diff_gates_elsz;
        return g->clone_ptr_off(off);
    }

    const mst *cell() const { return cell_.get(); }

    dim_t calc_off_diff_state(
            dim_t i0, dim_t i1, dim_t i2, dim_t i3, dim_t i4, dim_t i5) const {

        return calc_offset<5>(
                {conf_.n_dir, conf_.n_iter + 1, conf_.n_states + 1, conf_.mb,
                        conf_.scratch_diff_states_ld},
                {i0, i1, i2, i3, i4, i5});
    }

    const mst &diff_states() const { return get_storage(diff_states_); }

    std::unique_ptr<mst> diff_states(
            dim_t layer, dim_t dir, dim_t iter, dim_t state = 0) const {
        int aux_elsz = type_size(conf_.aux_data_type);

        if (!diff_states_) return {};
        auto off_
                = calc_off_diff_state(layer, dir, iter, state, 0, 0) * aux_elsz;
        return diff_states().clone_ptr_off(off_);
    }

private:
    const conf_t &conf_;

    std::unique_ptr<mst> gates_;
    std::unique_ptr<mst> diff_gates_;
    std::unique_ptr<mst> cell_;
    std::unique_ptr<mst> diff_states_;
};

inline size_t calc_global_range(const size_t lc_range, size_t gl_range) {
    return ((gl_range + (lc_range - 1)) / lc_range) * lc_range;
}

inline size_t calc_local_range(const exec_ctx_t &ctx) {
    // Check the device for the supported max worgroup size
    // TODO: 256 is an arbitrary ceiling to ensure we do not use too
    // many registers, can be improved in future.
    return std::floor(std::cbrt(std::min<size_t>(256,
            static_cast<xpu::sycl::stream_impl_t *>(ctx.stream()->impl())
                    ->queue()
                    ->get_device()
                    .get_info<::sycl::info::device::max_work_group_size>())));
}

inline void get_outer_strides(const memory_desc_wrapper &md, dims_t &ret) {
    for (int d = 4; d >= 0; d--) {
        if (d >= md.ndims()) {
            ret[d] = 0;
        } else if (md.padded_dims()[d] > 1) {
            ret[d] = md.strides()[d];
        } else if (d == md.ndims() - 1) {
            ret[d] = static_cast<dim_t>(1);
        } else {
            ret[d] = ret[d + 1] * md.padded_dims()[d + 1];
        }
    }
}

} // namespace rnn_utils

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

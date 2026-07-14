/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dnnl/dnnl_sycl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/primitive_desc_iface.hpp"
#include "common/primitive_iface.hpp"
#include "common/utils.hpp"

#include "xpu/sycl/engine_factory.hpp"
#include "xpu/sycl/stream_impl.hpp"

using dnnl::impl::status_t;
using dnnl::impl::stream_t;

status_t dnnl_sycl_interop_primitive_execute(
        const primitive_iface_t *primitive_iface, stream_t *stream, int nargs,
        const dnnl_exec_arg_t *args, const void *deps_, void *return_event_) {
    using namespace dnnl::impl;
    bool ok = !utils::any_null(primitive_iface, stream)
            && primitive_iface->engine() == stream->engine()
            && primitive_iface->engine()->runtime_kind() == runtime_kind::sycl
            && IMPLICATION(nargs > 0, args != nullptr);
    if (!ok) return status::invalid_arguments;

    auto *sycl_stream_impl
            = utils::downcast<dnnl::impl::xpu::sycl::stream_impl_t *>(
                    stream->impl());

    // Check arguments.
    exec_args_t exec_args;
    CHECK(cvt_primitive_args(
            primitive_iface->pd()->impl().get(), nargs, args, exec_args));

    // Note: there should be no fast exit between hooks.
    stream->before_exec_hook();

    if (deps_ != nullptr) {
        auto deps = dnnl::impl::xpu::sycl::event_t(
                *(const std::vector<::sycl::event> *)deps_);
        sycl_stream_impl->sycl_ctx().set_deps(std::move(deps));
    }

    // run primitive
    exec_ctx_t ctx(stream, std::move(exec_args));
    auto status = primitive_execute(primitive_iface, ctx);

    // return output event
    if (return_event_ != nullptr && status == status::success) {
        *(::sycl::event *)return_event_ = sycl_stream_impl->get_output_event();
    }

    stream->after_exec_hook();

    return status;
}

struct dnnl_sycl_interop_execute_handle {
    using exec_args_t = dnnl::impl::exec_args_t;
    using exec_ctx_t = dnnl::impl::exec_ctx_t;
    using stream_t = dnnl::impl::stream_t;

    static dnnl::impl::status_t make(dnnl_sycl_interop_execute_handle **out,
            primitive_iface_t *prim, stream_t *stream, exec_args_t &&args) {
        auto *h = new dnnl_sycl_interop_execute_handle(
                prim, stream, std::move(args));
        auto st = prim->prepare_ctx(h->ctx_);
        if (st != dnnl::impl::status::success) {
            delete h;
            return st;
        }
        *out = h;
        return dnnl::impl::status::success;
    }

    ~dnnl_sycl_interop_execute_handle() { prim_->release(); }

    dnnl_sycl_interop_execute_handle(const dnnl_sycl_interop_execute_handle &)
            = delete;
    dnnl_sycl_interop_execute_handle &operator=(
            const dnnl_sycl_interop_execute_handle &)
            = delete;

    primitive_iface_t *prim() const { return prim_; }
    stream_t *stream() const { return stream_; }
    exec_ctx_t &ctx() { return ctx_; }

private:
    dnnl_sycl_interop_execute_handle(
            primitive_iface_t *prim, stream_t *stream, exec_args_t &&args)
        : prim_(prim)
        , stream_(stream)
        , args_(std::move(args))
        , ctx_(stream, exec_args_t(args_)) {
        prim_->retain();
    }

    primitive_iface_t *prim_;
    stream_t *stream_;
    exec_args_t args_;
    exec_ctx_t ctx_;
};

status_t dnnl_sycl_interop_execute_handle_create(
        dnnl_sycl_interop_execute_handle_t *handle,
        const primitive_iface_t *primitive_iface, stream_t *stream, int nargs,
        const dnnl_exec_arg_t *args) {
    using namespace dnnl::impl;
    bool ok = !utils::any_null(handle, primitive_iface, stream)
            && primitive_iface->engine() == stream->engine()
            && primitive_iface->engine()->runtime_kind() == runtime_kind::sycl
            && IMPLICATION(nargs > 0, args != nullptr);
    if (!ok) return status::invalid_arguments;

    exec_args_t exec_args;
    CHECK(cvt_primitive_args(
            primitive_iface->pd()->impl().get(), nargs, args, exec_args));

    return dnnl_sycl_interop_execute_handle::make(handle,
            const_cast<primitive_iface_t *>(primitive_iface), stream,
            std::move(exec_args));
}

status_t dnnl_sycl_interop_execute_handle_destroy(
        dnnl_sycl_interop_execute_handle_t handle) {
    delete handle;
    return dnnl::impl::status::success;
}

status_t dnnl_sycl_interop_primitive_execute_fast(
        dnnl_sycl_interop_execute_handle_t handle, const void *deps_,
        void *return_event_) {
    using namespace dnnl::impl;
    if (!handle) return status::invalid_arguments;

    auto *sycl_stream_impl
            = utils::downcast<dnnl::impl::xpu::sycl::stream_impl_t *>(
                    handle->stream()->impl());

    if (deps_ != nullptr) {
        auto deps = dnnl::impl::xpu::sycl::event_t(
                *(const std::vector<::sycl::event> *)deps_);
        sycl_stream_impl->sycl_ctx().set_deps(std::move(deps));
    }

    auto status = handle->prim()->execute_fast(handle->ctx());

    if (return_event_ != nullptr && status == status::success) {
        *(::sycl::event *)return_event_ = sycl_stream_impl->get_output_event();
    }

    sycl_stream_impl->sycl_ctx().set_deps(dnnl::impl::xpu::sycl::event_t());

    return status;
}

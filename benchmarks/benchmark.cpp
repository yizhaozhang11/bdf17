#include <benchmark/benchmark.h>

#include <vector>

#include "ntt.h"
#include "ntt_plan.hpp"

#define REPETITIONS 3

namespace {

template <typename Transform>
CoeffPoly<Transform> DeterministicInput() {
    std::vector<uint64_t> data(Transform::N, 0);
    for (size_t i = 0; i < Transform::N; ++i) {
        data[i] = ((i * 17ULL) + 13ULL) % Transform::p;
    }
    return CoeffPoly<Transform>::FromUnsigned(data);
}

template <typename Transform, Backend B>
static void BM_PlanForward(benchmark::State &state) {
    using Plan = CanonicalNttPlan<Transform, B>;
    Plan plan;
    typename Plan::Workspace workspace;
    const auto coeff = DeterministicInput<Transform>();

    for (auto _ : state) {
        const auto eval = plan.forward(coeff, workspace);
        benchmark::DoNotOptimize(eval.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * Transform::N));
}

using Primitive769 = NTT<1152921504602791681ULL, 11ULL, 769, 11>;
using Primitive12289 = NTT<1152921504107839489ULL, 19ULL, 12289, 11>;

} // namespace

static void BM_PlanForwardPrimitive769Auto(benchmark::State &state) {
    BM_PlanForward<Primitive769, Backend::Auto>(state);
}

BENCHMARK(BM_PlanForwardPrimitive769Auto)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_PlanForwardPrimitive12289Auto(benchmark::State &state) {
    BM_PlanForward<Primitive12289, Backend::Auto>(state);
}

BENCHMARK(BM_PlanForwardPrimitive12289Auto)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_PlanForwardPrimitive12289Scalar(benchmark::State &state) {
    BM_PlanForward<Primitive12289, Backend::Scalar>(state);
}

BENCHMARK(BM_PlanForwardPrimitive12289Scalar)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

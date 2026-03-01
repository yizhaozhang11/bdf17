# bdf17 Architecture (Paper-Mapped)

## Scope

This repository is a BDF17-inspired research prototype, not production cryptography. The high-level structure mirrors the paper pipeline, but the default combine path is currently `ExpCrtVariant::TensorTrick`, and `ExpCrtVariant::Paper` is not implemented.

## Pipeline Summary (Figure 1 / Algorithm 7)

Algorithm 7 (`EvalBootstrap`) is implemented as an explicit staged pipeline in [include/bootstrap_runner.hpp](../include/bootstrap_runner.hpp) (`BootstrapRunner::RunTrial`):

1. `EncryptInputBits`
2. `PackBits`
3. `FrontendKeySwitchIfNeeded`
4. `FrontendModSwitchIn`
5. `RunAccumulatorP`
6. `RunAccumulatorQ`
7. `RunExpCRT`
8. `RunFunExtract`
9. `FrontendModSwitchOut`
10. `DecodeAndCheck`

Paper algorithm alignment:

- Algorithm 6 (`ExtExpInner`): [include/accumulator.hpp](../include/accumulator.hpp) (`ExtExpInner`) and underlying RLWE process code in [include/rlwe-impl.h](../include/rlwe-impl.h).
- Algorithm 4 (`ExpCRT`): [include/expcrt.hpp](../include/expcrt.hpp) (`ExpCRT`).
- Algorithm 3 (`FunExpExtract`): [include/fun_extract.hpp](../include/fun_extract.hpp) (`FunExtract`, LUT construction, trace, extraction).

## File-to-Paper Mapping

- [src/main.cpp](../src/main.cpp): thin CLI driver and experiment output formatting (text or JSON).
- [include/bootstrap_runner.hpp](../include/bootstrap_runner.hpp): stage-oriented bootstrap pipeline and per-stage metrics counters.
- [include/lwe_frontend.hpp](../include/lwe_frontend.hpp): LWE encrypt/decrypt, bit packing, modulus switching, optional LWE key-switching.
- [include/accumulator.hpp](../include/accumulator.hpp): accumulator state and `ExtExpInner`-style wrapper over p- and q-ring processing.
- [include/rlwe.h](../include/rlwe.h), [include/rlwe-impl.h](../include/rlwe-impl.h): RLWE/RGSW primitives, key-switching, external multiplication, bootstrapping key generation, accumulator `Process`.
- [include/expcrt.hpp](../include/expcrt.hpp): tensor combine state and `ExpCRT` variant dispatch.
- [include/fun_extract.hpp](../include/fun_extract.hpp): LUT polynomial construction, LUT multiply, trace, and extracted LWE sample generation.
- [include/params.hpp](../include/params.hpp): named crypto profiles and validation invariants.
- [include/experiment_config.hpp](../include/experiment_config.hpp): run-control config and deterministic RNG context.

## Current ExpCRT Status

- `TensorTrick` path:
  - Implemented and used by default.
  - Entry: `ExpCRT(..., ExpCrtVariant::TensorTrick)` in [include/expcrt.hpp](../include/expcrt.hpp).
  - Uses tensor combine plus key-switch into the target key.
- `Paper` path:
  - API enum exists (`ExpCrtVariant::Paper` in [include/expcrt_variant.hpp](../include/expcrt_variant.hpp)).
  - Not implemented in `ExpCRT` and is rejected by profile validation when unsupported.

This separation is intentional: keep the currently working tensor path stable while preserving an explicit seam for a future paper-faithful implementation.

## Parameter Profiles and Validation

Profiles are defined in [include/params.hpp](../include/params.hpp):

- `SmoothNtt1153x1297Profile` (default): experimental smooth-prime NTT profile.
- `ToyEquivalenceProfile`: small deterministic toy profile used for semantic/equivalence testing.

Profile metadata includes:

- ring transforms and derived dimensions/moduli,
- LWE and RLWE noise values,
- secret densities,
- intent flags (`experimental`, `toy`, `paper-comparison`),
- support flags for ExpCRT variants.

Validation (`ValidateProfileOrThrow`) checks key invariants early (modulus shape, dimensions, dim-reduction consistency, combine-path compatibility, explicit moduli, and variant support).

Experiment knobs (seed, trials, LUT, variant, JSON output, dim-reduction override) are in [include/experiment_config.hpp](../include/experiment_config.hpp) and parsed by [src/main.cpp](../src/main.cpp).

## Benchmark Modes

Benchmarks live in [benchmarks/benchmark.cpp](../benchmarks/benchmark.cpp) and cover two classes:

- Transform microbenchmarks:
  - primitive NTT forward benchmarks (`BM_PlanForward...`).
- Pipeline benchmarks:
  - stage-level warm benchmarks (`accum_p`, `accum_q`, `expcrt`, `fun_extract`),
  - end-to-end trial benchmark in warm mode,
  - end-to-end trial benchmark in cold mode (rebuild runner/state each iteration).

Each pipeline benchmark emits metadata (profile, seed, backend, setup mode, stage/kind labels) and stage/trial counters.

## What Equivalence Tests Prove (and Do Not Prove)

The equivalence suite in [tests/equivalence_test.cpp](../tests/equivalence_test.cpp) compares the current pair-basis/tensor-trick path to an independent paper-shaped folded reference model on toy instances.

What it does check:

- fold/index consistency and transport/twist identities,
- tensor-NTT agreement with coefficient-space tensor references,
- noiseless semantic agreement of LUT + trace + extraction behavior,
- a reference-coverage seam for future paper combine implementation.

What it does not check:

- large-parameter noise behavior or concrete security,
- constant-time behavior or side-channel resistance,
- paper-faithful runtime/memory claims,
- implementation correctness of a paper `ExpCRT` path (because that path is not yet implemented).

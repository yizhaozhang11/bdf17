# bdf17

Prime-length NTT / tensor-ring research prototype for experiments around **BDF17**:

> *Large FHE Gates from Tensored Homomorphic Accumulator* (IACR ePrint 2017/996)

This repository is **not production cryptography**. It is a research codebase for:

- exploring a **prime-length NTT** backend for the BDF17-style accumulator pipeline,
- testing a **typed polynomial / NTT-plan refactor**,
- validating a **pair-basis tensor implementation** against the paper's folded `R_pq` view on toy instances,
- and building an end-to-end demo with an **LWE frontend, bit packing, optional LWE key-switching, two accumulators, tensor combine, LUT application, trace, and LWE extraction**.

The current code is much closer to an end-to-end prototype than the earlier version of this repo, but it is still **not a paper-faithful reproduction** of Borogrove. In particular, the default path still uses a custom `TensorTrick` ExpCRT variant; the explicit `ExpCrtVariant::Paper` path is present as an API option but is **not implemented yet**.

---

## Status at a glance

### Implemented now

- **Prime-length circulant NTTs** with a mixed-radix `2^u * 3^v` kernel.
- **Typed coefficient/evaluation polynomials** with reusable NTT plans.
- **Backend selection** (`Auto`, `Scalar`, `Avx2`, `Avx512`) for NTT plans.
- **Heap-backed polynomial storage** (`typed_poly.hpp`), which avoids the old inline giant-array design.
- **LWE frontend**:
  - message encoding / decoding,
  - LWE encryption / decryption,
  - modulus switching,
  - little-endian bit packing,
  - optional LWE dimension-reduction key-switching.
- **Accumulator path**:
  - bootstrapping key generation in the `p`- and `q`-rings,
  - `Process()` / `ExtExpInner`-style accumulator evaluation,
  - mod-switch into the tensor-stage rings.
- **Tensor combine + extraction path**:
  - current `TensorTrick` ExpCRT-like combine,
  - LUT construction in the current tensor basis,
  - LUT multiply, trace `R_p x R_q -> R_p`, and LWE extraction.
- **Unit tests** for NTT correctness, backend equivalence, typed polynomial semantics, LWE frontend behavior, zero-sum RLWE `a` sampling, and toy-size equivalence checks against an independent paper-style reference model.

### Still experimental / incomplete

- `ExpCrtVariant::Paper` is **not implemented**.
- The default parameter set is **not** the Borogrove parameter set.
- The code does **not** claim full paper-level performance, memory, or noise reproduction.
- This is **not constant-time** or hardened for production use.

---

## What the current executable does

The demo entrypoint is `src/main.cpp` and the executable target is `bdf17`.

At a high level, it performs the following pipeline:

1. Sample an **LWE frontend secret**.
2. Optionally sample a distinct **LWE accumulator secret** and generate a frontend-to-accumulator LWE key-switch key.
3. Build a plaintext LUT (currently low-bit by default), lift it to tensor samples, and precompute its evaluation-domain polynomial.
4. Generate accumulator state:
   - fresh ring secrets for the `p`- and `q`-rings,
   - Galois key-switch keys,
   - bootstrapping keys from the accumulator LWE secret.
5. Generate tensor/ExpCRT state using the frontend secret plus the `p`- and `q`-ring secrets.
6. For each trial:
   - encrypt random input bits under the frontend secret,
   - pack them little-endian into one LWE ciphertext over `Z_t` with `t = 64`,
   - optionally key-switch to the accumulator LWE secret,
   - modulus-switch into the accumulator input modulus,
   - run the `p`- and `q`-ring accumulator processes,
   - combine them with `ExpCRT(..., ExpCrtVariant::TensorTrick)`,
   - apply the LUT, trace, and extract an LWE ciphertext,
   - modulus-switch the extracted sample back to the frontend modulus if needed,
   - decrypt and compare against the expected LUT value.

By default the packed message is a 6-bit word because `kPlainModulus = 64`, so `MaxPackingBits(64) = 6`.

---

## Current relationship to the BDF17 paper / Borogrove

This repository should be read as **"BDF17-inspired prototype with a different transform/backend strategy"**, not as a drop-in clone of Borogrove.

### What matches the paper at a structural level

- Two separate accumulators over rings of degrees `p` and `q`.
- A tensor/combine stage before extraction.
- LUT-based function evaluation during bootstrap.
- A trace-and-extract path that returns to an LWE-like output sample.
- BDF17-style **zero-sum CLWE `a` sampling** in `RLWEEncrypt`.
- A frontend design that explicitly distinguishes:
  - frontend modulus,
  - accumulator-input modulus,
  - extract modulus.

### What is intentionally different right now

- The default parameter set is tuned for a **smooth-prime NTT** experiment:
  - `p = 1153`, `q = 1297`,
  - with `p-1 = 1152 = 2^7 * 3^2`,
  - and `q-1 = 1296 = 2^4 * 3^4`.
- The current tensor/combine path uses `ExpCrtVariant::TensorTrick`, not the paper's explicit ExpCRT implementation.
- The code largely works in a **direct tensor / pair basis** for `R_p x R_q` and then tests equivalence to the paper's folded `R_pq` picture on toy instances.

### Important nuance about the equivalence tests

The repo now includes a substantial toy-size equivalence suite (`tests/equivalence_test.cpp`) that compares the current pair-basis formulation against an independent paper-style folded reference model.

Those tests check, in a **noiseless toy setting**, that:

- the current LUT semantics agree with the paper bootstrap function,
- the current fold / trace behavior matches the paper after the appropriate CRT transport / twists,
- the current `TensorTrick` ciphertext phase matches the paper-style reference phase,
- end-to-end LUT + trace semantics agree on toy examples.

That is useful evidence that the current basis choice is mathematically aligned on small instances, but it is **not the same thing** as having implemented the paper's exact ExpCRT algorithm for the default large parameters.

---

## Default parameters

The default parameter bundle lives in `include/params.hpp`.

### LWE / plaintext parameters

- `kLweFrontendDimension = 600`
- `kLweAccumulatorDimension = 600`
- `kPlainModulus = 64`
- `kKeySwitchBase = 2^8`
- `kLweKeySwitchBase = 2^8`
- `kEnableLweDimReduction = false`
- `kLweNoiseVar = 4.0`

Since the frontend and accumulator dimensions are both `600`, the optional LWE dimension-reduction path is currently compiled in but **inactive by default**.

### Ring / transform parameters

Accumulator rings:

- `NTTp = CircNTT<72057421557668737, 5, 1153, 5>`
- `NTTq = CircNTT<72057421557668737, 5, 1297, 10>`

Mod-switched rings used before tensor combine:

- `NTTpt = CircNTT<108533126017, 10, 1153, 5>`
- `NTTqt = CircNTT<108533126017, 10, 1297, 10>`

Tensor ring:

- `NTTpq = TensorNTTImpl<NTTpt, NTTqt>`

Derived moduli / dimensions:

- `kTensorDimension = 1153 * 1297 = 1495441`
- `kAccumulatorInputModulus = kTensorDimension`
- `kExtractModulus = NTTpq::Z::p = 108533126017`
- `kFrontendModulus = kExtractModulus` (default)

The code keeps the modulus roles separate even though the current default makes `kFrontendModulus == kExtractModulus`.

---

## Repository layout

```text
include/
  accumulator.hpp     Accumulator state and ExtExpInner-style processing wrapper
  expcrt.hpp          Tensor combine / ExpCRT state and current TensorTrick path
  fun_extract.hpp     LUT construction, trace, and LWE extraction
  lwe_frontend.hpp    LWE encrypt/decrypt, modswitch, packing, LWE key-switching
  ntt.h               Prime-length NTT, circulant NTT, tensor NTT kernels
  ntt_backend.hpp     Backend enum and compile-time backend detection
  ntt_plan.hpp        Canonical forward/inverse plan wrapper with reusable workspace
  params.hpp          Default parameter bundle
  rlwe.h              SchemeImpl interface
  rlwe-impl.h         SchemeImpl implementation
  typed_poly.hpp      Heap-backed typed coefficient/evaluation polynomial container
  zp.h                Modular arithmetic helpers

src/
  main.cpp            End-to-end demo driver

tests/
  test.cpp            NTT, tensor NTT, boundary-vector, Galois, zero-sum RLWE tests
  typed_poly_test.cpp Typed polynomial semantics
  ntt_plan_test.cpp   Plan/backends/workspace tests
  lwe_frontend_test.cpp
                      LWE encode/decode, packing, and LWE key-switch tests
  equivalence_test.cpp
                      Toy-size equivalence checks vs paper-style folded semantics
  equivalence_ref.hpp Reference helpers used by equivalence_test.cpp

benchmarks/
  benchmark.cpp       NTT-plan microbenchmarks

scripts/
  gen.sage            Helper for generating smooth-prime-friendly NTT moduli
```

---

## Transform strategy

The central performance experiment in this repo is the use of **prime-length NTTs** for the `p`- and `q`-rings.

For a prime ring degree `O`, the implementation uses a prime-length transform built from:

- a Rader-style reduction from size `O` to `O - 1`, and
- a mixed-radix kernel specialized to the case where `O - 1 = 2^u * 3^v`.

This is why the default experiment uses:

- `1153 - 1 = 1152 = 2^7 * 3^2`
- `1297 - 1 = 1296 = 2^4 * 3^4`

instead of the Borogrove demo's `1439` and `1447`.

The code now exposes this transform stack through:

- `CanonicalNttPlan<Transform, Backend>`
- typed coefficient and evaluation polys (`CoeffPoly`, `EvalPoly`)
- reusable workspace allocation in `CanonicalNttPlan::Workspace`

Backend selection is handled through `ntt_backend.hpp` and the NTT plan supports:

- `Backend::Auto`
- `Backend::Scalar`
- `Backend::Avx2`
- `Backend::Avx512`

The tests compare these backends on toy rings, the current default rings, and tensor rings where possible.

---

## Build

### CMake build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Useful options:

```bash
- DENABLE_BENCHMARKS=ON|OFF   # default ON
- DENABLE_AVX2=ON|OFF         # default ON
- DENABLE_AVX512=ON|OFF       # default OFF
```

### Dependency behavior

- Tests are always enabled by the top-level `CMakeLists.txt`.
- `tests/CMakeLists.txt` will try `find_package(GTest)` first and otherwise fall back to `FetchContent`.
- Benchmarks are optional; if enabled, `benchmarks/CMakeLists.txt` will try `find_package(benchmark)` first and otherwise fall back to `FetchContent`.

So if you do not have system-installed GTest / Google Benchmark, CMake may attempt to download them.

### Minimal demo-only build without CMake tests/benchmarks

For quick local experimentation with just `src/main.cpp`, a direct compile also works:

```bash
g++ -std=c++23 -O3 -march=native -fopenmp -Iinclude src/main.cpp -o bdf17_demo
```

To mimic the default AVX2-enabled CMake path on an AVX2-capable machine, add:

```bash
-DBDF17_ENABLE_AVX2=1 -mavx2
```

---

## Run

### End-to-end demo

```bash
./build/bdf17
```

The demo prints stage timings and, for each trial, the extracted result and the expected LUT value.

### Tests

```bash
ctest --test-dir build --output-on-failure
```

Optional large edge-case NTT matrix tests are gated behind an environment variable:

```bash
BDF17_ENABLE_EXTENDED_NTT_TESTS=1 ctest --test-dir build --output-on-failure -R NTTMatrix
```

### Benchmarks

```bash
./build/bdf17_benchmarks
```

The benchmark target now includes:

- NTT-plan microbenchmarks,
- stage-level pipeline benchmarks (`accum_p`, `accum_q`, `expcrt`, `fun_extract`),
- end-to-end bootstrap trial benchmarks in:
  - `setup=warm` mode (setup reused across iterations),
  - `setup=cold` mode (setup included in each iteration).

Each pipeline benchmark emits metadata/counters for profile, seed, backend, LUT kind, ring sizes, stage timings, and approximate key bytes.

Script-friendly export is available via Google Benchmark JSON output:

```bash
./build/bdf17_benchmarks --benchmark_filter=BM_Bootstrap --benchmark_format=json
```

---

## What the tests cover

### NTT / tensor NTT

- roundtrip correctness,
- linearity,
- boundary-vector stress cases,
- separable tensor-NTT agreement,
- optional larger `(u, v)` mixed-radix edge cases.

### NTT plans and backends

- plan forward/inverse matches raw transform semantics,
- `Auto` matches scalar,
- AVX2/AVX512 (when compiled) match scalar,
- reusable workspace behavior.

### Typed polynomial layer

- signed / unsigned coefficient normalization,
- monomial construction,
- coefficient-domain arithmetic,
- evaluation-domain pointwise multiplication,
- coefficient- and evaluation-domain Galois actions.

### LWE frontend

- message encode/decode,
- deterministic encrypt/decrypt tests,
- little-endian bit packing,
- rejection of over-wide packing,
- LWE key-switch correctness.

### RLWE / paper-alignment checks

- zero-sum `a` sampling in `RLWEEncrypt`,
- toy-size equivalence between the current tensor basis and a paper-style folded reference,
- noiseless semantic agreement for LUT + trace behavior.

---

## Notable implementation details

### Typed polynomials are heap-backed now

`typed_poly.hpp` stores coefficients in `std::unique_ptr<uint64_t[]>` rather than inline fixed-size arrays.
This makes the polynomial representation much safer for large tensor-ring objects and is one of the main engineering improvements relative to the earlier prototype state.

### Zero-sum CLWE sampling is implemented

`SchemeImpl::RLWEEncrypt` samples the RLWE `a` term in the **sum-zero subspace** by choosing random coefficients for indices `1..N-1` and then setting index `0` so the coefficient sum is `0 mod Q`.

That aligns the ring-side sampling more closely with the BDF17 / Borogrove CLWE model.

### Frontend / internal modulus flow is explicit

The main driver keeps three distinct modulus roles visible:

- `q_frontend`
- `q_accumulator_input`
- `q_extract_internal`

This makes it easier to experiment with the paper-style `Q0 -> pq -> Q_extract -> Q0` shape, even though the default parameters currently choose `q_frontend == q_extract_internal`.

---

## Current limitations / open gaps

These are the main things to keep in mind when reading or extending the code:

1. **Paper ExpCRT is still missing**
   - `ExpCrtVariant::Paper` throws at runtime.
   - The active path is `ExpCrtVariant::TensorTrick`.

2. **The default parameters are experimental, not Borogrove-faithful**
   - The repo is currently tuned for the smooth-prime NTT experiment.

3. **The benchmark target is still transform-centric**
   - there is no Borogrove-style full bootstrap timing / memory breakdown yet.

4. **No security estimator integration**
   - current defaults are engineering parameters for experimentation, not a finished security story.

5. **Not production crypto**
   - no constant-time claims,
   - no side-channel hardening,
   - no API stability guarantees.

---

## Suggested next steps

If your goal is to move this toward a paper-comparison artifact, the most meaningful next steps are:

- implement the actual `ExpCrtVariant::Paper` path,
- add stage-by-stage bootstrap benchmarking and memory accounting,
- introduce alternate parameter bundles (for example a Borogrove-style comparison profile),
- add a parameter search / validation flow that connects transform smoothness, modulus constraints, and correctness/noise constraints,
- keep extending the equivalence tests so the pair-basis and folded-basis stories remain easy to check.

---

## References

- BDF17: *Large FHE Gates from Tensored Homomorphic Accumulator* (IACR ePrint 2017/996)
- Borogrove: the reference implementation released by the paper's authors

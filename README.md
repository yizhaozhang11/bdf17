# bdf17 — Prime-length NTT prototype for “Large FHE Gates from Tensored Homomorphic Accumulator” (BDF17)

> **Research prototype / not production crypto.**
>
> This repository is an experiment-driven C++ prototype inspired by the paper:
>
> *Large FHE Gates from Tensored Homomorphic Accumulator* (BDF17, IACR ePrint 2017/996).
>
> The original authors also released a reference implementation (**Borogrove**) that targets the
> paper’s 6‑bit-gate parameter set.
>
> **Goal of this repo:** reproduce the BDF17 “large-gate bootstrapping” pipeline *and* explore
> faster transform strategies (prime-length NTT + parameter/prime selection) to reduce the dominant FFT cost.

---

## 1. What problem BDF17 solves (paper context)

Bootstrapping in FHEW/TFHE-style schemes can be viewed as:

1. a **linear** operation that computes (under encryption) the decryption inner product  
   \(L_c(s) = b - \langle a, s \rangle\) (mod the LWE modulus), and
2. a **nonlinear** map that extracts a fresh ciphertext encrypting a function of the resulting message.

BDF17’s key idea is to make this bootstrapping compute **“large gates”**:

- Instead of bootstrapping just one bit, set a plaintext modulus \(t = 2^k\),
- Pack \(k\) input bits into one plaintext word \(m \in \mathbb{Z}_t\),
- Evaluate an arbitrary lookup-table function \(f: \mathbb{Z}_t \to \mathbb{Z}_t\) *during bootstrapping*.

This turns “one bootstrap = one arbitrary k‑input gate” (or even k‑to‑k with repeated extraction).

### Tensored Homomorphic Accumulator (THA)

Classic accumulator bootstrapping becomes expensive when \(t\) grows.
BDF17 introduces a **tensored** approach:

- Run **two smaller accumulators** in rings of degree \(p\) and \(q\),
- Obtain encryptions of \(X^{m \bmod p}\) and \(Y^{m \bmod q}\),
- Use a CRT/tensor construction (“ExpCRT”) to combine them into an encryption of \(Z^{m \bmod pq}\)
  in a tensor ring of degree \(pq\),
- Apply a **function extraction** that multiplies by a “function polynomial” \(F(Z)\), traces down,
  and outputs a refreshed LWE ciphertext.

### Paper’s demo target (for orientation)

BDF17’s first implementation targets a **6‑bit input gate** (\(t=2^6=64\)).
The paper reports (single-threaded on a laptop of that era):

- one-time FFTW “wisdom” planning: ~68 minutes,
- key preprocessing: ~38 seconds,
- one 6‑input / 1‑output gate evaluation: ~6.4 seconds,
  with the breakdown roughly: ~0.60 s per accumulator (two of them), ~4.0 s for the large key switch
  inside function extraction, and ~0.55 s for “output-bit” related work,
- memory for key material: ~9.2 GB.

**Those numbers (and the parameter set behind them) are the “paper result” we eventually want to match.**

---

## 2. What this repository currently is (and is not)

### What’s implemented today (high level)

This repo implements the *shape* of the BDF17 pipeline:

1. **Accumulator step** in a \(p\)-ring and a \(q\)-ring  
   (roughly paper’s `ExtExpInner` / “homomorphic accumulator”).
2. **Tensor combine** the two outputs into a \(pq\)-ring ciphertext.
3. **Function extraction** via:
   - constructing a LUT polynomial `F`,
   - multiplying ciphertext by `F` in the transform domain,
   - tracing from \(R_{pq} \to R_p\),
   - extracting an LWE-like `(a_out, b_out)` and verifying.

### What’s missing vs the paper / Borogrove reference

This code is **not yet** an end-to-end faithful reproduction. Most notably it currently lacks:

- A proper **LWE encryption frontend** (with noise) for input ciphertexts.
  The demo synthesizes `(a,b)` using the secret key so it can test algebraic correctness.
- The paper/Borogrove **“combination”** step that packs *multiple* encrypted bits into one word.
- The explicit **LWE dimension reduction key switch** used in the paper demo
  (e.g., 1439 → 600 in Borogrove).
- A clearly separated implementation of the paper’s **ExpCRT** (Galois twists + rescale);
  this repo currently performs a simplified “tensor + key switch” construction.
- Stage-by-stage benchmarking and memory accounting comparable to Borogrove’s `stats/`.

So: treat this as a *prototype kernel* for experimenting with transforms and ring choices,
not as a drop-in reproduction.

---

## 3. Repo layout & “where to start reading”

```
include/
  ntt.h          Prime-length NTT + tensor NTT (fast path is 2^u*3^v)
  poly.h         Polynomial wrapper with domain flag (coeff vs NTT)
  zp.h           Zp arithmetic + Barrett fast multiply
  rlwe.h         SchemeImpl interface: RLWE/RGSW, key switching, accumulator
  rlwe-impl.h    Implementations (encrypt/decrypt/keyswitch/extmult/process)

src/
  bdf17.cpp      End-to-end demo of the current pipeline (single file “driver”)

benchmarks/
  benchmark.cpp  Microbenchmarks of NTT kernels (Google Benchmark)

scripts/
  gen.sage       Helper for generating primes/primitive roots satisfying NTT constraints

tests/
  test.cpp       (currently empty scaffold)
```

If you’re new:

1. Read `src/bdf17.cpp` first (it shows the full pipeline in one place).
2. Then read `include/rlwe.h` + `include/rlwe-impl.h` (ciphertexts/keyswitch/accumulator).
3. Finally read `include/ntt.h` (prime-length NTT implementation + tensor NTT).

---

## 4. Mathematical model & encoding conventions used here

### Rings

For a ring degree `O` (prime in the current parameter choices), we work in the **circulant ring**
\(R_O = \mathbb{Z}_Q[X]/(X^O - 1)\).

- `Poly<NTT>` stores a polynomial of length `NTT::N`.
- When using `CircNTT<..., O, ...>`, `NTT::N == O` and the transform is an NTT of size `O`
  implemented via a prime-length method (see below).

The tensor ring is the product ring of degrees `p*q` (implemented as a 2D tensor NTT):

\[
R_{pq} \cong \mathbb{Z}_Q[X]/(X^p-1) \otimes \mathbb{Z}_Q[Y]/(Y^q-1).
\]

### Ciphertext shapes in `SchemeImpl`

`SchemeImpl<Poly,B>` is a minimal RLWE/RGSW toolchain:

- **RLWE secret key**: `RLWEKey` is `std::vector<Poly>` of size `k`
- **RLWE ciphertext**: `RLWECiphertext` is `std::vector<Poly>` of size `k+1`
  - first `k` polys are the `a_i`
  - last poly is the `b`
- **RGSW ciphertext**: `RGSWCiphertext` is a pair of gadget ciphertexts
  `(ct_for_m*s, ct_for_m)`.

`Poly::is_coeff` indicates the domain:

- `is_coeff = true`: coefficient domain
- `is_coeff = false`: NTT domain

Important invariant:
- `Poly * Poly` multiplication is only allowed in NTT domain (`is_coeff=false`),
  and is pointwise multiplication.

### Message encoding used by the demo

The demo uses:

- plaintext modulus `Qplain = 64` (paper uses `t=64` as well)
- an “LWE modulus” effectively of size `pq = p*q`.

The driver synthesizes an LWE-like sample `(a,b)` so that:

\[
b - \langle a, s \rangle \equiv \left\lfloor \frac{pq}{t} \cdot m \right\rceil \pmod{pq},
\]

where `m = b0` is the intended plaintext in `[0, t)`.

The accumulator step then aims to produce an RLWE encryption of an exponent gadget
corresponding to that encoded message (conceptually `X^m` in the exponent space).

---

## 5. Pipeline overview and mapping to code

This repo’s `main()` in `src/bdf17.cpp` performs:

### Step A — choose predicate `f` (LUT)

- `f_plain`: the function on plaintext messages `m ∈ {0,…,t-1}`
- `f_ct`: an expanded LUT of length `pq` which accounts for modulus switching / scaling
  from `pq` back to `t`.

Currently the demo sets:
- `f_plain[m] = m & 1` (parity of the 6-bit word)

Then:
- `ConstructF(f_ct)` builds a polynomial `F` over the `pq`-ring used for extraction,
  and `F.ToNTT()` prepares it for pointwise multiplication.

### Step B — keys & evaluation keys

- `sk`: an LWE secret (length `n=600`) sampled as sparse ternary
- `skp`, `skq`: ring secrets for the `p`- and `q`-rings
- `schemeP`, `schemeQ`: `SchemeImpl` instances in `p`-ring and `q`-ring
  - `GaloisKeyGen()` precomputes galois key-switch keys for automorphisms
  - `BootstrappingKeyGen(sk)` encrypts the LWE secret components in RGSW form
- `schemePQ`: a `pq`-ring scheme used for the tensor/key-switch part
- `TensorKey(...)` builds a 3-component tensor secret key (see “ExpCRT note” below)
- `tensorBK = schemePQ.KeySwitchGen(skpq, skp0)` builds a key switch from the 3‑key tensor secret
  down to an embedded 1‑key secret used for extraction.

### Step C — accumulator / “linear” bootstrapping part

For each randomized test input, we build synthetic `(a,b)` and run:

- `schemeP.Process(BKp, a, b, Qplain)`  
- `schemeQ.Process(BKq, a, b, Qplain)`

`Process()` is the main accumulator routine and corresponds to the paper’s
“ExtExpInner / homomorphic accumulator” idea:
it repeatedly applies automorphisms (Galois conjugates), key switching,
and external multiplication by RGSW encryptions of secret components.

Then the demo mod-switches these ciphertexts into a second modulus (the “tensored stage modulus”):

- `ctp = SchemeP::ModSwitch<SchemePt>(...)`
- `ctq = SchemeQ::ModSwitch<SchemeQt>(...)`

### Step D — tensor combine (“ExpCRT-like”)

The repo currently combines `ctp` and `ctq` via:

- `ctpqt = TensorCt(ctp, ctq)` which creates a 2×2 tensor RLWE ciphertext (size 4)
- `tensor_ct = schemePQ.KeySwitch(ctpqt, tensorBK)` to reduce it back to a standard 2‑poly RLWE ciphertext

> **Note:** In the BDF17 paper this stage is a specific construction called **ExpCRT**
> (Galois twists + tensor product + rescaling) designed to control noise and scaling.
> This repo currently performs a simplified construction that is useful for experimentation,
> but should be treated as “not yet paper-equivalent” until proven/validated.

### Step E — function extraction & LWE output

- Multiply by LUT poly:
  - `tensor_ct[0] = F * tensor_ct[0]`
  - `tensor_ct[1] = F * tensor_ct[1]`
- Trace down:
  - `ct_trace = { TracePQtoP(tensor_ct[0]), TracePQtoP(tensor_ct[1]) }`
- Extract `(a_out, b_out)` by reading coefficients of the traced ciphertext.

Finally the demo **verifies correctness** by decrypting with the secret key `sk`
(“check, not a production decryption API”).

---

## 6. Prime-length NTT & parameter/prime selection tricks (the “novel FFT” angle)

A major motivation for this repo is to avoid the heavy FFTW cost reported in the paper
(especially the huge `R_{pq}` FFTs).

### What we do here

We pick ring degrees `p` and `q` such that:

- `p` and `q` are prime,
- `p-1` and `q-1` are **very smooth**, specifically of the form `2^u * 3^v`.

Example (current demo):
- `p = 1153`, so `p-1 = 1152 = 2^7 * 3^2`
- `q = 1297`, so `q-1 = 1296 = 2^4 * 3^4`

This allows the prime-length NTT (size `p` or `q`) to reduce to a fast mixed-radix transform
on length `p-1` / `q-1` using only radix‑2 and radix‑3 butterflies (implemented in `CT23NTT`).

### How the NTT works in code

- `NTT<p,g,O,w>` implements a prime-length NTT over the `O-1` non-zero indices
  using a Rader-style permutation (`gi`, `gi_inv`) and a CT23 transform.
- `CircNTT<p,g,O,w>` lifts that into a size-`O` NTT suitable for cyclic convolution
  in `Z_p[X]/(X^O-1)` by handling the DC term separately.
- `TensorNTTImpl<NTTp,NTTq>` implements a 2D NTT for the tensor ring by applying:
  - NTT along the `q` dimension for each `p` row, then
  - NTT along the `p` dimension for each `q` column.

### Modulus selection

To run NTTs of size `O` and `O-1`, we need a prime modulus `P` such that:

- `P ≡ 1 (mod O*(O-1))`

Because `p` and `q` rings share the same modulus in the demo, we also require:

- `P ≡ 1 (mod lcm(p*(p-1), q*(q-1)))`

The helper script `scripts/gen.sage` searches for such primes and prints template parameters.

---

## 7. Building & running

### Dependencies

Core code:
- C++23 compiler
- AVX2 (the NTT kernel uses `<immintrin.h>`)
- OpenMP (enabled via `-fopenmp` in the top-level CMake)

Optional:
- GoogleTest (for tests; fetched via `FetchContent` if not installed)
- Google Benchmark (for benchmarks; fetched via `FetchContent` if not installed)

### Build (default)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the demo:

```bash
./build/bdf17
```

Run benchmarks:

```bash
./build/bdf17_benchmarks
```

Run tests:

```bash
ctest --test-dir build
```

Run extended NTT edge-case matrix tests (includes large `2^u*3^v` cases):

```bash
BDF17_ENABLE_EXTENDED_NTT_TESTS=1 ctest --test-dir build -R NTTMatrix
```

---

## 8. Known engineering hazards / gotchas (important for contributors)

1. **Huge PQ polynomials can overflow the stack**  
   `Poly` stores coefficients as `uint64_t a[N]` inline.
   For `pq = p*q` this is ~1.5M coefficients → ~12MB per polynomial.
   Returning/allocating `PolyPQ` by value inside functions can exceed typical stack limits.

   If you see crashes early in execution: fix by switching PQ polys to heap storage
   or by rewriting tensor/LUT code to use output buffers / move-only types.

2. **Static scratch buffers are not thread-safe**  
   The NTT kernels use `static uint64_t reg[...]` as workspace.
   This is not safe with OpenMP or multi-threaded use.
   Prefer a thread-local or explicit “workspace” object.

3. **Domain discipline matters**  
   `Poly::operator*` assumes both operands are in NTT domain.
   Many functions do `ToCoeff()` / `ToNTT()` internally; avoid repeated toggling in hot paths.

4. **CT23 assumes `O-1` factors only into 2s and 3s**  
   The current fast NTT path is only correct under that assumption.
   If you change `p`/`q`, add invariant checks (or implement a more general mixed radix).

---

## 9. How to modify the demo

### Change the bootstrapped function `f`

In `src/bdf17.cpp`:

- `f_plain` defines the function on `m ∈ [0, Qplain)`.
- `f_ct` expands it to `[0, pq)` after scaling.

Example: majority on a 6-bit word (toy):

```cpp
for (size_t m = 0; m < Qplain; m++) {
    f_plain[m] = (__builtin_popcount((unsigned)m) >= 3);
}
```

### Change the ring degrees (p, q)

- Update the `using NTTp = ...` / `using NTTq = ...` lines.
- Regenerate suitable NTT modulus primes and primitive roots using `scripts/gen.sage`
  (or an extended version for your desired bit-length).

Remember:
- This repo’s NTT prefers `p-1` and `q-1` to be 2/3-smooth.
- If you move toward the paper’s `p=1439, q=1447`, you will need a different transform strategy
  (or a generalized NTT) because `1439-1` and `1447-1` are not 2/3-smooth.

---

## 10. Roadmap toward “paper reproduction + faster FFT” (suggested TODOs)

If the goal is a credible reproduction and a performance story, the next steps are:

### Reproduction completeness
- [ ] Implement **true LWE encryption** of inputs (with noise) instead of synthesizing `(a,b)`.
- [ ] Implement the **combination step** to pack multiple encrypted bits into one word `m`.
- [ ] Implement the paper’s **LWE dimension reduction key switch** (e.g., 1439 → 600).
- [ ] Implement a clearly separated **ExpCRT** stage (paper’s twists + rescale),
      or provide a proof/tests that the current tensor+KS variant is equivalent.

### Transform and parameter experiments
- [ ] Add a parameter search tool (Sage/Python) for selecting `p,q,P,g,w`
      under both correctness/noise constraints and transform smoothness constraints.
- [ ] Compare:
      - FFTW padding approach (Borogrove style),
      - prime-length NTT (this repo),
      - Bluestein/Chirp-Z style,
      - mixed-radix generalized NTT (if allowing small extra primes like 5,7,11).

### Engineering / benchmarking
- [ ] Fix memory model for PQ polynomials (heap-backed, move-only).
- [ ] Add stage-by-stage timers and peak memory accounting comparable to Borogrove.
- [ ] Add correctness tests (NTT roundtrip, multiplication vs naive, keyswitch, extmult, end-to-end).
- [ ] Make build flags configurable (avoid forcing `-Werror`, `-march=native`, `-mavx2` globally).

---

## 11. References

Paper and reference implementation (search by identifier/name):

```text
BDF17 paper: “Large FHE Gates from Tensored Homomorphic Accumulator”
IACR ePrint: 2017/996

Reference code by the authors (Borogrove):
https://github.com/gbonnoron/Borogrove
```

Related background families:
- FHEW / TFHE bootstrapping (accumulator-based gate bootstrapping)

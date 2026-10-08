# Task brief — Lean 4 / Mathlib formalisation of the sub-filter energy identity

You are given one mathematical identity and a proof skeleton (`SubfilterIdentity.lean`, written
without a compiler, so expect small fixes). Your job: make it compile with **no `sorry`**, prove the
listed theorems, and report exactly which hypotheses each theorem uses. Do not weaken statements to
make them easier; if a statement is wrong as written, say so and propose the corrected one.

## 0. What the identity says (context, one paragraph)

In wave–turbulence energetics, `s` is the Stokes drift of a surface wave (a slowly varying field),
`u` is the sub-filter (turbulent) velocity, `ω = curl u`, and `Av` is a filter/average. The identity

```
Av( u·(s × curl u) ) − s·divT( Av(u ⊗ u) ) = − s·grad( ½ Av(u·u) )           (1)
```

says: the work of the Craik–Leibovich force `s × ω` on the sub-filter motion (left term) equals the
rate at which the Reynolds force `divT Av(u⊗u)` of that motion on the filtered current changes the
wave's Doppler energy (middle term), up to a Stokes transport of turbulent kinetic energy (right
term). Physics is irrelevant for your job; the point is that (1) is an **algebraic identity**:
commutative ring + three derivations (Leibniz) + a linear averaging operator with two axioms.

## 1. The algebraic setting (this IS the theorem's hypothesis list)

* `R` a commutative ring and ℝ-algebra (`[CommRing R] [Algebra ℝ R]`).
* `d : Fin 3 → Derivation ℝ R R` (three ℝ-linear derivations; Leibniz is all that is used — we
  never need `d i (d j f) = d j (d i f)`).
* `Av : R →ₗ[ℝ] R` with
  * `Av_comm : ∀ i f, Av (d i f) = d i (Av f)`
  * `Av_res  : ∀ s ∈ resolved, ∀ f, Av (s * f) = s * Av f` where `resolved : Subalgebra ℝ R`.
* Vector fields `V R := Fin 3 → R`; tensors `T R := Fin 3 → Fin 3 → R`.
* Operators: `dot`, `cross` (explicit components via `![…]`), `grad f i = d i f`,
  `div u = ∑ i, d i (u i)`, `curl` (explicit components), `adv a b i = ∑ j, a j * d j (b i)`
  (that is (a·∇)b), `divT T i = ∑ j, d j (T i j)`, `outer a b i j = a i * b j`,
  `AvV`, `AvT` componentwise averages, `gradHalfSq b i = ∑ j, b j * d i (b j)`.

Hypotheses of the main theorem: `div u = 0` and `∀ i, s i ∈ resolved`. **Nothing else.** In
particular `Av u = 0` and idempotence `Av ∘ Av = Av` are NOT assumed for (1); they are used only in
`leonard_vanish`.

## 2. Theorems to prove (names as in the skeleton), with the intended proofs

| # | name | statement (informal) | proof idea |
|---|------|----------------------|------------|
| A1 | `dot_cross_cyclic` | a·(b×c) = b·(c×a) | `simp [dot, cross, Fin.sum_univ_three]; ring` |
| A2 | `dot_cross_swap` | a·(b×c) = −a·(c×b) | same |
| 1 | `step1` | b × curl b = gradHalfSq b − adv b b | `funext i; fin_cases i <;> simp [...] <;> ring`. **No Leibniz** needed: it is the ε–δ identity. |
| 1' | `d_sq`, `gradHalfSq_eq` | ∂(f f) = 2 f ∂f;  gradHalfSq b = ½ grad(b·b) | `Derivation.leibniz` (comes as `a • D b + b • D a`; rewrite `smul_eq_mul`). The ½ lives in ℝ acting on `R`; if `Algebra.smul_def` juggling is painful, prove the `2 •`-cleared form `2 • gradHalfSq b = grad (dot b b)` first and derive the ½ form from it. |
| 2 | `step2` | div u = 0 ⇒ adv u u = divT (outer u u) | Leibniz on `d j (u i * u j)`, split the sum, pull `u i` out of `∑ j, d j (u j) = div u`, use `hu`. |
| P | `pointwise` | u·(s×curl u) = s·divT(u⊗u) − s·gradHalfSq u | A1, A2, step1, step2, then `ring` after unfolding `dot`. |
| 3a | `Av_dot_res` | s resolved ⇒ Av(s·X) = s·AvV X | `map_sum` + `Av_res` termwise |
| 3b | `AvV_divT` | AvV (divT T) = divT (AvT T) | `map_sum` + `Av_comm` |
| 3c | `AvV_gradHalfSq` | AvV (gradHalfSq u) = ½ grad (Av (u·u)) | 1' + `LinearMap.map_smul` + `Av_comm` |
| **M** | **`identity_one`** | **(1)** | P, then `map_sub`, 3a twice, 3b, 3c, unfold, `ring` |
| 4a | `sym_antisym` | R symmetric ⇒ ∑ R_ij(∂_i s_j − ∂_j s_i) = 0 | expand `Fin.sum_univ_three`, rewrite the three off-diagonal symmetries, `ring` |
| 4b | `step4_split` | u·(s×ω) = div(u (s·u)) − s·gradHalfSq u − ∑ u_i u_j ∂_j s_i | Leibniz on `d i (u i * (s·u))`, use `hu`, compare with P using 4a with `Rt i j = u i * u j`. This is the MSM97 / Suzuki–Fox-Kemper three-term split: flux + Stokes advection + Stokes production. |
| L1 | `leonard` | Av(u_i u_j) − U_i U_j = L + C + Av(u'_i u'_j) for ANY linear Av | pure linearity: expand the product, `map_add`, `ring` |
| L2 | `leonard_vanish` | idempotent Av and U ∈ resolved ⇒ L = 0 and C = 0 | `Av_res`, idempotence |

Everything above is Part 1 and is **required**.

## 3. Part 2 (required): non-vacuity model

Show the axioms are satisfiable by a non-trivial instance. Suggested model:

* `R := Fin 2 → MvPolynomial (Fin 3) ℝ` (a two-member ensemble of polynomial fields in x,y,z).
* `d i` acts componentwise by `MvPolynomial.pderiv i` (exists in Mathlib as a `Derivation`); build
  the `Derivation` on the Pi ring by hand (`toLinearMap := LinearMap.pi (fun k => (pderiv i).toLinearMap ∘ₗ LinearMap.proj k)`, prove `leibniz'` componentwise).
* `Av f := fun _ => (2:ℝ)⁻¹ • (f 0 + f 1)` (ensemble mean, broadcast to both members).
* `resolved := { f | f 0 = f 1 }` as a `Subalgebra` (fields that do not depend on the ensemble member).
* Prove `Av_comm`, `Av_res`, and additionally `Av ∘ Av = Av` and `Av (f − Av f) = 0`, so this model is a
  genuine Reynolds operator. Then `identity_one` specialises to it.

If the Pi-ring derivation is too fiddly, an acceptable fallback is `R := MvPolynomial (Fin 3) ℝ` with
`Av := id`, `resolved := ⊤` (degenerate but shows consistency); say clearly which one you did.

## 4. Part 3 (optional, do after Parts 1–2 compile): the analytic steps

These are the depth-integrated statements used in the note. Real analysis in Mathlib.

* **Step 5 (algebra only).** For `σ > 0`, `σ^2 = g*k`, `N = g*a^2/(2*σ)`:
  `N * k * (2*k*Real.exp (2*k*z)) = a^2 * σ * k * Real.exp (2*k*z)`. (`field_simp; nlinarith`/`ring` using `hσ`.)
  Optional: `∫ z in Set.Iic 0, 2*k*Real.exp (2*k*z) = 1` for `k > 0` (improper integral; use
  `integral_exp_mul_Iic`-type lemmas or substitution; skip if Mathlib lacks a convenient form).
* **Step 6 (integration by parts on [−h, 0]).** For `τ uS : ℝ → ℝ` with continuous derivatives on
  `[-h, 0]` and `τ (-h) = 0`:
  `-(∫ z in (-h)..0, uS z * deriv τ z) = (∫ z in (-h)..0, τ z * deriv uS z) - τ 0 * uS 0`.
  Use `intervalIntegral.integral_mul_deriv_eq_deriv_mul`. This is "−∫ uˢ ∂_z τ̄ = P_S − W_S".

Do **not** attempt the small-amplitude action-projection check (§8 of the note); it rests on physical
modelling assumptions and is out of scope.

## 5. Project layout and commands

```
subfilter/
  lakefile.lean           -- Mathlib dependency (lake new subfilter math, or add `require mathlib`)
  lean-toolchain          -- match Mathlib's
  Subfilter/Identity.lean -- Part 1  (start from SubfilterIdentity.lean)
  Subfilter/Model.lean    -- Part 2
  Subfilter/Analytic.lean -- Part 3 (optional)
  Subfilter.lean          -- imports the above
  README.md               -- the report (see §7)
```

If Mathlib is not yet in the project: `lake new subfilter math && cd subfilter && lake exe cache get && lake build`.
Always run `lake exe cache get` before the first build; never build Mathlib from source.

## 6. Acceptance criteria

1. `lake build` succeeds with zero warnings about `sorry`.
2. `#print axioms Subfilter.identity_one` shows only `propext`, `Classical.choice`, `Quot.sound`.
   Same for `step4_split`, `leonard`, `leonard_vanish`.
3. Statements are not weakened. In particular `identity_one` must NOT assume `Av u = 0`,
   idempotence, or commuting derivations. If you find one of these is actually necessary, stop and
   report — that would be a mathematical finding, not a formalisation detail.
4. Each theorem carries a docstring naming the Step of the note it formalises (Step 1, 2, 3, 4,
   Leonard) and listing the axioms it uses.
5. Part 2 compiles and `example : Subfilter.Setup (Fin 2 → MvPolynomial (Fin 3) ℝ) := …` is provided.

## 7. Report back (README.md, ≤ 1 page)

* Table: theorem name → informal statement → hypotheses actually used (Leibniz / Av_comm / Av_res /
  idempotence / div u = 0 / s resolved).
* Any statement you had to correct, with the corrected form and why.
* Lean/Mathlib versions, and the exact `lake build` output tail.
* Anything in the skeleton's intended proofs that was wrong.

## 8. Known pitfalls

* `Derivation.leibniz : D (a * b) = a • D b + b • D a`; convert with `smul_eq_mul`. `Derivation.map_sum`,
  `Derivation.map_smul`, `map_add`, `map_sub`, `map_neg` are available via the `LinearMap` coercion.
* Evaluating `![x, y, z] 1` and `… 2`: `simp` needs `Matrix.cons_val_zero`, `Matrix.cons_val_one`,
  `Matrix.head_cons`, and for index 2 `Matrix.cons_val_two` (or `Matrix.cons_val_succ`). If `simp`
  stalls, use `Fin.sum_univ_three` and `show` the explicit components.
* `fin_cases i` after `funext i` works when `i : Fin 3` is a local hypothesis-free variable.
* `(2:ℝ)⁻¹ • x` in `R`: `Algebra.smul_def`, `map_inv₀`, `map_ofNat`. Alternative that avoids all of
  this: prove `2 • gradHalfSq b = grad (dot b b)` (with `2 • x = x + x`, `two_smul`) and state
  `identity_one` in the `2 •`-cleared form as well; keep the ½ form as a corollary over ℝ-algebras.
* `Finset.sum_comm` for swapping `∑ i, ∑ j`; `Finset.mul_sum` / `Finset.sum_mul` for pulling factors.
* Do not introduce `Classical.choice` through `Decidable` instances unnecessarily; it is allowed, but
  keep the axiom list to the standard three.

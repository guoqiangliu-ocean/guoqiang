/-!
# The sub-filter energy identity (Part 1)

Formal statement and proof of

    Av( u·(s × curl u) ) − s·divT( Av(u ⊗ u) ) = − s·grad( ½ Av(u·u) )          (1)

in a purely algebraic setting:

* `R` a commutative ring and `ℝ`-algebra ("fields");
* `d i`, `i : Fin 3`, three `ℝ`-linear derivations of `R` (the partial derivatives).
  Only the Leibniz rule is used; the derivations are **never** assumed to commute;
* `Av : R →ₗ[ℝ] R` a linear averaging operator with two axioms,
  `Av_comm` (it commutes with every `d i`) and `Av_res` (resolved fields pass through it);
* `resolved` a subalgebra of `R`, the "filtered-scale" fields.

Hypotheses of the main theorem `identity_one`: `div u = 0` and `∀ i, s i ∈ resolved`.
Neither `Av u = 0` nor idempotence of `Av` is assumed anywhere in Steps 1–4; idempotence is
used only in `leonard_vanish`.

Each theorem's docstring names the Step of the note it formalises and the axioms it uses.
-/
import Mathlib.RingTheory.Derivation.Basic
import Mathlib.Algebra.Algebra.Subalgebra.Basic
import Mathlib.Data.Fin.VecNotation
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.LinearCombination

open scoped BigOperators

namespace Subfilter

/-- Three derivations and an averaging operator on a commutative `ℝ`-algebra `R`.
This structure **is** the hypothesis list of the identity. -/
structure Setup (R : Type*) [CommRing R] [Algebra ℝ R] where
  /-- the three partial derivatives -/
  d        : Fin 3 → Derivation ℝ R R
  /-- the filter / average -/
  Av       : R →ₗ[ℝ] R
  /-- `Av` commutes with every derivative -/
  Av_comm  : ∀ (i : Fin 3) (f : R), Av (d i f) = d i (Av f)
  /-- the resolved (filtered-scale) fields -/
  resolved : Subalgebra ℝ R
  /-- resolved fields pass through the average -/
  Av_res   : ∀ s ∈ resolved, ∀ f : R, Av (s * f) = s * Av f

variable {R : Type*} [CommRing R] [Algebra ℝ R]

/-- Vector fields: triples of ring elements. -/
abbrev V (R : Type*) := Fin 3 → R

/-- Second-rank tensor fields. -/
abbrev T (R : Type*) := Fin 3 → Fin 3 → R

section algebra_only
/-! ### Vector algebra that does not involve the derivations. -/

/-- Scalar product `∑ i, a i * b i`. -/
def dot (a b : V R) : R := ∑ i, a i * b i

/-- Cross product, explicit components. -/
def cross (a b : V R) : V R :=
  ![a 1 * b 2 - a 2 * b 1, a 2 * b 0 - a 0 * b 2, a 0 * b 1 - a 1 * b 0]

/-- Outer product `(a ⊗ b) i j = a i * b j`. -/
def outer (a b : V R) : T R := fun i j => a i * b j

/-- (A1) a·(b×c) = b·(c×a). Pure commutative-ring algebra; uses nothing from `Setup`. -/
theorem dot_cross_cyclic (a b c : V R) : dot a (cross b c) = dot b (cross c a) := by
  simp [dot, cross, Fin.sum_univ_three]; ring

/-- (A2) a·(b×c) = −a·(c×b). Pure commutative-ring algebra; uses nothing from `Setup`. -/
theorem dot_cross_swap (a b c : V R) : dot a (cross b c) = - dot a (cross c b) := by
  simp [dot, cross, Fin.sum_univ_three]; ring

/-- `outer u u` is symmetric. -/
theorem outer_self_symm (u : V R) (i j : Fin 3) : outer u u i j = outer u u j i := by
  simp [outer, mul_comm]

end algebra_only

variable (S : Setup R)

/-! ### Differential operators built from the derivations. -/

/-- `(grad f) i = d i f`. -/
def grad (f : R) : V R := fun i => S.d i f
/-- `div u = ∑ i, d i (u i)`. -/
def div  (u : V R) : R := ∑ i, S.d i (u i)
/-- `curl u`, explicit components. -/
def curl (u : V R) : V R :=
  ![S.d 1 (u 2) - S.d 2 (u 1), S.d 2 (u 0) - S.d 0 (u 2), S.d 0 (u 1) - S.d 1 (u 0)]
/-- `(a·∇)b`. -/
def adv  (a b : V R) : V R := fun i => ∑ j, a j * S.d j (b i)
/-- `(∇·T)_i = ∑ j, ∂_j T_ij`. -/
def divT (Tt : T R) : V R := fun i => ∑ j, S.d j (Tt i j)
/-- Componentwise average of a vector field. -/
def AvV (u : V R) : V R := fun i => S.Av (u i)
/-- Componentwise average of a tensor field. -/
def AvT (Tt : T R) : T R := fun i j => S.Av (Tt i j)
/-- `(b_j ∂_i b_j)_i`, i.e. `½ ∇|b|²` written without the `½`. -/
def gradHalfSq (b : V R) : V R := fun i => ∑ j, b j * S.d i (b j)

/-! ### Step 1 -/

/-- **Step 1.** `b × (∇×b) = (b_j ∂_i b_j)_i − (b·∇)b`, pointwise, for any `b`.
Uses: nothing (the ε–δ identity in components; no Leibniz rule, no axiom of `Setup`). -/
theorem step1 (b : V R) : cross b (S.curl b) = S.gradHalfSq b - S.adv b b := by
  funext i
  fin_cases i <;>
    simp [cross, curl, gradHalfSq, adv, Fin.sum_univ_three] <;> ring

/-- **Step 1'.** Leibniz for a square: `∂_i (f f) = 2 f ∂_i f`.  Uses: Leibniz. -/
theorem d_sq (i : Fin 3) (f : R) : S.d i (f * f) = 2 * (f * S.d i f) := by
  rw [Derivation.leibniz]; simp only [smul_eq_mul]; ring

/-- `(2:ℝ)⁻¹ • (2 * x) = x` in an `ℝ`-algebra. -/
theorem half_smul_two_mul (x : R) : (2:ℝ)⁻¹ • ((2:R) * x) = x := by
  rw [two_mul, ← two_smul ℝ x, smul_smul, inv_mul_cancel₀ two_ne_zero, one_smul]

/-- **Step 1'** (cleared form). `2 • (b_j ∂_i b_j)_i = ∇(b·b)`.  Uses: Leibniz. -/
theorem two_smul_gradHalfSq (b : V R) : (2:ℝ) • S.gradHalfSq b = S.grad (dot b b) := by
  funext i
  simp only [Pi.smul_apply, gradHalfSq, grad, dot, map_sum, d_sq, Finset.smul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [two_mul, two_smul]

/-- **Step 1'.** `(b_j ∂_i b_j)_i = ½ ∇(b·b)`.  Uses: Leibniz. -/
theorem gradHalfSq_eq (b : V R) : S.gradHalfSq b = fun i => (2:ℝ)⁻¹ • S.d i (dot b b) := by
  funext i
  simp only [gradHalfSq, dot, map_sum, d_sq, Finset.smul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [half_smul_two_mul]

/-! ### Step 2 -/

/-- **Step 2.** `div u = 0 ⇒ (u·∇)u = ∇·(u ⊗ u)`.  Uses: Leibniz, `div u = 0`. -/
theorem step2 (u : V R) (hu : S.div u = 0) : S.adv u u = S.divT (outer u u) := by
  funext i
  simp only [adv, divT, outer, Derivation.leibniz, smul_eq_mul, Finset.sum_add_distrib]
  have h1 : ∑ j, u i * S.d j (u j) = u i * S.div u := by
    rw [div, Finset.mul_sum]
  rw [h1, hu, mul_zero, zero_add]

/-! ### Pointwise form of (1), before averaging -/

/-- **(P)** `u·(s×curl u) = s·divT(u⊗u) − s·gradHalfSq u`, pointwise.
Uses: Leibniz, `div u = 0` (through Step 2). No axiom on `Av`. -/
theorem pointwise (u s : V R) (hu : S.div u = 0) :
    dot u (cross s (S.curl u)) = dot s (S.divT (outer u u)) - dot s (S.gradHalfSq u) := by
  -- u·(s×ω) = s·(ω×u) = −s·(u×ω) = −s·[(b_j∂_ib_j) − (u·∇)u] = s·(u·∇)u − s·(b_j∂_ib_j)
  rw [dot_cross_cyclic, dot_cross_swap, step1, ← step2 S u hu]
  simp only [dot, Pi.sub_apply, mul_sub, Finset.sum_sub_distrib]
  ring

/-! ### Step 3: the average. Only `Av_comm` and `Av_res` are used. -/

/-- **Step 3a.** `s` resolved ⇒ `Av(s·X) = s·AvV X`.  Uses: linearity of `Av`, `Av_res`. -/
theorem Av_dot_res (s X : V R) (hs : ∀ i, s i ∈ S.resolved) :
    S.Av (dot s X) = dot s (S.AvV X) := by
  simp only [dot, map_sum, AvV]
  exact Finset.sum_congr rfl fun i _ => S.Av_res (s i) (hs i) (X i)

/-- **Step 3b.** `AvV (divT T) = divT (AvT T)`.  Uses: linearity of `Av`, `Av_comm`. -/
theorem AvV_divT (Tt : T R) : S.AvV (S.divT Tt) = S.divT (S.AvT Tt) := by
  funext i; simp only [AvV, divT, AvT, map_sum, S.Av_comm]

/-- **Step 3c.** `AvV (gradHalfSq u) = ½ ∇ (Av (u·u))`.  Uses: Leibniz (Step 1'), linearity, `Av_comm`. -/
theorem AvV_gradHalfSq (u : V R) :
    S.AvV (S.gradHalfSq u) = fun i => (2:ℝ)⁻¹ • S.d i (S.Av (dot u u)) := by
  funext i
  rw [AvV, gradHalfSq_eq]
  simp only [LinearMap.map_smul, S.Av_comm]

/-- **The identity (1)** (Main theorem), divided by ρ, with `k_t = ½ Av(u·u)`:

    Av(u·(s×curl u)) − s·divT(Av(u⊗u)) = − s·grad k_t.

Uses: Leibniz, `div u = 0`, `s` resolved, linearity of `Av`, `Av_comm`, `Av_res`.
Does **not** use `Av u = 0`, idempotence of `Av`, or commutation of the derivations. -/
theorem identity_one (u s : V R) (hu : S.div u = 0) (hs : ∀ i, s i ∈ S.resolved) :
    S.Av (dot u (cross s (S.curl u))) - dot s (S.divT (S.AvT (outer u u)))
      = - dot s (S.grad ((2:ℝ)⁻¹ • S.Av (dot u u))) := by
  rw [pointwise S u s hu, map_sub, Av_dot_res S _ _ hs, Av_dot_res S _ _ hs,
      AvV_divT, AvV_gradHalfSq]
  simp only [grad, dot, Derivation.map_smul]
  ring

/-- The identity (1) in the `2 •`-cleared form, free of the scalar `½`:

    2 • Av(u·(s×curl u)) − 2 • s·divT(Av(u⊗u)) = − s·grad (Av (u·u)).

Same hypotheses as `identity_one`. -/
theorem identity_one_two_smul (u s : V R) (hu : S.div u = 0) (hs : ∀ i, s i ∈ S.resolved) :
    (2:ℝ) • S.Av (dot u (cross s (S.curl u))) - (2:ℝ) • dot s (S.divT (S.AvT (outer u u)))
      = - dot s (S.grad (S.Av (dot u u))) := by
  rw [← smul_sub, identity_one S u s hu hs]
  simp only [grad, dot, Derivation.map_smul, smul_neg, Finset.smul_sum, mul_smul_comm,
    smul_smul, mul_inv_cancel₀ (two_ne_zero' ℝ), one_smul]

/-! ### Step 4: the three-term split and the symmetric × antisymmetric cancellation -/

/-- **Step 4a.** `R_ij` symmetric ⇒ `∑ R_ij (∂_i s_j − ∂_j s_i) = 0`.
Uses: nothing but commutative-ring algebra (no Leibniz, no axiom of `Setup`). -/
theorem sym_antisym (Rt : T R) (hsym : ∀ i j, Rt i j = Rt j i) (s : V R) :
    ∑ i, ∑ j, Rt i j * (S.d i (s j) - S.d j (s i)) = 0 := by
  simp only [Fin.sum_univ_three]
  have h01 := hsym 0 1; have h02 := hsym 0 2; have h12 := hsym 1 2
  rw [h01, h02, h12]; ring

/-- Expansion of the flux term: `div (u (s·u)) = ∑ u_i s_j ∂_i u_j + ∑ u_i u_j ∂_i s_j`
when `div u = 0`.  Uses: Leibniz, `div u = 0`. -/
theorem div_flux (u s : V R) (hu : S.div u = 0) :
    S.div (fun i => u i * dot s u)
      = ∑ i, ∑ j, u i * s j * S.d i (u j) + ∑ i, ∑ j, u i * u j * S.d i (s j) := by
  have e : ∀ i, S.d i (u i * dot s u)
      = ∑ j, (u i * (s j * S.d i (u j)) + u i * (u j * S.d i (s j))) + dot s u * S.d i (u i) := by
    intro i
    rw [Derivation.leibniz]
    simp only [smul_eq_mul, dot, map_sum, Derivation.leibniz, Finset.mul_sum, mul_add]
  simp only [div, e, Finset.sum_add_distrib]
  rw [← Finset.mul_sum, show (∑ i, S.d i (u i)) = S.div u from rfl, hu, mul_zero, add_zero]
  congr 1 <;> exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring

/-- `s·divT(u⊗u) = ∑ u_i s_j ∂_i u_j` when `div u = 0`.  Uses: Leibniz, `div u = 0`. -/
theorem dot_divT_outer (u s : V R) (hu : S.div u = 0) :
    dot s (S.divT (outer u u)) = ∑ i, ∑ j, u i * s j * S.d i (u j) := by
  have e : ∀ i, ∑ j, S.d j (u i * u j) = u i * S.div u + ∑ j, u j * S.d j (u i) := by
    intro i
    simp only [Derivation.leibniz, smul_eq_mul, Finset.sum_add_distrib, div, Finset.mul_sum]
  simp only [dot, divT, outer, e, hu, mul_zero, zero_add, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring

/-- **Step 4b.** Pointwise three-term split of `u·(s×ω)`:

    u·(s×ω) = div(u (s·u)) − s·gradHalfSq u − ∑ u_i u_j ∂_j s_i,

i.e. flux + Stokes advection + Stokes production (MSM97 / Suzuki–Fox-Kemper).
Uses: Leibniz, `div u = 0` (through (P), `div_flux`, `dot_divT_outer`) and the
symmetric × antisymmetric cancellation `sym_antisym` with `R_ij = u_i u_j`. No axiom on `Av`. -/
theorem step4_split (u s : V R) (hu : S.div u = 0) :
    dot u (cross s (S.curl u))
      = S.div (fun i => u i * dot s u)
        - dot s (S.gradHalfSq u)
        - ∑ i, ∑ j, u i * u j * S.d j (s i) := by
  have hA := sym_antisym S (outer u u) (outer_self_symm u) s
  have hA' : ∑ i, ∑ j, u i * u j * S.d i (s j) - ∑ i, ∑ j, u i * u j * S.d j (s i) = 0 := by
    rw [← hA]
    simp only [outer, mul_sub, Finset.sum_sub_distrib]
  rw [pointwise S u s hu, div_flux S u s hu, dot_divT_outer S u s hu]
  linear_combination (-1 : R) * hA'

/-- The averaged three-term split (eq. (s4) of the note):

    Av(u·(s×ω)) = div(AvV(u (s·u))) − s·grad k_t − ∑ Av(u_i u_j) ∂_j s_i.

Uses: everything `step4_split` uses, plus linearity of `Av`, `Av_comm`, `Av_res`, and
**additionally** that the derivatives `∂_j s_i` are resolved (`hs'`): without it the Stokes
production `Av(u_i u_j ∂_j s_i)` cannot be written as `Av(u_i u_j) ∂_j s_i`. Note that (1)
itself (`identity_one`) does not need `hs'`. -/
theorem step4_split_av (u s : V R) (hu : S.div u = 0) (hs : ∀ i, s i ∈ S.resolved)
    (hs' : ∀ i j, S.d j (s i) ∈ S.resolved) :
    S.Av (dot u (cross s (S.curl u)))
      = S.div (S.AvV (fun i => u i * dot s u))
        - dot s (S.grad ((2:ℝ)⁻¹ • S.Av (dot u u)))
        - ∑ i, ∑ j, S.Av (u i * u j) * S.d j (s i) := by
  rw [step4_split S u s hu, map_sub, map_sub, Av_dot_res S _ _ hs, AvV_gradHalfSq]
  simp only [div, AvV, map_sum, S.Av_comm, grad, dot, Derivation.map_smul]
  congr 1
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  rw [mul_comm (u i * u j), S.Av_res _ (hs' i j), mul_comm]

/-! ### Leonard decomposition -/

/-- **Leonard (L1).** For ANY linear `Av`:
`Av(u_i u_j) − U_i U_j = L_ij + C_ij + Av(u'_i u'_j)` with `U = Av u`, `u' = u − U`,
`L = Av(U_i U_j) − U_i U_j`, `C = Av(U_i u'_j) + Av(u'_i U_j)`.
Uses: linearity of `Av` only (no `Setup` axiom, no Leibniz). -/
theorem leonard (Av : R →ₗ[ℝ] R) (u : V R) (i j : Fin 3) :
    Av (u i * u j) - Av (u i) * Av (u j)
      = (Av (Av (u i) * Av (u j)) - Av (u i) * Av (u j))
        + (Av (Av (u i) * (u j - Av (u j))) + Av ((u i - Av (u i)) * Av (u j)))
        + Av ((u i - Av (u i)) * (u j - Av (u j))) := by
  have h : u i * u j
      = Av (u i) * Av (u j) + Av (u i) * (u j - Av (u j))
        + (u i - Av (u i)) * Av (u j) + (u i - Av (u i)) * (u j - Av (u j)) := by ring
  rw [h, map_add, map_add, map_add]; ring

/-- **Leonard (L2).** Under the Reynolds axioms (`Av` idempotent and `Av u` resolved) the
Leonard term and the cross term vanish.  Uses: `Av_res`, idempotence, linearity. -/
theorem leonard_vanish (u : V R) (i j : Fin 3)
    (hidem : ∀ f, S.Av (S.Av f) = S.Av f) (hres : ∀ k, S.Av (u k) ∈ S.resolved) :
    S.Av (S.Av (u i) * S.Av (u j)) - S.Av (u i) * S.Av (u j) = 0 ∧
    S.Av (S.Av (u i) * (u j - S.Av (u j))) + S.Av ((u i - S.Av (u i)) * S.Av (u j)) = 0 := by
  constructor
  · rw [S.Av_res _ (hres i), hidem]; ring
  · rw [S.Av_res _ (hres i), map_sub, hidem, sub_self, mul_zero, zero_add,
        mul_comm (u i - S.Av (u i)) (S.Av (u j)), S.Av_res _ (hres j), map_sub, hidem, sub_self,
        mul_zero]

end Subfilter

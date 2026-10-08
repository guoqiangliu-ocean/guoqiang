import Subfilter.Identity
import Mathlib.Algebra.MvPolynomial.PDeriv

/-!
# A non-vacuous model of the axioms (Part 2)

`R := Fin 2 → MvPolynomial (Fin 3) ℝ`: a two-member ensemble of polynomial fields in `x, y, z`.

* `d i` acts componentwise by `MvPolynomial.pderiv i`;
* `Av f := fun _ => ½ (f 0 + f 1)` is the ensemble mean, broadcast to both members;
* `resolved := { f | f 0 = f 1 }`, the fields that do not depend on the ensemble member.

We prove the two axioms `Av_comm`, `Av_res`, and in addition that `Av` is idempotent and
`Av (f − Av f) = 0`, so this `Av` is a genuine Reynolds operator; and we show the model is
non-degenerate (`resolved ≠ ⊤`, `Av ≠ id`, `d i ≠ 0`). Finally `identity_one` is specialised
to it on an explicit non-trivial `u` and `s`.
-/
open scoped BigOperators

noncomputable section

namespace Subfilter
namespace Model

open Setup

/-- The ring of ensemble fields: two members, each a polynomial in three variables. -/
abbrev R := Fin 2 → MvPolynomial (Fin 3) ℝ

/-- A single polynomial field, as shorthand. -/
abbrev P := MvPolynomial (Fin 3) ℝ

/-- `∂_i` acting componentwise on the ensemble. -/
def piDeriv (i : Fin 3) : Derivation ℝ R R where
  toFun f := fun k => MvPolynomial.pderiv i (f k)
  map_add' f g := by funext k; simp
  map_smul' c f := by funext k; simp
  map_one_eq_zero' := by funext k; simp
  leibniz' f g := by
    funext k
    simp [Derivation.leibniz, smul_eq_mul]

@[simp] theorem piDeriv_apply (i : Fin 3) (f : R) (k : Fin 2) :
    piDeriv i f k = MvPolynomial.pderiv i (f k) := rfl

/-- The ensemble mean, broadcast to both members. -/
def Av : R →ₗ[ℝ] R where
  toFun f := fun _ => (2:ℝ)⁻¹ • (f 0 + f 1)
  map_add' f g := by funext k; simp only [Pi.add_apply]; rw [← smul_add]; congr 1; abel
  map_smul' c f := by
    funext k
    simp only [Pi.smul_apply, RingHom.id_apply, ← smul_add, smul_smul, mul_comm]

@[simp] theorem Av_apply (f : R) (k : Fin 2) : Av f k = (2:ℝ)⁻¹ • (f 0 + f 1) := rfl

/-- The resolved fields: those that do not depend on the ensemble member. -/
def resolved : Subalgebra ℝ R where
  carrier := { f | f 0 = f 1 }
  mul_mem' {f g} (hf : f 0 = f 1) (hg : g 0 = g 1) := by
    change f 0 * g 0 = f 1 * g 1; rw [hf, hg]
  one_mem' := rfl
  add_mem' {f g} (hf : f 0 = f 1) (hg : g 0 = g 1) := by
    change f 0 + g 0 = f 1 + g 1; rw [hf, hg]
  zero_mem' := rfl
  algebraMap_mem' r := rfl

@[simp] theorem mem_resolved (f : R) : f ∈ resolved ↔ f 0 = f 1 := Iff.rfl

/-- Axiom `Av_comm`: the ensemble mean commutes with `∂_i`. -/
theorem Av_comm (i : Fin 3) (f : R) : Av (piDeriv i f) = piDeriv i (Av f) := by
  funext k; simp

/-- Axiom `Av_res`: member-independent fields pass through the ensemble mean. -/
theorem Av_res (s : R) (hs : s ∈ resolved) (f : R) : Av (s * f) = s * Av f := by
  rw [mem_resolved] at hs
  funext k
  simp only [Av_apply, Pi.mul_apply, smul_add, mul_add, mul_smul_comm]
  fin_cases k <;> simp [hs]

/-- The model `Setup`. -/
def setup : Setup R where
  d := piDeriv
  Av := Av
  Av_comm := Av_comm
  resolved := resolved
  Av_res := Av_res

/-- `example` required by the brief: the axioms are satisfiable. -/
example : Setup (Fin 2 → MvPolynomial (Fin 3) ℝ) := setup

/-! ### `Av` is a genuine Reynolds operator -/

theorem half_add_self (c : P) : (2:ℝ)⁻¹ • (c + c) = c := by
  rw [← two_smul ℝ c, smul_smul, inv_mul_cancel₀ two_ne_zero, one_smul]

/-- Idempotence `Av ∘ Av = Av`. -/
theorem Av_idem (f : R) : Av (Av f) = Av f := by
  funext k; exact half_add_self _

/-- `Av (f − Av f) = 0`: the fluctuation has zero mean. -/
theorem Av_fluct (f : R) : Av (f - Av f) = 0 := by
  rw [map_sub, Av_idem, sub_self]

/-- The mean of any field is resolved. -/
theorem Av_mem_resolved (f : R) : Av f ∈ resolved := by
  rw [mem_resolved]; rfl

/-! ### Non-degeneracy -/

/-- A member-dependent field: `X 0` on member `0`, `0` on member `1`. -/
def wobble : R := fun k => if k = 0 then MvPolynomial.X 0 else 0

theorem wobble_zero : wobble 0 = MvPolynomial.X 0 := by simp [wobble]
theorem wobble_one : wobble 1 = 0 := by simp [wobble]

/-- `resolved ≠ ⊤`: not every field is resolved. -/
theorem resolved_ne_top : resolved ≠ ⊤ := by
  intro h
  have : wobble ∈ resolved := h ▸ Algebra.mem_top
  rw [mem_resolved, wobble_zero, wobble_one] at this
  exact MvPolynomial.X_ne_zero 0 this

/-- `Av ≠ id`: the average really averages. -/
theorem Av_ne_id : Av ≠ LinearMap.id := by
  intro h
  have h1 : Av wobble 1 = wobble 1 := by rw [h]; rfl
  rw [Av_apply, wobble_zero, wobble_one, add_zero] at h1
  apply MvPolynomial.X_ne_zero (R := ℝ) (0 : Fin 3)
  have h2 : (2:ℝ) • ((2:ℝ)⁻¹ • (MvPolynomial.X 0 : P)) = (2:ℝ) • (0 : P) := by rw [h1]
  rwa [smul_smul, mul_inv_cancel₀ two_ne_zero, one_smul, smul_zero] at h2

/-- `d i ≠ 0`: the derivations are not trivial (`∂_i X_i = 1`). -/
theorem piDeriv_ne_zero (i : Fin 3) : piDeriv i ≠ 0 := by
  intro h
  have : piDeriv i (fun _ => MvPolynomial.X i) 0 = 1 := by simp
  rw [h] at this
  simp at this

/-! ### Specialising `identity_one` to the model -/

/-- A solenoidal, member-dependent sub-filter field with zero mean:
`u = (X 1, 0, 0)` on member 0 and `(−X 1, 0, 0)` on member 1. -/
def uModel : V R :=
  ![fun k => if k = 0 then MvPolynomial.X 1 else -MvPolynomial.X 1, 0, 0]

/-- A resolved, non-constant Stokes drift `s = (X 2, 0, 0)` (broadcast to both members). -/
def sModel : V R := ![fun _ => MvPolynomial.X 2, 0, 0]

theorem uModel_div : setup.div uModel = 0 := by
  funext k
  fin_cases k <;> simp [Setup.div, Fin.sum_univ_three, setup, uModel]

theorem sModel_resolved : ∀ i, sModel i ∈ setup.resolved := by
  intro i; fin_cases i <;> simp [sModel, setup]

theorem uModel_ne_zero : uModel ≠ 0 := by
  intro h
  have := congrFun (congrFun h 0) 0
  simp [uModel] at this

/-- The identity (1) holds in the model on `uModel`, `sModel`. -/
theorem identity_one_model :
    setup.Av (dot uModel (cross sModel (setup.curl uModel)))
        - dot sModel (setup.divT (setup.AvT (outer uModel uModel)))
      = - dot sModel (setup.grad ((2:ℝ)⁻¹ • setup.Av (dot uModel uModel))) :=
  identity_one setup uModel sModel uModel_div sModel_resolved

end Model
end Subfilter

end

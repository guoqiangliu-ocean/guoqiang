import Mathlib.Analysis.SpecialFunctions.ImproperIntegrals
import Mathlib.MeasureTheory.Integral.IntervalIntegral.IntegrationByParts

/-!
# The analytic steps (Part 3, optional)

* **Step 5.** The Doppler weight `N k · 2k e^{2kz}` of a deep-water wave equals its Stokes
  drift `a² σ k e^{2kz}` (pure algebra from `σ² = g k`, `N = g a² / 2σ`), and the weight
  `2k e^{2kz}` integrates to `1` over `(-∞, 0]`.
* **Step 6.** Integration by parts on `[-h, 0]` with `τ(-h) = 0`:
  `−∫ uˢ ∂_z τ̄ = ∫ τ̄ ∂_z uˢ − τ̄(0) uˢ(0)`, i.e. `−∫ uˢ ∂_z τ̄ = P_S − W_S`.
-/
open MeasureTheory Set

namespace Subfilter
namespace Analytic

/-! ### Step 5 -/

/-- **Step 5 (algebra).** With `σ > 0`, `σ² = g k` and `N = g a² / (2σ)`,
`N k (2k e^{2kz}) = a² σ k e^{2kz}`: the Doppler weight of the action is the Stokes drift. -/
theorem step5_weight (σ g k a z N : ℝ) (hσ : 0 < σ) (hdisp : σ ^ 2 = g * k)
    (hN : N = g * a ^ 2 / (2 * σ)) :
    N * k * (2 * k * Real.exp (2 * k * z)) = a ^ 2 * σ * k * Real.exp (2 * k * z) := by
  have hσ' : σ ≠ 0 := hσ.ne'
  have key : a ^ 2 * σ * k * Real.exp (2 * k * z)
      = a ^ 2 * σ ^ 2 * k * Real.exp (2 * k * z) / σ := by
    field_simp
    ring
  rw [key, hdisp, hN]
  field_simp
  ring

/-- **Step 5 (normalisation).** For `k > 0`, `∫_{-∞}^0 2k e^{2kz} dz = 1`. -/
theorem step5_weight_integral (k : ℝ) (hk : 0 < k) :
    ∫ z in Iic (0 : ℝ), 2 * k * Real.exp (2 * k * z) = 1 := by
  have h2k : 0 < 2 * k := by positivity
  rw [integral_const_mul, integral_exp_mul_Iic h2k 0]
  field_simp
  simp

/-! ### Step 6 -/

/-- **Step 6 (integration by parts on `[-h, 0]`).** If `τ` and `uS` are differentiable on
`[-h, 0]` with interval-integrable derivatives `τ'`, `uS'`, and `τ (-h) = 0`, then

    −∫_{-h}^0 uS τ' = ∫_{-h}^0 τ uS' − τ 0 · uS 0 .

In the note: `−∫ uˢ ∂_z τ̄ dz = P_S − W_S`. -/
theorem step6_parts (h : ℝ) (τ uS τ' uS' : ℝ → ℝ)
    (hτ : ∀ x ∈ uIcc (-h) 0, HasDerivAt τ (τ' x) x)
    (huS : ∀ x ∈ uIcc (-h) 0, HasDerivAt uS (uS' x) x)
    (hτ' : IntervalIntegrable τ' volume (-h) 0)
    (huS' : IntervalIntegrable uS' volume (-h) 0)
    (hbot : τ (-h) = 0) :
    -(∫ z in (-h)..0, uS z * τ' z) = (∫ z in (-h)..0, τ z * uS' z) - τ 0 * uS 0 := by
  rw [intervalIntegral.integral_mul_deriv_eq_deriv_mul huS hτ huS' hτ', hbot]
  have hc : (∫ z in (-h)..0, uS' z * τ z) = ∫ z in (-h)..0, τ z * uS' z := by
    simp only [mul_comm]
  rw [hc]
  ring

/-- **Step 6**, stated with `deriv` and continuity of the derivatives on `[-h, 0]`
(the hypotheses of the brief): `τ`, `uS` differentiable on `[-h, 0]` with continuous
derivatives there, `τ (-h) = 0`. -/
theorem step6_parts_deriv (h : ℝ) (τ uS : ℝ → ℝ)
    (hτ : ∀ x ∈ uIcc (-h) 0, DifferentiableAt ℝ τ x)
    (huS : ∀ x ∈ uIcc (-h) 0, DifferentiableAt ℝ uS x)
    (hτc : ContinuousOn (deriv τ) (uIcc (-h) 0))
    (huSc : ContinuousOn (deriv uS) (uIcc (-h) 0))
    (hbot : τ (-h) = 0) :
    -(∫ z in (-h)..0, uS z * deriv τ z)
      = (∫ z in (-h)..0, τ z * deriv uS z) - τ 0 * uS 0 :=
  step6_parts h τ uS (deriv τ) (deriv uS)
    (fun x hx => (hτ x hx).hasDerivAt) (fun x hx => (huS x hx).hasDerivAt)
    hτc.intervalIntegrable huSc.intervalIntegrable hbot

end Analytic
end Subfilter

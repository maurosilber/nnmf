# Binlets for NNMF unmixing — findings

Slides: https://claude.ai/artifact/JfXFQuYfpoFDvFH9pTE1Tq (figures from `figures.py`).
Reproduce: `pixi run experiments` (writes `results/`), then `pixi run figures`.

1. **Known spectra (exp. 1).** KL updates reach the Cramér–Rao bound (CRB) of the
   fractions; LS plateaus 2–20 % above it (more with spectral overlap). Iterations
   needed grow with overlap (~10 at Δλ = 150 nm, ~100 at 25 nm).
2. **Limits of the linear error model (exp. 2).** At f = 0 the plug-in error is ~2×
   too large and estimates are biased (+0.4 σ); at N ≤ 100 and f = 0.1 the true error is
   up to 1.4× the predicted one. With 6 overlapping spectra std is 0.5–0.7 × CRB.
   Motivates the pseudocount in the binlets test.
3. **Blind (exp. 3).** Without pure pixels absolute fractions keep a systematic error
   (~4 × CRB) that does not shrink with more pixels, but the difference between pixels
   with equal fractions stays calibrated (std z ≈ 1.0 from 200 pixels, KL). The binlets
   test only uses differences.
4. **Partially known spectra (exp. 4).** On mixed-only data, fixing the two outer spectra
   (K = 3) brings the worst error from 2.5× to 1.2× CRB.
5. **Binlets (exp. 5).** The test is calibrated (4.1–4.9 % rejections at α = 5 %, N = 10–1000).
   α = 0.01 gives the lowest error in flat regions (4.6× lower than raw at N = 100) and at
   edges (3× lower than the best box filter); a 7×7 box filter is better on smooth gradients.
6. **Bug fixed.** Binlets' non-linear reconstruction returns negative counts (95 % of pixels
   at N = 20, α = 0.2), which made the KL unmixing of a few pixels chaotic (f₁ down to −9.5).
   `binlets_unmix` now clips the denoised counts at 0.
7. **Camera noise, Gaussian approximation (exp. 6).** Model: Poisson → gain → bias →
   read noise → ADC. Quantization adds 1/12 ADU² of variance, and floor shifts the offset
   by −½ ADU. With read noise σ_r ≥ 0.5 ADU the mean and variance match within 2.5 %.
   Without the 1/12 term, dark-pixel variance is 30 % (σ_r = 0.5) or 8 % (σ_r = 1) too low.
   Unmixing photoelectrons with variance W·H + background (reweighted LS, `unmix(...,
   background=)`) reaches the extended bound with no bias, and the plug-in error is
   calibrated from N = 100. Read noise in 100 channels costs a lot at low N: at σ = 3 e⁻
   the bound is 3.4× the Poisson one at N = 100. Clipping electrons at 0 and running KL is
   biased (+0.3 to +0.7 bound for N ≤ 1000). The approximation breaks at N = 30, σ = 3 e⁻
   (bound 0.7). `least_squares_guess` now clips at ≥ 0, because totals can be negative.

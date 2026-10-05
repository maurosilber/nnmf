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

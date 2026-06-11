//! Small numerical helpers: standard-normal CDF and quantile.
//!
//! Self-contained approximations instead of a stats dependency — keeps the
//! core lean and WASM-clean. Accuracy is far beyond what the stochastic
//! fitters using them (ES-RUM) can resolve.

/// Standard normal CDF via Abramowitz & Stegun 7.1.26 (|err| ≤ 1.5e-7).
pub fn norm_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs() / std::f64::consts::SQRT_2);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = 1.0 - poly * (-(x * x) / 2.0).exp();
    if x >= 0.0 {
        0.5 * (1.0 + erf)
    } else {
        0.5 * (1.0 - erf)
    }
}

/// Standard normal quantile (inverse CDF) via Acklam's rational
/// approximation (relative error < 1.2e-8 over (0, 1)).
pub fn norm_ppf(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "norm_ppf domain is (0, 1), got {p}");

    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_reference_values() {
        for (x, want) in [
            (0.0, 0.5),
            (1.0, 0.8413447460685429),
            (-1.0, 0.15865525393145707),
            (1.959963984540054, 0.975),
            (-2.5, 0.006209665325776132),
        ] {
            assert!(
                (norm_cdf(x) - want).abs() < 2e-7,
                "cdf({x}) = {} vs {want}",
                norm_cdf(x)
            );
        }
    }

    #[test]
    fn ppf_inverts_cdf() {
        for p in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 0.999] {
            let x = norm_ppf(p);
            assert!(
                (norm_cdf(x) - p).abs() < 1e-6,
                "round trip at {p}: {}",
                norm_cdf(x)
            );
        }
        assert!((norm_ppf(0.975) - 1.959964).abs() < 1e-5);
    }
}

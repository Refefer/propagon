//! Sparse symmetric linear solves for the least-squares raters
//! (Massey, Colley, HodgeRank).
//!
//! Conjugate gradients on a row-compressed sparse matrix — no external
//! linear-algebra dependency. Assumes the matrix is symmetric and either
//! positive definite (Colley) or positive semi-definite with the constant
//! vector as its only kernel (graph Laplacians: Massey, HodgeRank); for the
//! latter, [`SparseSymmetric::solve_mean_zero`] projects the constant
//! component out of every iterate, which both restores convergence and pins
//! the solution to the canonical mean-zero representative.
//!
//! Gotcha: CG on an indefinite or asymmetric matrix silently diverges; the
//! callers in `algos/` only ever build the two shapes above.

use crate::error::{Error, Result};

/// Row-compressed symmetric sparse matrix: `rows[i]` lists `(j, a_ij)`.
/// Only the caller guarantees symmetry; entries are stored redundantly
/// (both `(i, j)` and `(j, i)`).
pub(crate) struct SparseSymmetric {
    rows: Vec<Vec<(u32, f64)>>,
}

impl SparseSymmetric {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            rows: vec![Vec::new(); n],
        }
    }

    /// Adds `v` to `a_ij` (and, when `i != j`, to `a_ji`).
    pub(crate) fn add(&mut self, i: usize, j: usize, v: f64) {
        self.rows[i].push((j as u32, v));

        if i != j {
            self.rows[j].push((i as u32, v));
        }
    }

    /// Merges duplicate entries; call once after assembly.
    pub(crate) fn compress(&mut self) {
        for row in &mut self.rows {
            row.sort_unstable_by_key(|e| e.0);
            let mut out: Vec<(u32, f64)> = Vec::with_capacity(row.len());

            for &(j, v) in row.iter() {
                match out.last_mut() {
                    Some(last) if last.0 == j => last.1 += v,
                    _ => out.push((j, v)),
                }
            }
            *row = out;
        }
    }

    fn matvec(&self, x: &[f64], out: &mut [f64]) {
        for (i, row) in self.rows.iter().enumerate() {
            out[i] = row.iter().map(|&(j, v)| v * x[j as usize]).sum();
        }
    }

    /// Conjugate gradients for `Ax = b` on a positive-definite system.
    pub(crate) fn solve(&self, b: &[f64], iterations: usize, tolerance: f64) -> Result<Vec<f64>> {
        self.cg(b, iterations, tolerance, false)
    }

    /// Conjugate gradients for a PSD system whose kernel is the constant
    /// vector (a graph Laplacian): both `b` and every iterate are projected
    /// onto the mean-zero subspace, yielding the mean-zero solution.
    pub(crate) fn solve_mean_zero(
        &self,
        b: &[f64],
        iterations: usize,
        tolerance: f64,
    ) -> Result<Vec<f64>> {
        self.cg(b, iterations, tolerance, true)
    }

    fn cg(&self, b: &[f64], iterations: usize, tolerance: f64, center: bool) -> Result<Vec<f64>> {
        let n = b.len();
        let mut b = b.to_vec();

        if center {
            demean(&mut b);
        }

        let b_norm = dot(&b, &b).sqrt().max(f64::MIN_POSITIVE);
        let mut x = vec![0.0; n];
        let mut r = b; // r = b - Ax with x = 0
        let mut p = r.clone();
        let mut ap = vec![0.0; n];
        let mut rr = dot(&r, &r);

        for _ in 0..iterations {
            if rr.sqrt() <= tolerance * b_norm {
                return Ok(x);
            }

            self.matvec(&p, &mut ap);

            if center {
                demean(&mut ap);
            }

            let pap = dot(&p, &ap);
            if !pap.is_finite() || pap <= 0.0 {
                return Err(Error::Numeric(format!(
                    "conjugate gradients hit a non-positive curvature ({pap}); \
                     the system is not positive (semi-)definite"
                )));
            }

            let alpha = rr / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            let rr_next = dot(&r, &r);
            let beta = rr_next / rr;
            rr = rr_next;

            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
        }

        if rr.sqrt() <= tolerance * b_norm * 1e3 {
            // Close enough to be useful; callers treat tolerance as a target,
            // not a guarantee, and log the residual.
            log::debug!("cg stopped at iteration cap, residual {:.3e}", rr.sqrt());
            return Ok(x);
        }

        Err(Error::Numeric(format!(
            "conjugate gradients failed to converge: residual {:.3e} after {iterations} iterations",
            rr.sqrt()
        )))
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn demean(v: &mut [f64]) {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    v.iter_mut().for_each(|x| *x -= mean);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SPD 3x3 hand-solve: A = [[4,1,0],[1,3,1],[0,1,2]], b = [1,2,3];
    /// elimination gives x = (2/9, 1/9, 13/9).
    #[test]
    fn cg_matches_dense_solve() {
        let mut a = SparseSymmetric::new(3);
        a.add(0, 0, 4.0);
        a.add(0, 1, 1.0);
        a.add(1, 1, 3.0);
        a.add(1, 2, 1.0);
        a.add(2, 2, 2.0);
        a.compress();

        let x = a.solve(&[1.0, 2.0, 3.0], 100, 1e-12).unwrap();
        let want = [2.0 / 9.0, 1.0 / 9.0, 13.0 / 9.0];
        for (got, want) in x.iter().zip(want) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
    }

    /// Path-graph Laplacian (kernel = constants): L = [[1,-1,0],[-1,2,-1],
    /// [0,-1,1]], b = (-1, 0, 1) → mean-zero solution (-1, 0, 1).
    #[test]
    fn mean_zero_solve_on_laplacian() {
        let mut l = SparseSymmetric::new(3);
        l.add(0, 0, 1.0);
        l.add(0, 1, -1.0);
        l.add(1, 1, 2.0);
        l.add(1, 2, -1.0);
        l.add(2, 2, 1.0);
        l.compress();

        let x = l.solve_mean_zero(&[-1.0, 0.0, 1.0], 100, 1e-12).unwrap();
        let want = [-1.0, 0.0, 1.0];
        for (got, want) in x.iter().zip(want) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
    }

    #[test]
    fn duplicate_entries_compress() {
        let mut a = SparseSymmetric::new(2);
        a.add(0, 0, 1.0);
        a.add(0, 0, 1.0);
        a.add(0, 1, -0.5);
        a.add(0, 1, -0.5);
        a.add(1, 1, 2.0);
        a.compress();

        let mut out = vec![0.0; 2];
        a.matvec(&[1.0, 1.0], &mut out);
        assert_eq!(out, vec![1.0, 1.0]);
    }
}

//! Thin parallelism shim.
//!
//! With the default `parallel` feature these helpers fan out via rayon; with
//! the feature off (the single-threaded WASM build) the same call sites
//! compile to sequential loops. Algorithm code uses only these helpers, never
//! rayon directly, so the two builds share one implementation.

/// Maps `0..n` to a `Vec<R>`, in parallel when available.
pub fn par_map_indexed<R, F>(n: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..n).map(f).collect()
    }
}

/// Applies `f(index, item)` to every element of a mutable slice.
pub fn par_for_each_mut<T, F>(items: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        items.par_iter_mut().enumerate().for_each(|(i, t)| f(i, t));
    }
    #[cfg(not(feature = "parallel"))]
    {
        items.iter_mut().enumerate().for_each(|(i, t)| f(i, t));
    }
}

/// Unstable sort, parallel when available.
pub fn par_sort_by<T, F>(items: &mut [T], cmp: F)
where
    T: Send,
    F: Fn(&T, &T) -> std::cmp::Ordering + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        items.par_sort_unstable_by(cmp);
    }
    #[cfg(not(feature = "parallel"))]
    {
        items.sort_unstable_by(cmp);
    }
}

/// Runs `f` inside the supplied rayon pool when one is given (and the
/// `parallel` feature is on); otherwise runs it directly.
pub fn run_scoped<R: Send>(opts: &crate::FitOptions<'_>, f: impl FnOnce() -> R + Send) -> R {
    #[cfg(feature = "parallel")]
    {
        match opts.pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = opts;
        f()
    }
}

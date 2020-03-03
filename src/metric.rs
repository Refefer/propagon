pub trait Metric: Send + Sync {
    // Computes the distance between two points
    #[inline] 
    fn distance(&self, x: &[f32], y: &[f32]) -> f32;

    // Normalizes the vectors, if necessary
    #[inline] 
    fn normalize(&self, x: &mut [f32]);

    // Initialization component score for DE
    #[inline] 
    fn component_range(&self, dims: usize) -> f32; 

    // Adds a score penalty to vectors.  It helps us avoid overflows.
    #[inline] 
    fn in_domain(&self, x: &[f32]) -> bool {
        true
    }

}

// Euclidean distance
pub struct EuclideanSpace;

impl Metric for EuclideanSpace {
    #[inline]
    fn distance(&self, x: &[f32], y: &[f32]) -> f32 {
        x.iter().zip(y.iter())
            .map(|(v1i, v2i)| (v1i - v2i).powi(2))
            .sum::<f32>()
            .powf(0.5)
    }

    fn normalize(&self, x: &mut [f32]) {}

    fn component_range(&self, dims: usize) -> f32 {
        (1f32.powi(2) / (dims as f32)).powf(0.5)
    }
}

// Euclidean distance
pub struct ManhattanSpace;

impl Metric for ManhattanSpace {
    #[inline]
    fn distance(&self, x: &[f32], y: &[f32]) -> f32 {
        x.iter().zip(y.iter())
            .map(|(v1i, v2i)| (v1i - v2i).abs())
            .sum::<f32>()
    }

    fn normalize(&self, x: &mut [f32]) {}

    fn component_range(&self, dims: usize) -> f32 {
        (1f32.powi(2) / (dims as f32)).powf(0.5)
    }
}

// Poincare distance
pub struct PoincareSpace;

impl Metric for PoincareSpace {

    #[inline]
    fn distance(&self, x: &[f32], y: &[f32]) -> f32 {
        let mut xy2 = 0.;
        let mut x2 = 0.;
        let mut y2 = 0.;
        for (xi, yi) in x.iter().zip(y.iter()) {
            xy2 += (xi - yi).powi(2);
            x2 += xi.powi(2);
            y2 += yi.powi(2);
        }

        // normalize by magnitude
        if x2 >= 1. {
            xy2 /= x2.powf(0.5);
            x2 = 1. - 1e-5;
        }
        if y2 >= 1. {
            xy2 /= y2.powf(0.5);
            y2 = 1. - 1e-5;
        }
        let k = (1. + 2. * (xy2 / ((1. - x2) * (1. - y2))));
        k.acosh()
    }

    fn normalize(&self, x: &mut [f32]) {
        let x_norm = x.iter().map(|xi| xi.powi(2)).sum::<f32>().powf(0.5);
        if x_norm > 1. {
            x.iter_mut().for_each(|xi| *xi /= x_norm);
        }
    }

    fn component_range(&self, dims: usize) -> f32 {
        (1. / (dims as f32)).powf(0.5)
    }

    fn in_domain(&self, x: &[f32]) -> bool {
        let norm = x.iter().map(|xi| xi.powi(2)).sum::<f32>().powf(0.5);
        norm <= 1.0
    }
}

// We'll assume a -1 curvature for the time being
pub struct HyperboloidSpace;

impl Metric for HyperboloidSpace {

    #[inline]
    fn distance(&self, x: &[f32], y: &[f32]) -> f32 {
        let mut xy = 0.;
        let mut x2 = 0.;
        let mut y2 = 0.;
        for (xi, yi) in x.iter().zip(y.iter()) {
            xy += xi * yi;
            x2 += xi.powi(2);
            y2 += yi.powi(2);
        }

        let k = ((1. + x2) * (1. + y2)).sqrt() - xy;
        k.acosh()
    }

    fn normalize(&self, x: &mut [f32]) { }

    fn component_range(&self, dims: usize) -> f32 {
        (1. / (dims as f32)).powf(0.5)
    }

    fn in_domain(&self, x: &[f32]) -> bool{
        let max_value = (std::f32::MAX - 1.).powf(0.5) / x.len() as f32;
        x.iter().all(|xi| xi.abs() < max_value)
    }
}

pub enum Space {
    // Euclidean distance
    Euclidean,

    // Manhattan distance
    Manhattan,

    // Hyperbolic space on the Poincare disk
    Poincare,

    // Hyperbolic space in Hyperboloid model
    Hyperboloid
}

impl Metric for Space {
    #[inline]
    fn distance(&self, x: &[f32], y: &[f32]) -> f32 {
        match self {
            Space::Euclidean   => EuclideanSpace.distance(x, y),
            Space::Poincare    => PoincareSpace.distance(x, y),
            Space::Hyperboloid => HyperboloidSpace.distance(x, y),
            Space::Manhattan   => ManhattanSpace.distance(x, y),
        }
    }

    #[inline]
    fn normalize(&self, x: &mut [f32]) {
        match self {
            Space::Euclidean   => EuclideanSpace.normalize(x),
            Space::Poincare    => PoincareSpace.normalize(x),
            Space::Hyperboloid => HyperboloidSpace.normalize(x),
            Space::Manhattan   => ManhattanSpace.normalize(x)
        }
    }

    #[inline]
    fn component_range(&self, dims: usize) -> f32 {
        match self {
            Space::Euclidean   => EuclideanSpace.component_range(dims),
            Space::Poincare    => PoincareSpace.component_range(dims),
            Space::Hyperboloid => HyperboloidSpace.component_range(dims),
            Space::Manhattan   => ManhattanSpace.component_range(dims)
        }
    }

    #[inline]
    fn in_domain(&self, x: &[f32]) -> bool {
        match self {
            Space::Euclidean   => EuclideanSpace.in_domain(x),
            Space::Poincare    => PoincareSpace.in_domain(x),
            Space::Hyperboloid => HyperboloidSpace.in_domain(x),
            Space::Manhattan   => ManhattanSpace.in_domain(x)
        }
    }


}



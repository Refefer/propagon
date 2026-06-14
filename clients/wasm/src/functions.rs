//! Module-level free functions: Wilson intervals and connected components.

use std::cell::RefCell;

use crate::Component;
use crate::bindings::exports::propagon::core::datasets::{GraphDataset, GraphDatasetBorrow};
use crate::bindings::exports::propagon::core::functions::Guest;
use crate::datasets::GraphData;

impl Guest for Component {
    fn wilson_interval(successes: f64, failures: f64, z: f64) -> (f64, f64) {
        propagon::algos::wilson_interval(successes, failures, z)
    }

    fn extract_components(graph: GraphDatasetBorrow<'_>, min_size: u32) -> Vec<GraphDataset> {
        let g = graph.get::<GraphData>();
        let borrowed = g.0.borrow();
        propagon::algos::extract_components(borrowed.view(), min_size as usize)
            .into_iter()
            .map(|inner| GraphDataset::new(GraphData(RefCell::new(inner))))
            .collect()
    }
}

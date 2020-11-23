use indicatif::{ProgressBar,ProgressStyle};

pub fn simple_pb(total_work: u64) -> ProgressBar {
    let pb = ProgressBar::new(total_work);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));
    pb.enable_steady_tick(200);
    pb.set_draw_delta(total_work as u64 / 1000);
    pb
}


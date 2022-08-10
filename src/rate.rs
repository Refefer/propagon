use hashbrown::{HashMap,HashSet};
use super::Games;
use crate::utils::tally_winners_losers;

#[derive(Clone,Copy,Debug)]
pub enum ConfidenceInterval {
    P90,
    P95,
    P50
}

pub struct Rate {
    pub ci: ConfidenceInterval
}

impl Rate {

    pub fn new(ci: ConfidenceInterval) -> Self {
        Rate { ci }
    }

    pub fn compute(&self, games: &Games) -> Vec<(u32, f32)> {
        
        // Compute rate stats
        let (mut winners, losers) = tally_winners_losers(games);
        
        // Get all tgeams.
        let mut all_teams: HashSet<_> = winners.keys().collect();
        all_teams.extend(losers.keys());

        // Just return the rate
        if matches!(self.ci, ConfidenceInterval::P50) {
            // Just return the point estimate
            all_teams.into_iter().map(|team| {
                let l_score = losers.get(team).unwrap_or(&(0, 0.)).1;
                let w_score = winners.get(team).unwrap_or(&(0, 0.)).1;
                (*team, w_score / (w_score + l_score))
            }).collect()

        } else {
            let z = if matches!(self.ci, ConfidenceInterval::P95) {
                1.96
            } else {
                1.645
            };

            let z_sqr = z * z;
            
            let mut scores = Vec::new();
            for team in all_teams.into_iter() {
                let w = winners.get(team).unwrap_or(&(0, 0.)).1;
                let l = losers.get(team).unwrap_or(&(0, 0.)).1;
                let score = wilson_confidence_interval(w, l, z);
                scores.push((*team, score));
            }
            scores.sort_by(|a, b| (b.1).partial_cmp(&a.1).expect("Shouldn't blow up!"));
            scores
        }
               
    }

}

// Computes the wilson confidence interval for binomial distributions; should perform
// better than the equivalent normal distribution approximation.
fn wilson_confidence_interval(wins: f32, losses: f32, z: f32) -> f32 {
    let n = wins + losses;
    let z2 = z.powf(2.);
    let nz2 = n + z2;
    let p_approx = (wins + z2 / 2.) / nz2 + (z / nz2) * (wins * losses / n + z2 / 4.).sqrt();
    p_approx
}

#[cfg(test)]
mod test_rate {
    use super::*;

    #[test]
    fn test_wilson_confidence() {
        let games = vec![
            (0, 1, 1.),
            (1, 2, 1.),
            (2, 1, 1.)
        ];
        
        // Check point estimate
        let rate = Rate::new(ConfidenceInterval::P50);
        let scores: HashMap<_,_> = rate.compute(&games).into_iter().collect();
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[&0], 1.0);
        assert_eq!(scores[&1], 1./3.);
        assert_eq!(scores[&2], 0.5);
        //
        
        println!("Next one!");
        let rate = Rate::new(ConfidenceInterval::P90);
        let scores: HashMap<_,_> = rate.compute(&games).into_iter().collect();
        assert_eq!(scores[&0], 1.);
        assert_eq!(scores[&1], 0.74649);
        assert_eq!(scores[&2], 0.879148);

    }

}

use std::collections::HashMap;
use std::hash::Hash;

/// Stats of a Player
#[derive(Debug,Clone,Copy)]
pub struct Player {

    /// Skill rating of a player
    pub rating: f64,

    /// Variance of a player's skill
    pub rd: f64,

    /// Measurement
    pub sigma: f64
}

impl Player {
    pub fn create( rating: f64, rd: f64, sigma: f64) -> Self {
        Player { rating: rating, rd: rd, sigma: sigma }
    }

    pub fn mu(&self, env: &Env) -> f64 {
        (self.rating - env.rating) / env.scale_factor
    }

    pub fn phi(&self, env: &Env) -> f64 {
        self.rd / env.scale_factor
    }

    pub fn bounds(&self) -> (f64, f64) {
        (self.rating - 2. * self.rd, self.rating + 2. * self.rd)
    }
}

/// Initializes a new Glicko2 environment
pub struct Env {
    rating: f64,
    rd: f64,
    sigma: f64,
    scale_factor: f64
}

impl Env {
    /// Creates a new environment with the provided values as defaults
    pub fn new(rating: f64, rd: f64, sigma: f64, scale_factor: f64) -> Self {
        Env {
            rating: rating,
            rd: rd,
            sigma: sigma,
            scale_factor: scale_factor
        }
    }

    /// Creates a new data series.
    pub fn new_match_set<'a, H: Clone + Hash + Eq + std::fmt::Debug>(&'a self, tau: f64) -> Series<'a, H> {
        Series::new(self, tau)
    }
}

impl Default for Env {
    fn default() -> Self {
        Env::new(1500., 350., 0.06, 173.7178f64)
    }
}

/// An updateable set of matches or series
pub struct Series<'a, ID: Hash + Eq> {
    env: &'a Env,
    tau: f64,
    teams: HashMap<ID, Player>
}

impl <'a, ID: Clone + Hash + Eq + std::fmt::Debug> Series<'a, ID> {
    /// Creates a new series using the provided environment 
    pub fn new(env: &'a Env, tau: f64) -> Self {
        Series {
            env: env,
            tau: tau,
            teams: HashMap::new()
        }
    }

    /// Updates the internal matches with the new set of results
    /// The first ID is a winner, the second one is the loser
    pub fn update(&mut self, games: Vec<(ID, ID)>) {
        // Compute v and delta for all matches
        let mut dv = HashMap::with_capacity(self.teams.capacity());
        for (win_team, loser_team) in games {

            // If we haven't seen the team before, add it to the mapping
            for t in [&win_team, &loser_team].iter() {
                if !self.teams.contains_key(t) {
                    let p = Player::create(self.env.rating, self.env.rd, self.env.sigma);
                    self.teams.insert((*t).clone(), p);
                }
            }

            // Insert winner and loser if not in teams
            for (w, l, s) in &[(&win_team, &loser_team, 1f64), 
                               (&loser_team, &win_team, 0.)] {
                //println!("{:?},{:?},{:?},{:?}", w, self.teams[w], l, self.teams[l]);
                let (vij, delta_i_j) = compute_vij_d(
                    &self.env, &self.teams[w], &self.teams[l], *s);

                let (vi, di) = dv.entry((*w).clone()).or_insert((0f64, 0f64));
                *vi += vij;
                *di += delta_i_j;
                //println!("vi: {}, di: {}", vij, delta_i_j);
                //assert!(di.is_finite());
                //assert!(vi.is_finite());
            }
        }

        // Compute new ratings, confidence, and volatility
        //for (team_id, team) in self.teams.iter_mut() {
        for team_id in dv.keys() {
            let team = self.teams.get_mut(team_id)
                .expect("This should have been added in the first pass");

            let (v_i, d_i) = dv[team_id];
            let v_t = 1. / v_i;
            let delta_t = v_t * d_i;
            let sigma_prime = compute_volatility(&self.env, team, delta_t, v_t, self.tau)
                .expect("Should be swell");

            let phi_star = (team.phi(&self.env).powi(2) + sigma_prime.powi(2)).powf(0.5);

            let phi_prime = 1. / (1. / phi_star.powi(2) + 1. / v_t).powf(0.5);
            let mu_prime = team.mu(&self.env) + phi_prime.powi(2) * d_i;
            let r_prime = mu_prime * self.env.scale_factor + self.env.rating;
            let rd_prime = phi_prime * self.env.scale_factor;
            //println!("{:?},{:?},{},{},{},{}",team_id,team, mu_prime, phi_prime, d_i, sigma_prime);
            //assert!(r_prime.is_finite());
            //assert!(rd_prime.is_finite());
            //assert!(sigma_prime.is_finite());
            *team = Player::create(r_prime, rd_prime, sigma_prime);
        }
    }

    pub fn teams(&self) -> &HashMap<ID,Player> {
        &self.teams
    }
}


#[inline]
fn g_of_phi(phi: f64) -> f64 {
    1. / (1. + 3. * phi.powi(2) / (std::f64::consts::PI).powi(2)).powf(0.5)
}

#[inline]
fn e(mu: f64, mu_j: f64, phi: f64) -> f64 {
    1. / (1. + (-g_of_phi(phi) * (mu - mu_j)).exp())
}

// Compute vi_j and \delta_i_j
fn compute_vij_d(env: &Env, winner: &Player, loser: &Player, score: f64) -> (f64, f64) {
    let mu_w = winner.mu(env);
    let mu_l = loser.mu(env);
    let phi_l = loser.phi(env);

    let g_phi = g_of_phi(phi_l);
    let expected_score =  e(mu_w, mu_l, phi_l);
    let vij = g_phi.powi(2) * expected_score * (1. - expected_score);
    let delta_i_j = g_phi * (score - expected_score);
    (vij, delta_i_j)
}

// Algorithm defined: 
// https://www.researchgate.net/profile/Mark_Glickman/publication/267801528_Example_of_the_Glicko-2_system/links/556c3d4408aefcb861d633e2/Example-of-the-Glicko-2-system.pdf
#[allow(non_snake_case)] 
fn compute_volatility(env: &Env, team: &Player, delta: f64, v: f64, tau: f64) -> Option<f64> {
    let phi = team.phi(env);
    let eps = 1e-7;
    let a = team.sigma.powi(2).ln();

    // This `t` I believe is for numerical stability
    let t = tau.powi(2).powf(0.5);
    let f = |x: f64| -> f64 {
        let nom1 = x.exp() * (delta.powi(2) - phi.powi(2) - v - x.exp()); 
        let denom1 = 2. * (phi.powi(2) + v + x.exp());

        nom1 / denom1 - (x - a) / tau.powi(2)
    };

    // We find the brackets for the algorithm below
    let mut A = a;
    let mut B = if delta.powi(2) > phi.powi(2) + v {
        (delta.powi(2) - phi.powi(2) - v).ln()
    } else {
        let mut k = 1.;
        for _ in 0..100 {
            let bp = f(a - k * t);
            if bp >= 0. {
                break
            } else {
                k += 1.;
            }
        }
        if k == 100. {
            return None
        }
        a - k * t
    };

    // Find the first zero
    let mut fa = f(A);
    let mut fb = f(B);
    for _i in 0..100 {
        if ((B - A).abs() <= eps) || A.is_nan() || B.is_nan() { break }
        let C = A + (A - B) * fa / (fb - fa);
        let fc = f(C);
        if (fc * fb) < 0. {
            A = B;
            fa = fb;
        } else {
            fa /= 2.;
        }

        B = C;
        fb = fc;
    }

    if A.is_nan() {
        eprintln!("Couldn't find a zero: {},{},{}", A, B, eps);
        return None
    }

    let sigma_2 = (A / 2.).exp();

    // New volatility
    Some(sigma_2)
}

#[cfg(test)]
mod test_glicko {
    use super::*;

    #[test]
    fn test_small_tau() {
        let env = Env::default();
        let mut series = Series::new(&env, 0.5);
        series.teams.insert(615756132, Player { rating: 1078.224870320442, rd: 231.8396899251802, sigma: 0.0599557629529191 });
        series.teams.insert(580568902, Player { rating: 1922.738120392382, rd: 136.9997727497604, sigma: 0.06095741696613419});

        series.update(vec![
            (615756132,580568902),
            (615756132,580568902)
        ]);

        assert!(series.teams[&615756132].sigma < 1.);
    }

    #[test]
    fn test_given_example() {
        let env = Env::default();
        let mut series = Series::new(&env, 0.5);

        series.teams.insert(1, Player { rating: 1500., rd: 200., sigma: 0.06 });
        series.teams.insert(2, Player { rating: 1400., rd: 30. , sigma: 0.06 });
        series.teams.insert(3, Player { rating: 1550., rd: 100., sigma: 0.06 });
        series.teams.insert(4, Player { rating: 1700., rd: 300., sigma: 0.06 });

        let games = vec![
            (1, 2),
            (3, 1),
            (4, 1)
        ];

        series.update(games);

        let team = series.teams[&1];
        assert!((team.rating - 1464.06).abs() < 0.01);
        assert!((team.rd - 151.52).abs() < 0.01);
        assert!((team.sigma - 0.05999).abs() < 0.01);
    }

}

//! Picks the next token from the model's logits.
//!
//! The steps run in a fixed order, each narrowing the candidates:
//!
//! 1. Penalties: lower the logits of recently used tokens, by a factor,
//!    by an amount per use, and by an amount per token.
//! 2. Top-k: keeps the `k` most likely tokens.
//! 3. Top-p: keeps the smallest set whose probabilities sum to `p`.
//! 4. Min-p: drops tokens less likely than `min_p` times the best one.
//! 5. Temperature: sharpens or flattens what is left.
//! 6. A random draw, or the best token when the temperature is zero.

/// Options for a [`Sampler`]. The defaults match common practice for chat.
#[derive(Clone, Debug)]
pub struct SamplerConfig {
    /// Randomness. Zero always picks the most likely token.
    pub temperature: f32,
    /// Keep this many of the most likely tokens. Zero keeps all.
    pub top_k: usize,
    /// Keep the most likely tokens whose probabilities add up to this.
    /// One keeps all.
    pub top_p: f32,
    /// Drop tokens less likely than this fraction of the best one.
    /// Zero keeps all.
    pub min_p: f32,
    /// Divide the logits of recent tokens by this. One changes nothing.
    pub repeat_penalty: f32,
    /// How many recent tokens the penalties look at.
    pub repeat_last_n: usize,
    /// Subtract this from a token's logit for every time it was used
    /// recently. Zero changes nothing.
    pub frequency_penalty: f32,
    /// Subtract this from the logit of every token that was used recently,
    /// however often. Zero changes nothing.
    pub presence_penalty: f32,
    /// Seed of the random draw. The same seed repeats the same choices.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> SamplerConfig {
        SamplerConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
        }
    }
}

/// A small, fast random number generator (xoshiro256**).
#[derive(Clone, Debug)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// Seeds the generator. Any seed is valid, including zero.
    pub fn new(seed: u64) -> Rng {
        // SplitMix64 spreads the seed over the whole state.
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        Rng {
            state: [next(), next(), next(), next()],
        }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// A uniform float in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A candidate token and its logit.
#[derive(Clone, Copy)]
struct Candidate {
    token: u32,
    logit: f32,
}

/// Chooses tokens according to a [`SamplerConfig`].
pub struct Sampler {
    config: SamplerConfig,
    rng: Rng,
    /// Reused between calls to avoid allocating per token.
    candidates: Vec<Candidate>,
    probabilities: Vec<f32>,
    /// The recent tokens, sorted, for counting how often each was used.
    recent: Vec<u32>,
}

impl Sampler {
    /// Creates a sampler.
    pub fn new(config: SamplerConfig) -> Sampler {
        let rng = Rng::new(config.seed);
        Sampler {
            config,
            rng,
            candidates: Vec::new(),
            probabilities: Vec::new(),
            recent: Vec::new(),
        }
    }

    /// The options this sampler was created with.
    pub fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Picks a token from `logits`. `history` is the context so far, used by
    /// the penalties on recent tokens.
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        let config = &self.config;
        let any_penalty = config.repeat_penalty != 1.0
            || config.frequency_penalty != 0.0
            || config.presence_penalty != 0.0;
        let penalize = any_penalty && config.repeat_last_n > 0 && !history.is_empty();

        // The common greedy case needs no candidate list at all.
        if config.temperature <= 0.0 && !penalize {
            return argmax(logits);
        }

        let candidates = &mut self.candidates;
        candidates.clear();
        candidates.extend(logits.iter().enumerate().map(|(token, &logit)| Candidate {
            token: token as u32,
            logit,
        }));
        if candidates.is_empty() {
            return 0;
        }

        if penalize {
            // Sorting the recent tokens puts equal ones next to each other,
            // so each distinct token is handled once, with its count.
            let recent = &mut self.recent;
            recent.clear();
            recent
                .extend_from_slice(&history[history.len().saturating_sub(config.repeat_last_n)..]);
            recent.sort_unstable();
            for uses in recent.chunk_by(|a, b| a == b) {
                let Some(candidate) = candidates.get_mut(uses[0] as usize) else {
                    continue;
                };
                // Dividing a negative logit would raise it, so those are
                // multiplied instead.
                if candidate.logit > 0.0 {
                    candidate.logit /= config.repeat_penalty;
                } else {
                    candidate.logit *= config.repeat_penalty;
                }
                candidate.logit -=
                    uses.len() as f32 * config.frequency_penalty + config.presence_penalty;
            }
        }
        if config.temperature <= 0.0 {
            return best(candidates).token;
        }

        // Top-k. A partial selection finds the k best without sorting the
        // whole vocabulary.
        let descending = |a: &Candidate, b: &Candidate| b.logit.total_cmp(&a.logit);
        if config.top_k > 0 && config.top_k < candidates.len() {
            candidates.select_nth_unstable_by(config.top_k - 1, descending);
            candidates.truncate(config.top_k);
        }
        candidates.sort_unstable_by(descending);

        // Probabilities of the survivors, before temperature.
        let probabilities = &mut self.probabilities;
        softmax(candidates, 1.0, probabilities);

        // Top-p: the shortest prefix that reaches the target mass.
        if config.top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut keep = candidates.len();
            for (index, probability) in probabilities.iter().enumerate() {
                cumulative += probability;
                if cumulative >= config.top_p {
                    keep = index + 1;
                    break;
                }
            }
            candidates.truncate(keep.max(1));
        }

        // Min-p: relative to the most likely token.
        if config.min_p > 0.0 {
            let threshold = probabilities[0] * config.min_p;
            let keep = probabilities[..candidates.len()]
                .iter()
                .take_while(|&&p| p >= threshold)
                .count();
            candidates.truncate(keep.max(1));
        }

        // Temperature, then a draw proportional to probability.
        softmax(candidates, config.temperature, probabilities);
        let target = self.rng.next_f32();
        let mut cumulative = 0.0;
        for (candidate, probability) in candidates.iter().zip(probabilities.iter()) {
            cumulative += probability;
            if target < cumulative {
                return candidate.token;
            }
        }
        // Rounding can leave the total a hair under one.
        candidates[candidates.len() - 1].token
    }
}

/// Index of the largest logit. The first one wins a tie.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (index, &logit) in logits.iter().enumerate() {
        if logit > logits[best] {
            best = index;
        }
    }
    best as u32
}

/// The candidate with the largest logit. The lowest token id wins a tie.
fn best(candidates: &[Candidate]) -> Candidate {
    let mut best = candidates[0];
    for candidate in candidates {
        if candidate.logit > best.logit {
            best = *candidate;
        }
    }
    best
}

/// Writes the probabilities of `candidates` at `temperature` to `out`.
fn softmax(candidates: &[Candidate], temperature: f32, out: &mut Vec<f32>) {
    out.clear();
    let max = candidates
        .iter()
        .fold(f32::NEG_INFINITY, |max, c| max.max(c.logit));
    out.extend(
        candidates
            .iter()
            .map(|c| ((c.logit - max) / temperature).exp()),
    );
    let total: f32 = out.iter().sum();
    out.iter_mut().for_each(|p| *p /= total);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequency_and_presence_penalties_count_recent_uses() {
        // Four equal logits. Token 1 was used three times, token 2 once.
        let logits = [2.0f32, 2.0, 2.0, 2.0];
        let history = [1, 2, 1, 1];
        let greedy = |frequency_penalty: f32, presence_penalty: f32| {
            let config = SamplerConfig {
                temperature: 0.0,
                frequency_penalty,
                presence_penalty,
                ..SamplerConfig::default()
            };
            let mut sampler = Sampler::new(config);
            sampler.sample(&logits, &history);
            sampler
                .candidates
                .iter()
                .map(|candidate| candidate.logit)
                .collect::<Vec<f32>>()
        };
        // Per use: three uses cost three times as much as one.
        assert_eq!(greedy(0.5, 0.0), [2.0, 0.5, 1.5, 2.0]);
        // Per token: any number of uses costs the same.
        assert_eq!(greedy(0.0, 0.5), [2.0, 1.5, 1.5, 2.0]);
        assert_eq!(greedy(0.5, 0.25), [2.0, 0.25, 1.25, 2.0]);
    }

    #[test]
    fn the_repeat_penalty_applies_once_per_token() {
        let config = SamplerConfig {
            temperature: 0.0,
            repeat_penalty: 2.0,
            ..SamplerConfig::default()
        };
        let mut sampler = Sampler::new(config);
        // Token 0 was used twice and still pays the penalty once.
        assert_eq!(sampler.sample(&[4.0, 3.0, -1.0], &[0, 0, 2]), 1);
        let logits: Vec<f32> = sampler
            .candidates
            .iter()
            .map(|candidate| candidate.logit)
            .collect();
        assert_eq!(logits, [2.0, 3.0, -2.0]);
    }

    fn config() -> SamplerConfig {
        SamplerConfig {
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            ..Default::default()
        }
    }

    /// Draws `n` tokens and counts how often each id comes up.
    fn histogram(config: SamplerConfig, logits: &[f32], n: usize) -> Vec<usize> {
        let mut sampler = Sampler::new(config);
        let mut counts = vec![0; logits.len()];
        for _ in 0..n {
            counts[sampler.sample(logits, &[]) as usize] += 1;
        }
        counts
    }

    #[test]
    fn zero_temperature_is_greedy() {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        });
        assert_eq!(sampler.sample(&[0.1, 2.0, -1.0, 2.0], &[]), 1);
        assert_eq!(argmax(&[-5.0, -1.0, -3.0]), 1);
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn draws_follow_the_distribution() {
        // Probabilities 0.090, 0.245, 0.665.
        let counts = histogram(
            SamplerConfig {
                temperature: 1.0,
                ..config()
            },
            &[0.0, 1.0, 2.0],
            20_000,
        );
        let share = |index: usize| counts[index] as f64 / 20_000.0;
        assert!((share(0) - 0.090).abs() < 0.015, "{counts:?}");
        assert!((share(1) - 0.245).abs() < 0.02, "{counts:?}");
        assert!((share(2) - 0.665).abs() < 0.02, "{counts:?}");
    }

    #[test]
    fn temperature_sharpens_and_flattens() {
        let logits = [0.0, 1.0, 2.0];
        let cold = histogram(
            SamplerConfig {
                temperature: 0.2,
                ..config()
            },
            &logits,
            5_000,
        );
        let hot = histogram(
            SamplerConfig {
                temperature: 5.0,
                ..config()
            },
            &logits,
            5_000,
        );
        assert!(cold[2] > 4_900, "{cold:?}");
        assert!(hot[0] > 1_200, "{hot:?}");
    }

    #[test]
    fn top_k_limits_the_candidates() {
        let logits = [1.0, 5.0, 4.0, 0.0, 4.5];
        let counts = histogram(
            SamplerConfig {
                top_k: 2,
                temperature: 3.0,
                ..config()
            },
            &logits,
            4_000,
        );
        assert_eq!(counts[0] + counts[2] + counts[3], 0, "{counts:?}");
        assert!(counts[1] > 0 && counts[4] > 0);
    }

    #[test]
    fn top_p_keeps_the_head_of_the_distribution() {
        // Probabilities 0.64, 0.23, 0.09, 0.03: the first two reach 0.8.
        let logits = [3.0, 2.0, 1.0, 0.0];
        let counts = histogram(
            SamplerConfig {
                top_p: 0.8,
                temperature: 1.0,
                ..config()
            },
            &logits,
            4_000,
        );
        assert_eq!(counts[2] + counts[3], 0, "{counts:?}");
        assert!(counts[0] > 0 && counts[1] > 0);
        // The best token always survives.
        let counts = histogram(
            SamplerConfig {
                top_p: 0.0,
                ..config()
            },
            &logits,
            100,
        );
        assert_eq!(counts[0], 100);
    }

    #[test]
    fn min_p_drops_the_unlikely() {
        let logits = [3.0, 2.0, 1.0, -8.0];
        let counts = histogram(
            SamplerConfig {
                min_p: 0.2,
                temperature: 1.0,
                ..config()
            },
            &logits,
            4_000,
        );
        // Relative to the best: 1.0, 0.37, 0.14, 0.00002.
        assert_eq!(counts[2] + counts[3], 0, "{counts:?}");
        assert!(counts[1] > 0);
    }

    #[test]
    fn repetition_penalty_moves_away_from_recent_tokens() {
        let logits = [2.0, 1.9, -1.0];
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            repeat_penalty: 1.5,
            ..config()
        });
        assert_eq!(sampler.sample(&logits, &[]), 0);
        assert_eq!(sampler.sample(&logits, &[0]), 1);
        // Negative logits are pushed further down, not up.
        assert_eq!(sampler.sample(&[-1.0, -1.2], &[0]), 1);
        // Tokens outside the window are not penalized.
        let mut windowed = Sampler::new(SamplerConfig {
            repeat_last_n: 1,
            ..sampler.config().clone()
        });
        assert_eq!(windowed.sample(&logits, &[0, 2]), 0);
        // Ids outside the vocabulary in the history are ignored.
        assert_eq!(sampler.sample(&logits, &[99]), 0);
    }

    #[test]
    fn a_seed_repeats_its_choices() {
        let logits: Vec<f32> = (0..50).map(|i| (i % 7) as f32 * 0.3).collect();
        let draw = |seed: u64| {
            let mut sampler = Sampler::new(SamplerConfig {
                seed,
                ..Default::default()
            });
            (0..40)
                .map(|_| sampler.sample(&logits, &[]))
                .collect::<Vec<u32>>()
        };
        assert_eq!(draw(7), draw(7));
        assert_ne!(draw(7), draw(8));
    }

    #[test]
    fn random_floats_stay_in_range() {
        let mut rng = Rng::new(0);
        let mut sum = 0.0f64;
        for _ in 0..10_000 {
            let value = rng.next_f32();
            assert!((0.0..1.0).contains(&value));
            sum += value as f64;
        }
        assert!((sum / 10_000.0 - 0.5).abs() < 0.02);
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let mut sampler = Sampler::new(SamplerConfig::default());
        assert_eq!(sampler.sample(&[], &[]), 0);
        assert_eq!(sampler.sample(&[1.0], &[0, 0, 0]), 0);
        let token = sampler.sample(&[f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY], &[]);
        assert_eq!(token, 1);
    }
}

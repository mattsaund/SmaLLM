//! Picks the next token from the model's logits.
//!
//! The steps run in llama.cpp's default order, each narrowing the
//! candidates. A step whose setting is off is skipped.
//!
//! 1. Logit bias: adds a fixed amount to chosen tokens' logits.
//! 2. Penalties: lower the logits of recently used tokens, by a factor,
//!    by an amount per use, and by an amount per token.
//! 3. Top-n-sigma: keeps tokens within `n` standard deviations of the
//!    best logit.
//! 4. Top-k: keeps the `k` most likely tokens.
//! 5. Typical: keeps the tokens whose surprise is closest to the
//!    distribution's entropy, up to a probability mass.
//! 6. Top-p: keeps the smallest set whose probabilities sum to `p`.
//! 7. Min-p: drops tokens less likely than `min_p` times the best one.
//! 8. XTC: now and then drops the most likely tokens, all but the least
//!    likely of those above a threshold.
//! 9. Temperature: sharpens or flattens what is left. A dynamic range
//!    moves it with the entropy of the candidates.
//! 10. A random draw, or the best token when the temperature is zero.
//!
//! Mirostat replaces steps 3 to 8: after the temperature it truncates the
//! candidates to keep the surprise of the chosen tokens near a target.

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
    /// Keep the tokens whose surprise is closest to the entropy, up to this
    /// probability mass. One keeps all.
    pub typical_p: f32,
    /// Keep tokens within this many standard deviations of the best logit.
    /// Zero keeps all.
    pub top_n_sigma: f32,
    /// How often XTC drops the most likely tokens. Zero never does.
    pub xtc_probability: f32,
    /// The probability a token needs for XTC to drop it. Above 0.5 turns
    /// XTC off.
    pub xtc_threshold: f32,
    /// Let the temperature move this far either way with the entropy of the
    /// candidates. Zero keeps it fixed.
    pub dynatemp_range: f32,
    /// How the dynamic temperature follows the entropy: 1 is linear.
    pub dynatemp_exponent: f32,
    /// Mirostat: 0 off, 1 for the first version, 2 for the second.
    pub mirostat: u8,
    /// Mirostat's target surprise, in bits.
    pub mirostat_tau: f32,
    /// How fast Mirostat corrects toward its target.
    pub mirostat_eta: f32,
    /// Added to the logits of chosen tokens. Negative infinity bans one.
    pub logit_bias: Vec<(u32, f32)>,
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
            typical_p: 1.0,
            top_n_sigma: 0.0,
            xtc_probability: 0.0,
            xtc_threshold: 0.1,
            dynatemp_range: 0.0,
            dynatemp_exponent: 1.0,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            logit_bias: Vec::new(),
            seed: 0,
        }
    }
}

impl SamplerConfig {
    /// The defaults, with the sampling settings a model file recommends
    /// (`general.sampling.*`) put over them, as llama.cpp reads them. A
    /// caller's own settings then go over these.
    pub fn from_model_file(file: &crate::gguf::Gguf) -> SamplerConfig {
        let mut config = SamplerConfig::default();
        let number = |name: &str| file.get_f32(&format!("general.sampling.{name}"));
        let whole = |name: &str| file.get_usize(&format!("general.sampling.{name}"));
        if let Some(value) = number("temp") {
            config.temperature = value;
        }
        if let Some(value) = whole("top_k") {
            config.top_k = value;
        }
        if let Some(value) = number("top_p") {
            config.top_p = value;
        }
        if let Some(value) = number("min_p") {
            config.min_p = value;
        }
        if let Some(value) = number("penalty_repeat") {
            config.repeat_penalty = value;
        }
        if let Some(value) = whole("penalty_last_n") {
            config.repeat_last_n = value;
        }
        if let Some(value) = number("xtc_probability") {
            config.xtc_probability = value;
        }
        if let Some(value) = number("xtc_threshold") {
            config.xtc_threshold = value;
        }
        if let Some(value) = whole("mirostat").filter(|&value| value <= 2) {
            config.mirostat = value as u8;
        }
        if let Some(value) = number("mirostat_tau") {
            config.mirostat_tau = value;
        }
        if let Some(value) = number("mirostat_eta") {
            config.mirostat_eta = value;
        }
        config
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
    /// Candidate positions in the order the typical step ranks them.
    order: Vec<usize>,
    /// Mirostat's running estimate of the surprise it allows, in bits.
    mu: f32,
}

impl Sampler {
    /// Creates a sampler.
    pub fn new(config: SamplerConfig) -> Sampler {
        let rng = Rng::new(config.seed);
        Sampler {
            mu: 2.0 * config.mirostat_tau,
            config,
            rng,
            candidates: Vec::new(),
            probabilities: Vec::new(),
            recent: Vec::new(),
            order: Vec::new(),
        }
    }

    /// The options this sampler was created with.
    pub fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Picks a token from `logits`. `history` is the context so far, used by
    /// the penalties on recent tokens.
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        let Sampler {
            config,
            rng,
            candidates,
            probabilities,
            recent,
            order,
            mu,
        } = self;
        let any_penalty = config.repeat_penalty != 1.0
            || config.frequency_penalty != 0.0
            || config.presence_penalty != 0.0;
        let penalize = any_penalty && config.repeat_last_n > 0 && !history.is_empty();
        let greedy =
            config.temperature <= 0.0 && config.dynatemp_range <= 0.0 && config.mirostat == 0;

        // The common greedy case needs no candidate list at all.
        if greedy && !penalize && config.logit_bias.is_empty() {
            return argmax(logits);
        }

        candidates.clear();
        candidates.extend(logits.iter().enumerate().map(|(token, &logit)| Candidate {
            token: token as u32,
            logit,
        }));
        if candidates.is_empty() {
            return 0;
        }

        for &(token, bias) in &config.logit_bias {
            if let Some(candidate) = candidates.get_mut(token as usize) {
                candidate.logit += bias;
            }
        }

        if penalize {
            // Sorting the recent tokens puts equal ones next to each other,
            // so each distinct token is handled once, with its count.
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
        if greedy {
            return best(candidates).token;
        }

        let descending = |a: &Candidate, b: &Candidate| b.logit.total_cmp(&a.logit);
        if config.mirostat != 0 {
            return mirostat(config, rng, mu, candidates, probabilities, logits.len());
        }

        // Top-n-sigma, on the logits of every candidate.
        if config.top_n_sigma > 0.0 && candidates.len() > 1 {
            let finite = candidates.iter().filter(|c| c.logit > f32::NEG_INFINITY);
            let (count, sum, max) = finite.fold((0usize, 0f64, f32::NEG_INFINITY), |acc, c| {
                (acc.0 + 1, acc.1 + c.logit as f64, acc.2.max(c.logit))
            });
            if count > 0 {
                let mean = sum / count as f64;
                let variance = candidates
                    .iter()
                    .filter(|c| c.logit > f32::NEG_INFINITY)
                    .map(|c| (c.logit as f64 - mean).powi(2))
                    .sum::<f64>()
                    / count as f64;
                let floor = max - config.top_n_sigma * variance.sqrt() as f32;
                candidates.retain(|c| c.logit >= floor);
            }
        }

        // Top-k. A partial selection finds the k best without sorting the
        // whole vocabulary.
        if config.top_k > 0 && config.top_k < candidates.len() {
            candidates.select_nth_unstable_by(config.top_k - 1, descending);
            candidates.truncate(config.top_k);
        }
        candidates.sort_unstable_by(descending);

        // Probabilities of the survivors, before temperature.
        softmax(candidates, 1.0, probabilities);

        // Typical: rank by how far each token's surprise is from the
        // entropy, and keep the closest ones up to the target mass.
        if config.typical_p < 1.0 && candidates.len() > 1 {
            let entropy: f32 = probabilities
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -p * p.ln())
                .sum();
            let distance = |p: f32| (-p.ln() - entropy).abs();
            order.clear();
            order.extend(0..candidates.len());
            order.sort_by(|&a, &b| {
                distance(probabilities[a]).total_cmp(&distance(probabilities[b]))
            });
            let mut cumulative = 0.0;
            let mut keep = order.len();
            for (rank, &index) in order.iter().enumerate() {
                cumulative += probabilities[index];
                if cumulative > config.typical_p {
                    keep = rank + 1;
                    break;
                }
            }
            // Keep the chosen ones in order of logit, which the later
            // steps expect.
            let mut kept = vec![false; candidates.len()];
            order[..keep].iter().for_each(|&index| kept[index] = true);
            let mut flags = kept.iter();
            candidates.retain(|_| *flags.next().unwrap_or(&false));
            softmax(candidates, 1.0, probabilities);
        }

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

        // XTC: with some probability, drop every token above the threshold
        // but the least likely of them, so that the obvious choice is not
        // always taken.
        if config.xtc_probability > 0.0
            && config.xtc_threshold <= 0.5
            && candidates.len() >= 2
            && rng.next_f32() <= config.xtc_probability
        {
            softmax(candidates, 1.0, probabilities);
            let above = probabilities
                .iter()
                .take_while(|&&p| p >= config.xtc_threshold)
                .count();
            if above >= 2 {
                candidates.drain(..above - 1);
            }
        }

        // Temperature, then a draw proportional to probability.
        softmax(candidates, 1.0, probabilities);
        let temperature = if config.dynatemp_range > 0.0 {
            dynamic_temperature(probabilities, config)
        } else {
            config.temperature
        };
        if temperature <= 0.0 {
            return candidates[0].token;
        }
        softmax(candidates, temperature, probabilities);
        candidates[draw(rng, probabilities)].token
    }
}

/// The temperature for candidates with these probabilities: from
/// `temperature - range` for a certain choice to `temperature + range` for a
/// uniform one, by the normalized entropy raised to the exponent.
fn dynamic_temperature(probabilities: &[f32], config: &SamplerConfig) -> f32 {
    if probabilities.len() <= 1 {
        return config.temperature;
    }
    let low = (config.temperature - config.dynatemp_range).max(0.0);
    let high = config.temperature + config.dynatemp_range;
    let entropy: f32 = probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    let normalized = entropy / (probabilities.len() as f32).ln();
    low + (high - low) * normalized.powf(config.dynatemp_exponent)
}

/// Picks an index with the given probabilities, which sum to one.
fn draw(rng: &mut Rng, probabilities: &[f32]) -> usize {
    let target = rng.next_f32();
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if target < cumulative {
            return index;
        }
    }
    // Rounding can leave the total a hair under one.
    probabilities.len().saturating_sub(1)
}

/// Mirostat, after the temperature: truncates the candidates so that the
/// surprise of the chosen token, `-log2 p`, stays near `tau`, and moves its
/// estimate `mu` after each choice. Version 1 sets top-k from an estimate
/// of how the probabilities fall off (Zipf's exponent over the 100 most
/// likely tokens). Version 2 keeps the tokens whose surprise is under `mu`.
/// Both follow llama.cpp.
fn mirostat(
    config: &SamplerConfig,
    rng: &mut Rng,
    mu: &mut f32,
    candidates: &mut Vec<Candidate>,
    probabilities: &mut Vec<f32>,
    n_vocab: usize,
) -> u32 {
    candidates.sort_unstable_by(|a, b| b.logit.total_cmp(&a.logit));
    if config.temperature <= 0.0 {
        return candidates[0].token;
    }
    softmax(candidates, config.temperature, probabilities);
    if config.mirostat == 1 {
        let m = 100.min(candidates.len());
        let (mut sum_ti_bi, mut sum_ti_sq) = (0.0f32, 0.0f32);
        for i in 0..m.saturating_sub(1) {
            let t_i = ((i + 2) as f32 / (i + 1) as f32).ln();
            let b_i = (probabilities[i] / probabilities[i + 1]).ln();
            sum_ti_bi += t_i * b_i;
            sum_ti_sq += t_i * t_i;
        }
        let s_hat = sum_ti_bi / sum_ti_sq;
        let epsilon_hat = s_hat - 1.0;
        let k = ((epsilon_hat * 2f32.powf(*mu)) / (1.0 - (n_vocab as f32).powf(-epsilon_hat)))
            .powf(1.0 / s_hat);
        // A degenerate estimate keeps one token, as llama.cpp's cast does.
        let k = if k.is_finite() && k >= 1.0 {
            k as usize
        } else {
            1
        };
        candidates.truncate(k.max(1));
    } else {
        let keep = probabilities
            .iter()
            .take_while(|&&p| -p.log2() <= *mu)
            .count();
        candidates.truncate(keep.max(1));
    }
    softmax(candidates, config.temperature, probabilities);
    let index = draw(rng, probabilities);
    let surprise = -probabilities[index].log2();
    *mu -= config.mirostat_eta * (surprise - config.mirostat_tau);
    candidates[index].token
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

    /// Logits whose probabilities are `0.5, 0.2, 0.2, 0.1`.
    fn four() -> [f32; 4] {
        [0.5f32.ln(), 0.2f32.ln(), 0.2f32.ln(), 0.1f32.ln()]
    }

    #[test]
    fn logit_bias_raises_and_bans_tokens() {
        let greedy = |logit_bias: Vec<(u32, f32)>| {
            let mut sampler = Sampler::new(SamplerConfig {
                temperature: 0.0,
                logit_bias,
                ..SamplerConfig::default()
            });
            sampler.sample(&[1.0, 2.0, 3.0], &[])
        };
        assert_eq!(greedy(vec![]), 2);
        assert_eq!(greedy(vec![(0, 5.0)]), 0);
        assert_eq!(greedy(vec![(2, f32::NEG_INFINITY)]), 1);
        // A token outside the vocabulary is ignored.
        assert_eq!(greedy(vec![(9, 5.0)]), 2);
    }

    #[test]
    fn top_n_sigma_keeps_tokens_near_the_best() {
        let logits = [10.0, 9.5, 0.0, 0.0, 0.0, 0.0];
        let config = SamplerConfig {
            top_n_sigma: 1.0,
            ..config()
        };
        let counts = histogram(config, &logits, 2000);
        assert!(counts[0] > 0 && counts[1] > 0);
        assert_eq!(counts[2..].iter().sum::<usize>(), 0, "{counts:?}");
    }

    #[test]
    fn typical_keeps_the_tokens_of_typical_surprise() {
        // The entropy is 1.22 nats. Tokens 1 and 2, with a surprise of 1.61,
        // are closer to it than token 0 (0.69), and together pass 0.3.
        let config = SamplerConfig {
            typical_p: 0.3,
            ..config()
        };
        let counts = histogram(config, &four(), 2000);
        assert_eq!((counts[0], counts[3]), (0, 0), "{counts:?}");
        assert!(counts[1] > 0 && counts[2] > 0);
    }

    #[test]
    fn xtc_drops_the_likeliest_tokens_but_the_last_above_its_threshold() {
        // Three tokens reach 0.15, so the first two are dropped every time.
        let config = SamplerConfig {
            xtc_probability: 1.0,
            xtc_threshold: 0.15,
            ..config()
        };
        let counts = histogram(config.clone(), &four(), 2000);
        assert_eq!((counts[0], counts[1]), (0, 0), "{counts:?}");
        assert!(counts[2] > 0 && counts[3] > 0);
        // A threshold above one half turns it off.
        let off = SamplerConfig {
            xtc_threshold: 0.6,
            ..config
        };
        assert!(histogram(off, &four(), 2000)[0] > 0);
    }

    #[test]
    fn dynamic_temperature_follows_the_entropy() {
        let config = SamplerConfig {
            temperature: 1.0,
            dynatemp_range: 0.5,
            ..SamplerConfig::default()
        };
        // A uniform choice gets the highest temperature, a certain one the
        // lowest.
        assert!((dynamic_temperature(&[0.25; 4], &config) - 1.5).abs() < 1e-5);
        assert!((dynamic_temperature(&[1.0, 0.0, 0.0], &config) - 0.5).abs() < 1e-5);
        let half = dynamic_temperature(&[0.5, 0.5, 0.0, 0.0], &config);
        assert!(half > 0.5 && half < 1.5);
        assert_eq!(dynamic_temperature(&[1.0], &config), 1.0);
    }

    #[test]
    fn mirostat_follows_its_target_surprise() {
        for version in [1, 2] {
            // A target far below any token's surprise leaves the best one.
            let mut sampler = Sampler::new(SamplerConfig {
                temperature: 1.0,
                mirostat: version,
                mirostat_tau: 0.01,
                ..SamplerConfig::default()
            });
            for _ in 0..20 {
                assert_eq!(sampler.sample(&four(), &[]), 0, "version {version}");
            }
            // A high target lets every token through, and moves `mu`.
            let config = SamplerConfig {
                temperature: 1.0,
                mirostat: version,
                mirostat_tau: 8.0,
                ..config()
            };
            let counts = histogram(config.clone(), &four(), 2000);
            assert!(
                counts.iter().all(|&count| count > 0),
                "version {version}: {counts:?}"
            );
            let mut sampler = Sampler::new(config);
            let start = sampler.mu;
            sampler.sample(&four(), &[]);
            assert_ne!(sampler.mu, start);
        }
    }

    #[test]
    fn model_files_set_the_defaults_they_recommend() {
        use crate::gguf::Gguf;
        use crate::gguf::testing::Builder;
        let mut builder = Builder::new();
        builder
            .string("general.architecture", "llama")
            .f32("general.sampling.temp", 1.0)
            .u32("general.sampling.top_k", 64)
            .f32("general.sampling.top_p", 0.95)
            .u32("general.sampling.mirostat", 2);
        let file = Gguf::from_bytes(builder.build()).unwrap();
        let config = SamplerConfig::from_model_file(&file);
        assert_eq!(
            (config.temperature, config.top_k, config.top_p),
            (1.0, 64, 0.95)
        );
        assert_eq!(config.mirostat, 2);
        // What the file does not name keeps the built-in default.
        assert_eq!(config.min_p, SamplerConfig::default().min_p);
    }
}

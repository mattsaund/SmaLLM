//! The state of recurrent blocks, and how to go back in it.
//!
//! A recurrent block (Mamba, RWKV) folds each token into a state of fixed
//! size, so it needs no cache that grows with the context. But a state
//! cannot be taken apart again. To cut the context back, as when a new
//! prompt shares only part of the old one, the session restores a saved
//! state and reads the tokens after it again.
//!
//! These states are large next to a short convolution's: a block keeps a
//! matrix per head. So they are saved sparingly, at the end of each prompt,
//! and only the last [`CHECKPOINTS`] of them are kept. That covers a chat.
//! The next prompt repeats the conversation up to the end of the reply, so
//! the state after the last prompt comes back and the reply is read again
//! as one batch, which is quick.

use super::config::{Config, Mixer};

/// Saved states kept. Older ones are dropped first.
pub const CHECKPOINTS: usize = 3;

/// The floats of state each block keeps, zero for blocks that are not
/// recurrent. Each recurrent mixer adds its arm here.
pub fn state_sizes(config: &Config) -> Vec<usize> {
    config
        .layers
        .iter()
        .map(|layer| match layer.mixer {
            Mixer::Mamba(ssm) => ssm.state_floats(),
            _ => 0,
        })
        .collect()
}

/// The states of a session's recurrent blocks.
pub struct RecurrentState {
    /// For each block, where its state starts in `state` and how many
    /// floats it has. Zero floats for other blocks.
    slots: Vec<(usize, usize)>,
    /// Every block's state after `len` positions, block after block.
    state: Vec<f32>,
    /// Positions folded in so far.
    len: usize,
    /// Saved states: a position and the whole state there, oldest first.
    checkpoints: Vec<(usize, Vec<f32>)>,
}

impl RecurrentState {
    /// Zero states for blocks of `sizes[i]` floats each.
    pub fn new(sizes: &[usize]) -> RecurrentState {
        let mut slots = Vec::with_capacity(sizes.len());
        let mut total = 0;
        for &size in sizes {
            slots.push((total, size));
            total += size;
        }
        RecurrentState {
            slots,
            state: vec![0.0; total],
            len: 0,
            checkpoints: Vec::new(),
        }
    }

    /// Whether block `layer` keeps a recurrent state.
    pub fn has(&self, layer: usize) -> bool {
        self.slots.get(layer).is_some_and(|&(_, size)| size > 0)
    }

    /// Block `layer`'s state, to read and update in place.
    pub fn block(&mut self, layer: usize) -> &mut [f32] {
        let (start, size) = self.slots[layer];
        &mut self.state[start..start + size]
    }

    /// Marks the positions before `end` as folded in, once every block has
    /// taken its part of a batch.
    pub fn advance(&mut self, end: usize) {
        self.len = end;
    }

    /// Keeps a copy of the current state, for a later cut back to here.
    /// The session calls this at the end of each prompt.
    pub fn save(&mut self) {
        if self.state.is_empty()
            || self
                .checkpoints
                .last()
                .is_some_and(|(at, _)| *at == self.len)
        {
            return;
        }
        let copy = if self.checkpoints.len() >= CHECKPOINTS {
            // Reuse the oldest copy's memory.
            let (_, mut oldest) = self.checkpoints.remove(0);
            oldest.copy_from_slice(&self.state);
            oldest
        } else {
            self.state.clone()
        };
        self.checkpoints.push((self.len, copy));
    }

    /// Cuts back to `len` positions, and returns the position the state
    /// now stands at. That is `len` when it is the current position or a
    /// saved one. Otherwise it is the last saved position before `len`, or
    /// zero, and the tokens from there to `len` must be read again.
    pub fn rewind(&mut self, len: usize) -> usize {
        if self.state.is_empty() {
            self.len = len;
            return len;
        }
        if len >= self.len {
            return self.len;
        }
        self.checkpoints.retain(|(at, _)| *at <= len);
        match self.checkpoints.last() {
            Some((at, saved)) => {
                self.state.copy_from_slice(saved);
                self.len = *at;
            }
            None => {
                self.state.fill(0.0);
                self.len = 0;
            }
        }
        self.len
    }

    /// Removes positions `keep..keep + discard`, as the session does when
    /// its context slides. A state does not depend on positions, so it
    /// stays as it is, and still holds what it took from the removed
    /// tokens, as attention caches do after a shift. Saved states are
    /// dropped: their positions are gone.
    pub fn shift(&mut self, keep: usize, discard: usize) {
        let discard = discard.min(self.len.saturating_sub(keep));
        self.checkpoints.clear();
        self.len -= discard;
    }

    /// Bytes of memory the states and saved copies take.
    pub fn bytes(&self) -> usize {
        let saved: usize = self.checkpoints.iter().map(|(_, copy)| copy.len()).sum();
        (self.state.len() + saved) * size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes `value` into every block's state.
    fn fill(state: &mut RecurrentState, value: f32) {
        for layer in 0..3 {
            state.block(layer).fill(value);
        }
    }

    #[test]
    fn blocks_get_their_own_states() {
        let mut state = RecurrentState::new(&[2, 0, 3]);
        assert!(state.has(0) && !state.has(1) && state.has(2));

        state.block(0).fill(1.0);
        state.block(2).fill(2.0);
        assert_eq!(state.block(0), [1.0, 1.0]);
        assert!(state.block(1).is_empty());
        assert_eq!(state.block(2), [2.0, 2.0, 2.0]);
        assert_eq!(state.bytes(), 5 * 4);
        assert_eq!(RecurrentState::new(&[0, 0]).bytes(), 0);
    }

    #[test]
    fn rewinding_restores_the_last_saved_state_before() {
        let mut state = RecurrentState::new(&[2, 0, 1]);
        // A prompt of 10 tokens, then 5 generated one at a time.
        fill(&mut state, 10.0);
        state.advance(10);
        state.save();
        fill(&mut state, 15.0);
        state.advance(15);
        assert_eq!(state.bytes(), 2 * 3 * 4);
        // Going back into the reply restores the state after the prompt.
        assert_eq!(state.rewind(13), 10);
        assert_eq!(state.block(0), [10.0, 10.0]);
        // Back to a saved position exactly needs nothing read again.
        assert_eq!(state.rewind(10), 10);
        // Before every saved state, everything is read again.
        assert_eq!(state.rewind(4), 0);
        assert_eq!(state.block(2), [0.0]);
        assert_eq!(state.bytes(), 3 * 4);
    }

    #[test]
    fn only_the_last_states_are_kept() {
        let mut state = RecurrentState::new(&[1, 0, 0]);
        for turn in 1..=CHECKPOINTS + 2 {
            fill(&mut state, turn as f32);
            state.advance(turn * 10);
            state.save();
            // Saving twice at one position keeps one copy.
            state.save();
        }
        assert_eq!(state.bytes(), (1 + CHECKPOINTS) * 4);
        // The two oldest are gone.
        assert_eq!(state.rewind(25), 0);
        let mut state = RecurrentState::new(&[1, 0, 0]);
        for turn in 1..=CHECKPOINTS + 2 {
            fill(&mut state, turn as f32);
            state.advance(turn * 10);
            state.save();
        }
        assert_eq!(state.rewind(35), 30);
        assert_eq!(state.block(0), [3.0]);
    }

    #[test]
    fn shifting_keeps_the_state_and_drops_saved_ones() {
        let mut state = RecurrentState::new(&[1, 0, 0]);
        fill(&mut state, 7.0);
        state.advance(20);
        state.save();
        state.shift(4, 6);
        assert_eq!(state.block(0), [7.0]);
        assert_eq!(state.bytes(), 4);
        // Now at 14 positions, with nothing saved to go back to.
        assert_eq!(state.rewind(14), 14);
        assert_eq!(state.rewind(13), 0);
    }

    #[test]
    fn models_without_recurrent_blocks_follow_any_rewind() {
        let mut state = RecurrentState::new(&[0, 0]);
        state.advance(9);
        state.save();
        assert_eq!(state.bytes(), 0);
        assert_eq!(state.rewind(5), 5);
    }
}

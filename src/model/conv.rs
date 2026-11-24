//! The state of the short convolution blocks, and how to go back in it.
//!
//! A short convolution (LFM2, LFM3) sees its input at the last few
//! positions only. Its state is those inputs: `taps - 1` vectors per block,
//! a few kilobytes, where an attention block needs a cache that grows with
//! the context.
//!
//! The cost of so little state comes when the context is cut back, as when
//! a new prompt shares only part of the old one. The inputs before the cut
//! are gone. Two things bring them back:
//!
//! - Each block keeps the inputs of the last [`RECENT`] positions as well,
//!   in a ring. Cutting back that far costs nothing. This is the usual case
//!   in a chat, where the next prompt repeats the conversation so far.
//! - The state of every block is saved each [`CHECKPOINT_EVERY`] positions.
//!   Going back further restores the last checkpoint before the cut, and
//!   the session reads the tokens after it again (see `Session::truncate`).
//!
//! Keeping every position instead would cost a vector per block and
//! position, more than the attention cache of these models.

/// Positions between two saved states.
pub const CHECKPOINT_EVERY: usize = 256;

/// Positions the context can be cut back without reading tokens again.
pub const RECENT: usize = 8;

/// The inputs that the convolution blocks still need.
pub struct ConvState {
    /// For each block, its index in `rings`, or `None` for other blocks.
    slots: Vec<Option<usize>>,
    /// Earlier positions each output reads: `taps - 1`.
    rows: usize,
    /// Floats in one input.
    n_embd: usize,
    /// Inputs each ring holds: `rows + RECENT`.
    capacity: usize,
    /// Per convolution block, its latest inputs. Position `p` is at row
    /// `p % capacity`.
    rings: Vec<Vec<f32>>,
    /// Positions evaluated so far.
    len: usize,
    /// The first position the rings still hold.
    valid_from: usize,
    /// Saved states: a position `p`, then each block's inputs at the `rows`
    /// positions before `p`, block after block.
    checkpoints: Vec<(usize, Vec<f32>)>,
}

impl ConvState {
    /// An empty state. `is_conv` marks the convolution blocks, which sum
    /// `taps` positions of `n_embd` floats.
    pub fn new(is_conv: &[bool], taps: usize, n_embd: usize) -> ConvState {
        let mut slots = Vec::with_capacity(is_conv.len());
        let mut count = 0;
        for &conv in is_conv {
            slots.push(conv.then(|| {
                count += 1;
                count - 1
            }));
        }
        let rows = taps.saturating_sub(1);
        let capacity = rows + RECENT;
        ConvState {
            slots,
            rows,
            n_embd,
            capacity,
            rings: vec![vec![0.0; capacity * n_embd]; count],
            len: 0,
            valid_from: 0,
            checkpoints: Vec::new(),
        }
    }

    /// Earlier positions each output reads: `taps - 1`.
    pub fn rows(&self) -> usize {
        self.rows
    }

    fn slot(&self, layer: usize) -> usize {
        self.slots[layer].expect("layer is a convolution block")
    }

    /// Fills `state` with block `layer`'s inputs at the `rows` positions
    /// before `position`, oldest first. Positions before the start of the
    /// context read as zero.
    pub fn read(&self, layer: usize, position: usize, state: &mut [f32]) {
        let ring = &self.rings[self.slot(layer)];
        let d = self.n_embd;
        for (index, row) in state.chunks_exact_mut(d).enumerate() {
            match (position + index).checked_sub(self.rows) {
                None => row.fill(0.0),
                Some(earlier) => {
                    debug_assert!(earlier >= self.valid_from && earlier < self.len);
                    let at = (earlier % self.capacity) * d;
                    row.copy_from_slice(&ring[at..at + d]);
                }
            }
        }
    }

    /// Records block `layer`'s inputs for a batch of `count` tokens.
    ///
    /// `history` holds the inputs from position `first - rows` up to
    /// `first + count`: the state [`read`](Self::read) gave, then one row
    /// per token. A checkpoint is saved at every multiple of
    /// [`CHECKPOINT_EVERY`] the batch reaches.
    pub fn write(&mut self, layer: usize, first: usize, count: usize, history: &[f32]) {
        let slot = self.slot(layer);
        let (d, rows) = (self.n_embd, self.rows);
        let mut position = (first / CHECKPOINT_EVERY + 1) * CHECKPOINT_EVERY;
        while position <= first + count {
            // The state at `position` is the `rows` inputs before it.
            let start = (position - first) * d;
            self.save(slot, position, &history[start..start + rows * d]);
            position += CHECKPOINT_EVERY;
        }
        // Only the last `capacity` tokens of a long batch stay in the ring.
        let ring = &mut self.rings[slot];
        for token in count.saturating_sub(self.capacity)..count {
            let at = ((first + token) % self.capacity) * d;
            let row = (rows + token) * d;
            ring[at..at + d].copy_from_slice(&history[row..row + d]);
        }
    }

    /// Marks the positions before `end` as evaluated, once every block has
    /// written its part of a batch.
    pub fn advance(&mut self, end: usize) {
        self.len = end;
        self.valid_from = self.valid_from.max(end.saturating_sub(self.capacity));
    }

    /// Stores block `slot`'s state for `position` in that position's
    /// checkpoint, making the checkpoint when it is the first block there.
    fn save(&mut self, slot: usize, position: usize, state: &[f32]) {
        let len = self.rows * self.n_embd;
        let index = match self
            .checkpoints
            .iter()
            .rposition(|(saved, _)| *saved == position)
        {
            Some(index) => index,
            None => {
                self.checkpoints
                    .push((position, vec![0.0; self.rings.len() * len]));
                self.checkpoints.len() - 1
            }
        };
        self.checkpoints[index].1[slot * len..(slot + 1) * len].copy_from_slice(state);
    }

    /// Cuts the context back to `len` positions, and returns the position
    /// the state now stands at. That is `len` when the rings still hold the
    /// inputs before it. Otherwise it is the last checkpoint at or before
    /// `len`, or zero, and the tokens from there to `len` must be evaluated
    /// again.
    pub fn rewind(&mut self, len: usize) -> usize {
        self.checkpoints.retain(|(saved, _)| *saved <= len);
        if self.rings.is_empty() || len.saturating_sub(self.rows) >= self.valid_from {
            self.len = len;
            return len;
        }
        let (d, rows) = (self.n_embd, self.rows);
        let position = match self.checkpoints.last() {
            Some((position, saved)) => {
                for (ring, state) in self.rings.iter_mut().zip(saved.chunks_exact(rows * d)) {
                    for (index, row) in state.chunks_exact(d).enumerate() {
                        let at = ((position - rows + index) % self.capacity) * d;
                        ring[at..at + d].copy_from_slice(row);
                    }
                }
                *position
            }
            None => 0,
        };
        self.len = position;
        self.valid_from = position.saturating_sub(rows);
        position
    }

    /// Removes positions `keep..keep + discard` and moves the later ones
    /// down, as the session does when its context slides. A block's state
    /// does not depend on positions, so only where the rings keep each
    /// input changes. Checkpoints are dropped: their positions are gone.
    pub fn shift(&mut self, keep: usize, discard: usize) {
        let discard = discard.min(self.len.saturating_sub(keep));
        self.checkpoints.clear();
        if discard == 0 || self.rings.is_empty() {
            self.len -= discard;
            return;
        }
        let d = self.n_embd;
        let first = self
            .valid_from
            .max(self.len.saturating_sub(self.capacity))
            .max(keep + discard);
        let mut moved = vec![0.0; (self.len - first) * d];
        for ring in &mut self.rings {
            for (index, position) in (first..self.len).enumerate() {
                let at = (position % self.capacity) * d;
                moved[index * d..(index + 1) * d].copy_from_slice(&ring[at..at + d]);
            }
            for (index, position) in (first - discard..self.len - discard).enumerate() {
                let at = (position % self.capacity) * d;
                ring[at..at + d].copy_from_slice(&moved[index * d..(index + 1) * d]);
            }
        }
        self.len -= discard;
        self.valid_from = first - discard;
    }

    /// Bytes of memory the rings and checkpoints take.
    pub fn bytes(&self) -> usize {
        let floats = self.rings.iter().map(Vec::len).sum::<usize>()
            + self
                .checkpoints
                .iter()
                .map(|(_, saved)| saved.len())
                .sum::<usize>();
        floats * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs batches over positions `from..to` in every block, with the
    /// input of position `p` holding `p + 1` in every element.
    fn feed(state: &mut ConvState, layers: &[usize], from: usize, to: usize, batch: usize) {
        let (rows, d) = (state.rows, state.n_embd);
        let mut first = from;
        while first < to {
            let count = batch.min(to - first);
            for &layer in layers {
                let mut history = vec![0.0; (rows + count) * d];
                state.read(layer, first, &mut history[..rows * d]);
                for token in 0..count {
                    let row = (rows + token) * d;
                    history[row..row + d].fill((first + token + 1) as f32);
                }
                state.write(layer, first, count, &history);
            }
            first += count;
            state.advance(first);
        }
    }

    /// The state a block would read at `position`.
    fn state_at(state: &ConvState, layer: usize, position: usize) -> Vec<f32> {
        let mut out = vec![-1.0; state.rows * state.n_embd];
        state.read(layer, position, &mut out);
        out
    }

    /// The state the inputs of `feed` give at `position`.
    fn expected(rows: usize, d: usize, position: usize) -> Vec<f32> {
        (0..rows)
            .flat_map(|index| {
                let earlier = (position + index).checked_sub(rows);
                vec![earlier.map_or(0.0, |p| (p + 1) as f32); d]
            })
            .collect()
    }

    #[test]
    fn the_state_holds_the_last_inputs() {
        let mut state = ConvState::new(&[false, true], 3, 2);
        assert_eq!(state.rows(), 2);
        assert_eq!(state_at(&state, 1, 0), [0.0; 4]);
        feed(&mut state, &[1], 0, 1, 4);
        assert_eq!(state_at(&state, 1, 1), [0.0, 0.0, 1.0, 1.0]);
        feed(&mut state, &[1], 1, 40, 7);
        assert_eq!(state_at(&state, 1, 40), expected(2, 2, 40));
    }

    #[test]
    fn short_cuts_use_the_ring() {
        let mut state = ConvState::new(&[true], 4, 3);
        feed(&mut state, &[0], 0, 100, 32);
        // Back as far as the ring reaches, at no cost.
        for len in [100, 99, 95, 92] {
            assert_eq!(state.rewind(len), len);
            assert_eq!(state_at(&state, 0, len), expected(3, 3, len));
        }
        // Going on from there overwrites what came after.
        feed(&mut state, &[0], 92, 96, 1);
        assert_eq!(state_at(&state, 0, 96), expected(3, 3, 96));
    }

    #[test]
    fn long_cuts_restore_a_checkpoint() {
        let mut state = ConvState::new(&[true, false, true], 3, 1);
        feed(&mut state, &[0, 2], 0, 3 * CHECKPOINT_EVERY + 50, 100);
        let rings = 2 * (2 + RECENT) * 4;
        let saved = 3 * (2 * 2 * 4);
        assert_eq!(state.bytes(), rings + saved);

        // Between checkpoints: back to the one before the cut.
        let at = state.rewind(2 * CHECKPOINT_EVERY + 7);
        assert_eq!(at, 2 * CHECKPOINT_EVERY);
        for layer in [0, 2] {
            assert_eq!(state_at(&state, layer, at), expected(2, 1, at));
        }
        // The session then reads the tokens up to the cut again.
        feed(&mut state, &[0, 2], at, at + 7, 4);
        assert_eq!(state_at(&state, 0, at + 7), expected(2, 1, at + 7));

        // Later checkpoints are gone and earlier ones stay.
        assert_eq!(state.bytes(), rings + 2 * (2 * 2 * 4));
        assert_eq!(state.rewind(CHECKPOINT_EVERY + 1), CHECKPOINT_EVERY);
        assert_eq!(state.rewind(10), 0);
        assert_eq!(state_at(&state, 0, 0), [0.0, 0.0]);
        assert_eq!(state.bytes(), rings);
    }

    #[test]
    fn shifting_keeps_the_recent_inputs() {
        let mut state = ConvState::new(&[true], 3, 1);
        feed(&mut state, &[0], 0, 300, 32);
        state.shift(1, 100);
        // The state at the new end is the inputs of the old positions 298
        // and 299, which held 299 and 300.
        assert_eq!(state_at(&state, 0, 200), [299.0, 300.0]);
        assert_eq!(state.bytes(), (2 + RECENT) * 4);
    }

    #[test]
    fn models_without_convolutions_cut_anywhere() {
        let mut state = ConvState::new(&[false, false], 3, 4);
        state.advance(500);
        assert_eq!(state.rewind(3), 3);
        assert_eq!(state.bytes(), 0);
    }
}

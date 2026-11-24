//! The key/value cache: what every earlier token contributes to attention.
//!
//! Each layer stores one key vector and one value vector per position. The
//! cache grows in fixed-size chunks as positions are filled, so:
//!
//! - Memory tracks the tokens actually in context, not the configured limit.
//! - The context window has no built-in maximum. It ends where memory does.
//! - Nothing is reserved up front, which matters on systems that refuse to
//!   overcommit memory and on 32-bit address spaces.

/// Positions held by one chunk.
pub const CHUNK_POSITIONS: usize = 256;

/// The keys and values of one head of one layer, as a list of chunks.
///
/// Storing each head on its own keeps its positions back to back in memory.
/// Attention reads one head at a time, so it walks a plain sequential run.
#[derive(Default)]
struct HeadCache {
    keys: Vec<Box<[f32]>>,
    values: Vec<Box<[f32]>>,
}

/// Keys and values for every layer, head and position.
pub struct KvCache {
    /// Key/value heads per layer.
    n_heads: usize,
    /// Width of one head's key or value vector.
    head_dim: usize,
    /// One cache per layer and head, layer by layer.
    heads: Vec<HeadCache>,
    /// Positions that hold valid data.
    len: usize,
}

impl KvCache {
    /// An empty cache for `n_layer` layers of `n_heads` key/value heads,
    /// each `head_dim` wide.
    pub fn new(n_layer: usize, n_heads: usize, head_dim: usize) -> KvCache {
        let heads = (0..n_layer * n_heads)
            .map(|_| HeadCache::default())
            .collect();
        KvCache {
            n_heads,
            head_dim,
            heads,
            len: 0,
        }
    }

    /// Positions that hold valid data.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Bytes of memory currently allocated.
    pub fn allocated_bytes(&self) -> usize {
        let chunks: usize = self
            .heads
            .iter()
            .map(|head| head.keys.len() + head.values.len())
            .sum();
        chunks * CHUNK_POSITIONS * self.head_dim * std::mem::size_of::<f32>()
    }

    /// Makes room for positions `0..positions` in every layer.
    pub fn reserve(&mut self, positions: usize) {
        let chunks = positions.div_ceil(CHUNK_POSITIONS);
        let chunk_len = CHUNK_POSITIONS * self.head_dim;
        for head in &mut self.heads {
            while head.keys.len() < chunks {
                head.keys.push(vec![0.0; chunk_len].into_boxed_slice());
                head.values.push(vec![0.0; chunk_len].into_boxed_slice());
            }
        }
    }

    /// Marks positions `0..len` as valid. They must have been written.
    pub fn set_len(&mut self, len: usize) {
        debug_assert!(
            self.heads
                .iter()
                .all(|head| head.keys.len() * CHUNK_POSITIONS >= len)
        );
        self.len = len;
    }

    /// Forgets every position from `len` on, and frees the chunks that no
    /// longer hold anything.
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }
        self.len = len;
        let chunks = len.div_ceil(CHUNK_POSITIONS);
        for head in &mut self.heads {
            head.keys.truncate(chunks);
            head.values.truncate(chunks);
        }
    }

    /// Stores the key and value of `position` in `layer`. Both hold every
    /// head side by side. The position must have been reserved.
    pub fn write(&mut self, layer: usize, position: usize, key: &[f32], value: &[f32]) {
        let (chunk, start) = (
            position / CHUNK_POSITIONS,
            (position % CHUNK_POSITIONS) * self.head_dim,
        );
        let heads = &mut self.heads[layer * self.n_heads..(layer + 1) * self.n_heads];
        let parts = key
            .chunks_exact(self.head_dim)
            .zip(value.chunks_exact(self.head_dim));
        for (head, (key, value)) in heads.iter_mut().zip(parts) {
            head.keys[chunk][start..start + self.head_dim].copy_from_slice(key);
            head.values[chunk][start..start + self.head_dim].copy_from_slice(value);
        }
    }

    /// Calls `visit` with each stretch of contiguous positions in
    /// `first..end` of one head: the first position of the stretch, then
    /// its keys and values, `head_dim` floats per position.
    pub fn for_each_run(
        &self,
        layer: usize,
        head: usize,
        first: usize,
        end: usize,
        mut visit: impl FnMut(usize, &[f32], &[f32]),
    ) {
        let cache = &self.heads[layer * self.n_heads + head];
        let mut position = first;
        while position < end {
            let chunk = position / CHUNK_POSITIONS;
            let run_end = end.min((chunk + 1) * CHUNK_POSITIONS);
            let start = (position % CHUNK_POSITIONS) * self.head_dim;
            let stop = start + (run_end - position) * self.head_dim;
            visit(
                position,
                &cache.keys[chunk][start..stop],
                &cache.values[chunk][start..stop],
            );
            position = run_end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A recognizable vector for a layer, head and position.
    fn pattern(layer: usize, head: usize, position: usize, dim: usize, offset: f32) -> Vec<f32> {
        (0..dim)
            .map(|i| (layer * 1_000_000 + head * 100_000 + position * 10 + i) as f32 + offset)
            .collect()
    }

    /// The key or value of every head of a position, side by side.
    fn row(layer: usize, heads: usize, position: usize, dim: usize, offset: f32) -> Vec<f32> {
        (0..heads)
            .flat_map(|head| pattern(layer, head, position, dim, offset))
            .collect()
    }

    fn fill(cache: &mut KvCache, layers: usize, heads: usize, positions: usize, dim: usize) {
        cache.reserve(positions);
        for layer in 0..layers {
            for position in 0..positions {
                let key = row(layer, heads, position, dim, 0.0);
                let value = row(layer, heads, position, dim, 0.5);
                cache.write(layer, position, &key, &value);
            }
        }
        cache.set_len(positions);
    }

    #[test]
    fn written_positions_read_back_across_chunks() {
        let (layers, heads, dim, positions) = (2, 3, 4, CHUNK_POSITIONS * 2 + 17);
        let mut cache = KvCache::new(layers, heads, dim);
        assert_eq!(cache.len(), 0);
        fill(&mut cache, layers, heads, positions, dim);
        assert_eq!(cache.len(), positions);

        for layer in 0..layers {
            for head in 0..heads {
                let mut seen = 0;
                cache.for_each_run(layer, head, 0, positions, |first, keys, values| {
                    assert_eq!(first, seen);
                    for (i, (key, value)) in keys
                        .chunks_exact(dim)
                        .zip(values.chunks_exact(dim))
                        .enumerate()
                    {
                        assert_eq!(key, pattern(layer, head, first + i, dim, 0.0));
                        assert_eq!(value, pattern(layer, head, first + i, dim, 0.5));
                    }
                    seen += keys.len() / dim;
                });
                assert_eq!(seen, positions);
            }
        }
    }

    #[test]
    fn runs_respect_the_requested_window() {
        let dim = 2;
        let mut cache = KvCache::new(1, 2, dim);
        fill(&mut cache, 1, 2, CHUNK_POSITIONS + 10, dim);
        let mut runs = Vec::new();
        cache.for_each_run(
            0,
            1,
            CHUNK_POSITIONS - 3,
            CHUNK_POSITIONS + 4,
            |first, keys, _| {
                runs.push((first, keys.len() / dim));
                assert_eq!(&keys[..dim], pattern(0, 1, first, dim, 0.0));
            },
        );
        assert_eq!(runs, [(CHUNK_POSITIONS - 3, 3), (CHUNK_POSITIONS, 4)]);
        cache.for_each_run(0, 0, 5, 5, |_, _, _| panic!("an empty window has no runs"));
    }

    #[test]
    fn memory_follows_the_filled_length() {
        let (layers, heads, dim) = (3, 2, 4);
        let mut cache = KvCache::new(layers, heads, dim);
        assert_eq!(cache.allocated_bytes(), 0);
        fill(&mut cache, layers, heads, 1, dim);
        let one_chunk = layers * heads * 2 * CHUNK_POSITIONS * dim * 4;
        assert_eq!(cache.allocated_bytes(), one_chunk);
        fill(&mut cache, layers, heads, CHUNK_POSITIONS + 1, dim);
        assert_eq!(cache.allocated_bytes(), 2 * one_chunk);

        cache.truncate(CHUNK_POSITIONS);
        assert_eq!(cache.len(), CHUNK_POSITIONS);
        assert_eq!(cache.allocated_bytes(), one_chunk);
        // Truncating never grows the cache.
        cache.truncate(CHUNK_POSITIONS * 5);
        assert_eq!(cache.len(), CHUNK_POSITIONS);
        cache.truncate(0);
        assert_eq!(cache.allocated_bytes(), 0);
    }
}

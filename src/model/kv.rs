//! The key/value cache: what every earlier token contributes to attention.
//!
//! Each layer stores one key vector and one value vector per position. The
//! cache grows in fixed-size chunks as positions are filled, so:
//!
//! - Memory tracks the tokens actually in context, not the configured limit.
//! - The context window has no built-in maximum. It ends where memory does.
//! - Nothing is reserved up front, which matters on systems that refuse to
//!   overcommit memory and on 32-bit address spaces.
//!
//! Layers can differ in their number of heads and head width, and a layer
//! without a cache of its own, such as a convolution block, takes no space.
//! Values may be narrower than keys, or absent: latent attention reads its
//! values out of its keys.

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

/// Where one layer's heads are in the cache.
#[derive(Clone, Copy)]
struct LayerShape {
    /// Index of the layer's first head in `heads`.
    first: usize,
    /// Key/value heads of the layer. Zero for a layer without a cache.
    n_heads: usize,
    /// Width of one head's key vector.
    head_dim: usize,
    /// Width of one head's value vector. Zero when the layer keeps keys
    /// only.
    value_dim: usize,
}

/// Keys and values for every layer, head and position.
pub struct KvCache {
    layers: Vec<LayerShape>,
    /// One cache per layer and head, layer by layer.
    heads: Vec<HeadCache>,
    /// Width of each head in `heads`: its keys, then its values.
    head_dims: Vec<usize>,
    value_dims: Vec<usize>,
    /// Positions that hold valid data.
    len: usize,
}

impl KvCache {
    /// An empty cache. `shapes` gives each layer's number of key/value
    /// heads, the width of their keys and the width of their values. Zero
    /// heads means the layer has no cache, and a value width of zero that
    /// it keeps keys only.
    pub fn new(shapes: &[(usize, usize, usize)]) -> KvCache {
        let mut layers = Vec::with_capacity(shapes.len());
        let mut head_dims = Vec::new();
        let mut value_dims = Vec::new();
        for &(n_heads, head_dim, value_dim) in shapes {
            layers.push(LayerShape {
                first: head_dims.len(),
                n_heads,
                head_dim,
                value_dim,
            });
            head_dims.extend(std::iter::repeat_n(head_dim, n_heads));
            value_dims.extend(std::iter::repeat_n(value_dim, n_heads));
        }
        KvCache {
            layers,
            heads: head_dims.iter().map(|_| HeadCache::default()).collect(),
            head_dims,
            value_dims,
            len: 0,
        }
    }

    /// Positions that hold valid data.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Bytes of memory currently allocated.
    pub fn allocated_bytes(&self) -> usize {
        self.heads
            .iter()
            .zip(self.head_dims.iter().zip(&self.value_dims))
            .map(|(head, (dim, value_dim))| {
                (head.keys.len() * dim + head.values.len() * value_dim) * CHUNK_POSITIONS
            })
            .sum::<usize>()
            * std::mem::size_of::<f32>()
    }

    /// Makes room for positions `0..positions` in every layer.
    pub fn reserve(&mut self, positions: usize) {
        let chunks = positions.div_ceil(CHUNK_POSITIONS);
        let dims = self.head_dims.iter().zip(&self.value_dims);
        for (head, (&dim, &value_dim)) in self.heads.iter_mut().zip(dims) {
            while head.keys.len() < chunks {
                head.keys
                    .push(vec![0.0; CHUNK_POSITIONS * dim].into_boxed_slice());
                // An empty chunk allocates nothing.
                head.values
                    .push(vec![0.0; CHUNK_POSITIONS * value_dim].into_boxed_slice());
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

    /// Removes positions `keep..keep + discard` and moves the later ones
    /// down to close the gap. `fix_key` is called with each moved key, one
    /// head at a time, so the caller can rotate it to its new position.
    pub fn shift(
        &mut self,
        keep: usize,
        discard: usize,
        mut fix_key: impl FnMut(usize, &mut [f32]),
    ) {
        let len = self.len;
        let discard = discard.min(len.saturating_sub(keep));
        if discard == 0 {
            return;
        }
        let at = |position: usize, dim: usize| {
            (
                position / CHUNK_POSITIONS,
                (position % CHUNK_POSITIONS) * dim,
            )
        };
        for (layer, shape) in self.layers.iter().enumerate() {
            let (dim, value_dim) = (shape.head_dim, shape.value_dim);
            let (mut row, mut value_row) = (vec![0.0; dim], vec![0.0; value_dim]);
            for head in &mut self.heads[shape.first..shape.first + shape.n_heads] {
                // Moving down in increasing order never overwrites a row
                // that is still to be moved.
                for from in keep + discard..len {
                    let ((source, start), (target, place)) =
                        (at(from, dim), at(from - discard, dim));
                    row.copy_from_slice(&head.keys[source][start..start + dim]);
                    fix_key(layer, &mut row);
                    head.keys[target][place..place + dim].copy_from_slice(&row);
                    let ((source, start), (target, place)) =
                        (at(from, value_dim), at(from - discard, value_dim));
                    value_row.copy_from_slice(&head.values[source][start..start + value_dim]);
                    head.values[target][place..place + value_dim].copy_from_slice(&value_row);
                }
            }
        }
        self.truncate(len - discard);
    }

    /// Stores the key and value of `position` in `layer`. Both hold every
    /// head side by side. The position must have been reserved. A layer
    /// that keeps keys only takes an empty value.
    pub fn write(&mut self, layer: usize, position: usize, key: &[f32], value: &[f32]) {
        let shape = self.layers[layer];
        let (dim, value_dim) = (shape.head_dim, shape.value_dim);
        let chunk = position / CHUNK_POSITIONS;
        let (start, value_start) = (
            (position % CHUNK_POSITIONS) * dim,
            (position % CHUNK_POSITIONS) * value_dim,
        );
        let heads = &mut self.heads[shape.first..shape.first + shape.n_heads];
        for (index, head) in heads.iter_mut().enumerate() {
            head.keys[chunk][start..start + dim].copy_from_slice(&key[index * dim..][..dim]);
            head.values[chunk][value_start..value_start + value_dim]
                .copy_from_slice(&value[index * value_dim..][..value_dim]);
        }
    }

    /// Calls `visit` with each stretch of contiguous positions in
    /// `first..end` of one head: the first position of the stretch, then
    /// its keys and values, `head_dim` and `value_dim` floats per position.
    pub fn for_each_run(
        &self,
        layer: usize,
        head: usize,
        first: usize,
        end: usize,
        mut visit: impl FnMut(usize, &[f32], &[f32]),
    ) {
        let shape = self.layers[layer];
        let cache = &self.heads[shape.first + head];
        let (dim, value_dim) = (shape.head_dim, shape.value_dim);
        let mut position = first;
        while position < end {
            let chunk = position / CHUNK_POSITIONS;
            let run_end = end.min((chunk + 1) * CHUNK_POSITIONS);
            let offset = position % CHUNK_POSITIONS;
            let run = run_end - position;
            visit(
                position,
                &cache.keys[chunk][offset * dim..(offset + run) * dim],
                &cache.values[chunk][offset * value_dim..(offset + run) * value_dim],
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
        let mut cache = KvCache::new(&[(heads, dim, dim); 2]);
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
        let mut cache = KvCache::new(&[(2, dim, dim)]);
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
        let mut cache = KvCache::new(&[(heads, dim, dim); 3]);
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

    #[test]
    fn shifting_closes_the_gap() {
        let mut cache = KvCache::new(&[(1, 2, 2), (0, 0, 0)]);
        let positions = CHUNK_POSITIONS + 10;
        cache.reserve(positions);
        for position in 0..positions {
            let value = position as f32;
            cache.write(0, position, &[value, -value], &[value, value]);
        }
        cache.set_len(positions);
        // Drop positions 3 to 102. Every moved key is marked.
        cache.shift(3, 100, |layer, key| {
            assert_eq!(layer, 0);
            key[1] = 1.0;
        });
        assert_eq!(cache.len(), positions - 100);
        let mut seen = Vec::new();
        cache.for_each_run(0, 0, 0, cache.len(), |_, keys, values| {
            for (key, value) in keys.chunks_exact(2).zip(values.chunks_exact(2)) {
                seen.push((key[0], key[1], value[0]));
            }
        });
        assert_eq!(seen[2], (2.0, -2.0, 2.0));
        assert_eq!(seen[3], (103.0, 1.0, 103.0));
        assert_eq!(
            seen.last(),
            Some(&((positions - 1) as f32, 1.0, (positions - 1) as f32))
        );
    }

    #[test]
    fn layers_can_differ_or_have_no_cache() {
        // A wide layer, a layer without a cache, and a narrow one.
        let mut cache = KvCache::new(&[(1, 8, 8), (0, 0, 0), (2, 4, 4)]);
        cache.reserve(1);
        cache.write(0, 0, &[1.0; 8], &[2.0; 8]);
        cache.write(2, 0, &[3.0; 8], &[4.0; 8]);
        cache.set_len(1);
        cache.for_each_run(0, 0, 0, 1, |_, keys, values| {
            assert_eq!((keys, values), (&[1.0; 8][..], &[2.0; 8][..]));
        });
        cache.for_each_run(2, 1, 0, 1, |_, keys, values| {
            assert_eq!((keys, values), (&[3.0; 4][..], &[4.0; 4][..]));
        });
        // Three heads of eight or four floats, keys and values.
        assert_eq!(
            cache.allocated_bytes(),
            2 * CHUNK_POSITIONS * (8 + 4 + 4) * 4
        );
    }

    #[test]
    fn values_can_be_narrower_than_keys_or_absent() {
        // A layer that keeps keys only, as latent attention does, and one
        // whose two heads have values half as wide as their keys.
        let mut cache = KvCache::new(&[(1, 6, 0), (2, 4, 2)]);
        let positions = CHUNK_POSITIONS + 5;
        cache.reserve(positions);
        for position in 0..positions {
            let p = position as f32;
            cache.write(0, position, &[p; 6], &[]);
            let values = [p + 0.5, p + 0.5, -p - 0.5, -p - 0.5];
            cache.write(1, position, &[p, p, p, p, -p, -p, -p, -p], &values);
        }
        cache.set_len(positions);
        // Keys only: six floats, then nothing, per position.
        assert_eq!(
            cache.allocated_bytes(),
            2 * CHUNK_POSITIONS * (6 + 2 * (4 + 2)) * 4
        );
        // Drop positions 2 to 101. Every moved key is marked.
        cache.shift(2, 100, |_, key| key[0] += 1000.0);
        let mut seen = Vec::new();
        cache.for_each_run(0, 0, 0, cache.len(), |first, keys, values| {
            assert!(values.is_empty());
            for (index, key) in keys.chunks_exact(6).enumerate() {
                seen.push((first + index, key[0], key[5]));
            }
        });
        assert_eq!(seen.len(), positions - 100);
        assert_eq!(seen[1], (1, 1.0, 1.0));
        assert_eq!(seen[2], (2, 1102.0, 102.0));
        cache.for_each_run(1, 1, 0, cache.len(), |first, keys, values| {
            for (index, (key, value)) in
                keys.chunks_exact(4).zip(values.chunks_exact(2)).enumerate()
            {
                let p = (if first + index < 2 {
                    first + index
                } else {
                    first + index + 100
                }) as f32;
                let moved = if first + index < 2 { 0.0 } else { 1000.0 };
                assert_eq!(key, [-p + moved, -p, -p, -p]);
                assert_eq!(value, [-p - 0.5, -p - 0.5]);
            }
        });
    }
}

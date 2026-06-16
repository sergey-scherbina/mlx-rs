use mlx_rs::{
    error::Exception,
    ops::{
        concatenate_axis,
        indexing::{IndexOp, TryIndexMutOp},
        zeros_dtype,
    },
    Array,
};

// TODO: somehow move quantized methods to a separate trait?
pub trait KeyValueCache {
    fn is_quantized(&self) -> bool {
        false
    }

    /// Returns the group size used for quantization. `None` if not quantized.
    fn group_size(&self) -> Option<i32> {
        None
    }

    /// Returns the number of bits used for quantization. `None` if not quantized.
    fn bits(&self) -> Option<i32> {
        None
    }

    fn offset(&self) -> i32;

    fn max_size(&self) -> Option<i32>;

    fn update_and_fetch(&mut self, keys: Array, values: Array)
        -> Result<(Array, Array), Exception>;

    /// Push the cache's live arrays into `out` so the caller can `eval` them — used
    /// to materialize (and free) each prefill chunk's activations before the next
    /// chunk, bounding the prefill memory peak. Default: nothing (caches with no
    /// persistent arrays, or that don't benefit, opt out).
    fn collect_eval<'a>(&'a self, _out: &mut Vec<&'a Array>) {}
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn is_quantized(&self) -> bool {
        T::is_quantized(self)
    }

    fn group_size(&self) -> Option<i32> {
        T::group_size(self)
    }

    fn bits(&self) -> Option<i32> {
        T::bits(self)
    }

    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values)
    }

    fn collect_eval<'a>(&'a self, out: &mut Vec<&'a Array>) {
        T::collect_eval(self, out)
    }
}

/// Block size (positions) the KV buffers grow by. Matches Python `mlx_lm`'s
/// `KVCache.step`.
const KV_STEP: i32 = 256;

/// KV cache that pre-allocates its key/value buffers in [`KV_STEP`]-sized blocks
/// and writes each step in place (`slice_update`), returning a view of the used
/// prefix — instead of re-`concatenate`-ing (and reallocating) the entire history
/// every decode step. Mirrors Python `mlx_lm`'s `KVCache`: byte-identical output,
/// but the per-step O(context) copy becomes an amortised O(1) write (one growth
/// `concatenate` every [`KV_STEP`] steps). `offset` is the used length; the
/// buffers' `[-2]` length is the (>= offset) capacity. (The name is kept for
/// call-site compatibility; it no longer concatenates per step.)
#[derive(Debug, Clone, Default)]
pub struct ConcatKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl ConcatKeyValueCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// The cache's live arrays (full pre-allocated buffers), for forcing
    /// materialization (e.g. between prefill chunks, so each chunk's activations
    /// are freed before the next).
    pub fn state_arrays(&self) -> impl Iterator<Item = &Array> {
        self.keys.iter().chain(self.values.iter())
    }

    /// Allocated capacity along the sequence axis (`>= offset`).
    fn capacity(&self) -> i32 {
        self.keys.as_ref().map_or(0, |k| {
            let s = k.shape();
            s[s.len() - 2]
        })
    }

    /// Drop the cache back to its first `len` positions, keeping the allocated
    /// buffers (the kept prefix `[0, len)` is untouched; later writes overwrite
    /// `[len, ..)`). Enables prefix reuse across requests: when a new prompt
    /// extends a previous one, truncate to the shared prefix length and prefill
    /// only the new suffix. A no-op if `len >= offset`. Byte-exact: reads use
    /// `index((.., .., ..offset, ..))`, and `[0, len)` was written by the same
    /// prefill the fresh path would run, so reusing it yields identical KV.
    pub fn truncate(&mut self, len: i32) {
        if len < self.offset {
            self.offset = len.max(0);
        }
    }

    /// The used `(keys, values)` prefix `[.., .., ..offset, ..]` (length `offset`),
    /// or `None` if empty. For assembling a batched cache from per-sequence caches.
    pub fn kv_used(&self) -> Option<(Array, Array, i32)> {
        match (&self.keys, &self.values) {
            (Some(k), Some(v)) if self.offset > 0 => Some((
                k.index((.., .., ..self.offset, ..)),
                v.index((.., .., ..self.offset, ..)),
                self.offset,
            )),
            _ => None,
        }
    }

    /// Build a cache directly from a `[B,H,L,D]` keys/values pair already at `offset`
    /// positions — for stacking per-sequence caches onto the batch axis (ragged
    /// batched decode). Subsequent `update_and_fetch` grows + appends past `offset`.
    pub fn from_kv(keys: Array, values: Array, offset: i32) -> Self {
        Self { keys: Some(keys), values: Some(values), offset }
    }
}

impl KeyValueCache for ConcatKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let ks = keys.shape().to_vec(); // [B, H, L, Dk]
        let (b, h, l, dk) = (ks[0], ks[1], ks[2], ks[3]);
        let dv = *values.shape().last().unwrap();
        let prev = self.offset;

        // Grow only when the buffer can't hold L more positions — i.e. roughly
        // once every KV_STEP decode steps, not every step.
        if self.keys.is_none() || prev + l > self.capacity() {
            let n_steps = (KV_STEP + l - 1) / KV_STEP;
            let add = n_steps * KV_STEP;
            let new_k = zeros_dtype(&[b, h, add, dk], keys.dtype())?;
            let new_v = zeros_dtype(&[b, h, add, dv], values.dtype())?;
            match (self.keys.take(), self.values.take()) {
                (Some(k), Some(v)) => {
                    // Drop a partially-filled trailing block before appending.
                    let (k, v) = if prev % KV_STEP != 0 {
                        (k.index((.., .., ..prev, ..)), v.index((.., .., ..prev, ..)))
                    } else {
                        (k, v)
                    };
                    self.keys = Some(concatenate_axis(&[k, new_k], -2)?);
                    self.values = Some(concatenate_axis(&[v, new_v], -2)?);
                }
                _ => {
                    self.keys = Some(new_k);
                    self.values = Some(new_v);
                }
            }
        }

        self.offset = prev + l;
        // Write this step's keys/values in place at [.., .., prev:offset, ..].
        self.keys
            .as_mut()
            .expect("keys")
            .try_index_mut((.., .., prev..self.offset, ..), keys)?;
        self.values
            .as_mut()
            .expect("values")
            .try_index_mut((.., .., prev..self.offset, ..), values)?;

        // Return a view of the used prefix (length == offset).
        Ok((
            self.keys
                .as_ref()
                .unwrap()
                .index((.., .., ..self.offset, ..)),
            self.values
                .as_ref()
                .unwrap()
                .index((.., .., ..self.offset, ..)),
        ))
    }

    fn collect_eval<'a>(&'a self, out: &mut Vec<&'a Array>) {
        out.extend(self.state_arrays());
    }
}

/// TODO: A generic KV Cache
pub struct DefaultKeyValueCache {}

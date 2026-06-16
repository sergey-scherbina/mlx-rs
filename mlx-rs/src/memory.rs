//! Metal memory introspection — bytes MLX has allocated on the unified-memory device.
//!
//! These reflect the GPU/Metal buffers (model weights, KV cache, activations), which a
//! process's RSS does NOT capture on Apple Silicon. Use them to observe how much memory a
//! resident model holds and to confirm it is reclaimed when the model is dropped.

/// Bytes currently allocated and in use by MLX's Metal allocator (live buffers — model
/// weights + caches + in-flight activations). Drops back toward zero once the arrays
/// holding them are freed. Returns 0 if the query fails (e.g. no Metal device).
pub fn get_active_memory() -> usize {
    let mut res: usize = 0;
    // SAFETY: `res` is a valid out-pointer; the C call only writes the byte count.
    unsafe {
        mlx_sys::mlx_get_active_memory(&mut res as *mut usize);
    }
    res
}

/// High-water mark of active memory since the last [`reset_peak_memory`]. Returns 0 on
/// query failure.
pub fn get_peak_memory() -> usize {
    let mut res: usize = 0;
    // SAFETY: see [`get_active_memory`].
    unsafe {
        mlx_sys::mlx_get_peak_memory(&mut res as *mut usize);
    }
    res
}

/// Bytes held in MLX's buffer cache (freed buffers kept for reuse, not returned to the OS).
/// Returns 0 on query failure.
pub fn get_cache_memory() -> usize {
    let mut res: usize = 0;
    // SAFETY: see [`get_active_memory`].
    unsafe {
        mlx_sys::mlx_get_cache_memory(&mut res as *mut usize);
    }
    res
}

/// Reset the peak-memory counter to the current active memory.
pub fn reset_peak_memory() {
    // SAFETY: no arguments; resets an internal counter.
    unsafe {
        mlx_sys::mlx_reset_peak_memory();
    }
}

/// Set the buffer-cache limit (bytes): freed Metal buffers above this are returned to the
/// OS on the next allocation instead of kept for reuse. Lowering it keeps the resident
/// footprint near the *active* memory (weights + live KV) at the cost of some re-allocation
/// — important on a shared host so one model doesn't hoard all of unified memory. Returns
/// the previous limit.
pub fn set_cache_limit(limit: usize) -> usize {
    let mut prev: usize = 0;
    // SAFETY: `prev` is a valid out-pointer; the C call writes the old limit and sets the new.
    unsafe {
        mlx_sys::mlx_set_cache_limit(&mut prev as *mut usize, limit);
    }
    prev
}

/// Set the soft memory limit (bytes) for graph evaluation — allocations beyond it wait or
/// relax rather than grab more unified memory. Returns the previous limit.
pub fn set_memory_limit(limit: usize) -> usize {
    let mut prev: usize = 0;
    // SAFETY: see [`set_cache_limit`].
    unsafe {
        mlx_sys::mlx_set_memory_limit(&mut prev as *mut usize, limit);
    }
    prev
}

/// Set the wired (non-pageable) memory limit (bytes). Returns the previous limit.
pub fn set_wired_limit(limit: usize) -> usize {
    let mut prev: usize = 0;
    // SAFETY: see [`set_cache_limit`].
    unsafe {
        mlx_sys::mlx_set_wired_limit(&mut prev as *mut usize, limit);
    }
    prev
}

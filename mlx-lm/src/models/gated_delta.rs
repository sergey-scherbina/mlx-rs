//! GatedDeltaNet recurrence for Qwen3-Next / Qwen3.6 linear-attention layers,
//! ported from Python `mlx_lm.models.gated_delta`. Two paths:
//!   - [`gated_delta_kernel`] — the fast custom Metal kernel (`mx.fast.metal_kernel`):
//!     the whole `T`-step scan in one GPU dispatch. The default (~3x faster
//!     prefill). Requires `Dk % 32 == 0`.
//!   - [`gated_delta_ops`] — the pure-ops reference (sequential per-token scan).
//!     Numerically identical; the validated oracle and the `ROZUM_GD_OPS=1`
//!     escape hatch.
//! Single-stream (batch 1, no padding) so the SSM mask is always None.

use mlx_rs::{
    error::Exception,
    nn,
    ops::{expand_dims_axes, indexing::IndexOp, repeat_axis, stack_axis, zeros},
    Array, Dtype,
};

/// `g = exp(-exp(A_log) * softplus(a + dt_bias))`, computed in f32 (matching
/// Python's `A_log.astype(float32)`). `a`: `[B, T, Hv]`, `A_log`/`dt_bias`:
/// `[Hv]` -> `g`: `[B, T, Hv]`.
pub fn compute_g(a_log: &Array, a: &Array, dt_bias: &Array) -> Result<Array, Exception> {
    let inner = nn::softplus(&a.add(dt_bias)?)?;
    let a_exp = a_log.as_dtype(Dtype::Float32)?.exp()?;
    a_exp.multiply(&inner)?.negative()?.exp()
}

/// One recurrent step.
/// `q,k`: `[B, H, Dk]`, `v`: `[B, H, Dv]`, `g`/`beta`: `[B, H]`,
/// `state`: `[B, H, Dv, Dk]` -> (`y`: `[B, H, Dv]`, new `state`).
fn delta_step(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array), Exception> {
    let decay = expand_dims_axes(g, &[2, 3])?; // [B,H,1,1]
    let state = state.multiply(&decay)?;
    let k4 = expand_dims_axes(k, &[2])?; // [B,H,1,Dk]
    let kv_mem = state.multiply(&k4)?.sum_axes(&[-1], false)?; // [B,H,Dv]
    let beta1 = expand_dims_axes(beta, &[2])?; // [B,H,1]
    let delta = v.subtract(&kv_mem)?.multiply(&beta1)?; // [B,H,Dv]
    let delta1 = expand_dims_axes(&delta, &[3])?; // [B,H,Dv,1]
    let state = state.add(&k4.multiply(&delta1)?)?; // [B,H,Dv,Dk]
    let q4 = expand_dims_axes(q, &[2])?; // [B,H,1,Dk]
    let y = state.multiply(&q4)?.sum_axes(&[-1], false)?; // [B,H,Dv]
    Ok((y, state))
}

/// Sequential delta-rule scan over the time axis.
/// `q,k`: `[B, T, Hk, Dk]`, `v`: `[B, T, Hv, Dv]`, `g`/`beta`: `[B, T, Hv]`,
/// optional `state`: `[B, Hv, Dv, Dk]` (zeros if None) ->
/// (`y`: `[B, T, Hv, Dv]`, new `state`).
pub fn gated_delta_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: Option<Array>,
) -> Result<(Array, Array), Exception> {
    let qs = q.shape();
    let (b, t, hk, dk) = (qs[0], qs[1], qs[2], qs[3]);
    let vs = v.shape();
    let (hv, dv) = (vs[2], vs[3]);
    let out_dtype = q.dtype();

    // The recurrence runs in f32 (matching Python's float state / Metal kernel);
    // bf16 accumulation across positions drifts enough to break greedy parity.
    let q = q.as_dtype(Dtype::Float32)?;
    let k = k.as_dtype(Dtype::Float32)?;
    let v = v.as_dtype(Dtype::Float32)?;
    let g = g.as_dtype(Dtype::Float32)?;
    let beta = beta.as_dtype(Dtype::Float32)?;

    let mut state = match state {
        Some(s) => s,
        None => zeros::<f32>(&[b, hv, dv, dk])?,
    };

    // Broadcast K/V grouping: repeat key heads up to value heads.
    let repeat_factor = hv / hk;
    let (q, k) = if repeat_factor > 1 {
        (
            repeat_axis::<f32>(q, repeat_factor, -2)?,
            repeat_axis::<f32>(k, repeat_factor, -2)?,
        )
    } else {
        (q, k)
    };

    if std::env::var("ROZUM_GD_DEBUG").is_ok() {
        let l2 = |x: &Array| {
            x.index((0, -1))
                .square()
                .and_then(|s| s.sum(None))
                .and_then(|s| s.sqrt())
                .map(|s| s.item::<f32>())
                .unwrap_or(f32::NAN)
        };
        eprintln!(
            "GD g_l2={:.4} beta_l2={:.4} q_rep_l2={:.4} q_shape={:?}",
            l2(&g),
            l2(&beta),
            l2(&q),
            q.shape()
        );
    }
    let mut ys: Vec<Array> = Vec::with_capacity(t as usize);
    for ti in 0..t {
        let (y, ns) = delta_step(
            &q.index((.., ti)),
            &k.index((.., ti)),
            &v.index((.., ti)),
            &g.index((.., ti)),
            &beta.index((.., ti)),
            &state,
        )?;
        state = ns;
        ys.push(y);
    }
    let y = stack_axis(&ys, 1)?.as_dtype(out_dtype)?; // [B,T,Hv,Dv]
    Ok((y, state))
}

/// The fast Metal kernel (`mx.fast.metal_kernel`), scalar gating, no mask —
/// ported verbatim from Python `mlx_lm.models.gated_delta`. One GPU dispatch
/// does the whole `T`-step scan (vs the O(T) ops path). Requires `Dk % 32 == 0`.
const GATED_DELTA_SOURCE: &str = r#"
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }

    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * static_cast<float>(g_[hv_idx]);
        kv_mem += state[i] * static_cast<float>(k_[s_idx]);
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[hv_idx]);

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
        out += state[i] * static_cast<float>(q_[s_idx]);
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }

      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
"#;

fn gated_delta_metal_kernel() -> &'static mlx_rs::fast::MetalKernel {
    static KERNEL: std::sync::OnceLock<mlx_rs::fast::MetalKernel> = std::sync::OnceLock::new();
    KERNEL.get_or_init(|| {
        mlx_rs::fast::MetalKernel::new(
            "gated_delta_step",
            &["q", "k", "v", "g", "beta", "state_in", "T"],
            &["y", "state_out"],
            GATED_DELTA_SOURCE,
        )
        .expect("build gated_delta kernel")
    })
}

/// Metal-kernel delta-rule scan (the fast path). `q,k`: `[B,T,Hk,Dk]` (NOT
/// head-repeated — the kernel maps Hk->Hv internally), `v`: `[B,T,Hv,Dv]`,
/// `g`/`beta`: `[B,T,Hv]`, `state`: `[B,Hv,Dv,Dk]` (zeros if None) ->
/// (`y`: `[B,T,Hv,Dv]`, new `state`).
pub fn gated_delta_kernel(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: Option<Array>,
) -> Result<(Array, Array), Exception> {
    let qs = q.shape();
    let (b, t, hk, dk) = (qs[0], qs[1], qs[2], qs[3]);
    let vs = v.shape();
    let (hv, dv) = (vs[2], vs[3]);
    let state = match state {
        Some(s) => s,
        None => zeros::<f32>(&[b, hv, dv, dk])?,
    };
    let t_scalar = Array::from_int(t);
    let template = [
        mlx_rs::fast::TemplateArg::Dtype("InT", q.dtype()),
        mlx_rs::fast::TemplateArg::Dtype("StT", state.dtype()),
        mlx_rs::fast::TemplateArg::Int("Dk", dk),
        mlx_rs::fast::TemplateArg::Int("Dv", dv),
        mlx_rs::fast::TemplateArg::Int("Hk", hk),
        mlx_rs::fast::TemplateArg::Int("Hv", hv),
    ];
    let y_shape = [b, t, hv, dv];
    let st_shape = [b, hv, dv, dk];
    let outs = gated_delta_metal_kernel().apply(
        &[q, k, v, g, beta, &state, &t_scalar],
        &template,
        (32, dv, b * hv),
        (32, 4, 1),
        &[&y_shape[..], &st_shape[..]],
        &[q.dtype(), state.dtype()],
        mlx_rs::StreamOrDevice::default(),
    )?;
    // A blocking per-call GPU sync is REQUIRED for correctness when this kernel runs
    // inside the full Qwen3.6 forward; without it the run produces garbage from the
    // 2nd token. Root-cause dig (2026-06-12) — it is NOT what the old comment claimed
    // (state_out buffer donation):
    //   * MLX's allocator never hands out a held state buffer (instrumented
    //     MetalAllocator: 0 state-sized buffers reused while live);
    //   * disabling buffer reuse-from-cache (MLX patch) does NOT fix it;
    //   * forcing one giant command buffer (MLX_MAX_OPS_PER_BUFFER) does NOT fix it;
    //   * evaling ANY live per-layer intermediate fixes it — the kernel's projection
    //     INPUTS (q/k/v or g/beta) suffice; evaling the already-concrete cache state
    //     does not. So a per-layer GPU sync is what matters, not which array.
    //   * a faithful standalone repro (kernel + caches + 4-bit quant + real dims, no
    //     eval) stays byte-exact — it only manifests in the full 27B/35B model.
    // => an MLX metal-backend execution-ordering / buffer-lifetime issue at graph
    // scale (the kernel reads not-yet-ready input values), papered over by the sync.
    // Cost: ~48 syncs/token = the ~12-vs-17 t/s decode gap. `ROZUM_GD_NONE=1` skips
    // it (fast but corrupts — A/B only).
    if std::env::var_os("ROZUM_GD_NONE").is_none() {
        mlx_rs::transforms::eval([&outs[0], &outs[1]])?;
    }
    Ok((outs[0].clone(), outs[1].clone()))
}

/// Full update used by the linear-attention layer: `beta = sigmoid(b)`,
/// `g = compute_g(A_log, a, dt_bias)`, then the delta-rule scan. Uses the fast
/// Metal kernel by default; `ROZUM_GD_OPS=1` forces the ops reference path.
/// `q,k`: `[B,T,Hk,Dk]`, `v`: `[B,T,Hv,Dv]`, `a,b`: `[B,T,Hv]`,
/// `A_log`/`dt_bias`: `[Hv]`.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_update(
    q: &Array,
    k: &Array,
    v: &Array,
    a: &Array,
    b: &Array,
    a_log: &Array,
    dt_bias: &Array,
    state: Option<Array>,
) -> Result<(Array, Array), Exception> {
    let beta = mlx_rs::nn::sigmoid(b)?;
    let g = compute_g(a_log, a, dt_bias)?;
    if std::env::var("ROZUM_GD_OPS").is_ok() {
        gated_delta_ops(q, k, v, &g, &beta, state)
    } else {
        gated_delta_kernel(q, k, v, &g, &beta, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;

    // Raw kernel call with EXPLICIT eval control (bypasses the env gate), so one
    // test can A/B per-call-eval vs eval-at-end.
    fn run_kernel_eval(
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        state: Option<Array>,
        do_eval: bool,
    ) -> (Array, Array) {
        let qs = q.shape();
        let (b, t, hk, dk) = (qs[0], qs[1], qs[2], qs[3]);
        let vs = v.shape();
        let (hv, dv) = (vs[2], vs[3]);
        let state = state.unwrap_or_else(|| zeros::<f32>(&[b, hv, dv, dk]).unwrap());
        let t_scalar = Array::from_int(t);
        let template = [
            mlx_rs::fast::TemplateArg::Dtype("InT", q.dtype()),
            mlx_rs::fast::TemplateArg::Dtype("StT", state.dtype()),
            mlx_rs::fast::TemplateArg::Int("Dk", dk),
            mlx_rs::fast::TemplateArg::Int("Dv", dv),
            mlx_rs::fast::TemplateArg::Int("Hk", hk),
            mlx_rs::fast::TemplateArg::Int("Hv", hv),
        ];
        let y_shape = [b, t, hv, dv];
        let st_shape = [b, hv, dv, dk];
        let outs = gated_delta_metal_kernel()
            .apply(
                &[q, k, v, g, beta, &state, &t_scalar],
                &template,
                (32, dv, b * hv),
                (32, 4, 1),
                &[&y_shape[..], &st_shape[..]],
                &[q.dtype(), state.dtype()],
                mlx_rs::StreamOrDevice::default(),
            )
            .unwrap();
        if do_eval {
            eval([&outs[0], &outs[1]]).unwrap();
        }
        (outs[0].clone(), outs[1].clone())
    }

    // Attempt to reproduce the GatedDeltaNet `state_out` buffer-donation hazard in
    // ISOLATION — no model, no 27B. A decode-like forward of N "layers": each layer
    // derives q/k/v/g/beta from a running `hidden`, runs the real kernel, stores the
    // new state in a `cache`, adds the kernel output into `hidden` (residual). A
    // SECOND token reads the cached states. A/Bs per-call-eval vs eval-at-end.
    //
    // RESULT (2026-06-12): does NOT reproduce (max|Δ|=0) even with the conv cache, a
    // growing ConcatKV-style cache (every 4th layer), an MLP, 4-bit AFQ quantized
    // matmuls, real dims (d=5120, 64 layers, Hv=48/Dk=Dv=128) and 8 tokens. So the
    // token-2 garbage in the real 27B is NOT caused by the mlx-rs binding, the
    // kernel, the caches, quantization, or graph depth in isolation. The leading
    // remaining difference is real memory pressure (the model holds ~15 GB of
    // DISTINCT per-layer weights; this test shares weights so MLX always mallocs
    // fresh and never recycles a held buffer). Next step to root-cause: instrument
    // MLX core's MetalAllocator on the real model, or run this under real pressure.
    //   cargo test -p mlx-lm --release gated_delta_donation_repro -- --nocapture
    #[test]
    fn gated_delta_donation_repro() {
        use mlx_rs::ops::matmul;
        let (b, hk, hv, dk, dv) = (1, 16, 48, 128, 128);
        let d = 5120i32;
        let n_layers = 64usize;
        let n_tokens = 8usize;
        let bf = mlx_rs::Dtype::Bfloat16;
        let f32t = mlx_rs::Dtype::Float32;

        // Deterministic small weights (no RNG): keep activations bounded over depth.
        let det = |rows: i32, cols: i32, seed: f32| -> Array {
            let n = (rows * cols) as usize;
            let v: Vec<f32> = (0..n)
                .map(|i| ((i as f32) * 0.0007 + seed).sin() * 0.02)
                .collect();
            Array::from_slice(&v, &[rows, cols]).as_dtype(bf).unwrap()
        };
        // 4-bit AFQ quantized weights ([out,in]; quantized_matmul transpose=true) —
        // the real model is quantized, the last big structural difference. The g/beta
        // gates stay plain (hv=48 isn't a multiple of 32, so not quantizable).
        let inter = 17408i32;
        let mkq = |out: i32, inn: i32, seed: f32| -> (Array, Array, Array) {
            mlx_rs::ops::quantize(&det(out, inn, seed), 64, 4).unwrap()
        };
        let wq = mkq(hk * dk, d, 0.1);
        let wk = mkq(hk * dk, d, 0.2);
        let wv = mkq(hv * dv, d, 0.3);
        let wg = det(d, hv, 0.4);
        let wb = det(d, hv, 0.5);
        let wo = mkq(d, hv * dv, 0.6);
        let w_up = mkq(inter, d, 0.7);
        let w_down = mkq(d, inter, 0.8);
        let w_attn_qkv = mkq(hk * dk, d, 0.9);
        let w_attn_o = mkq(d, hk * dk, 1.1);
        let qmm = |x: &Array, w: &(Array, Array, Array)| -> Array {
            mlx_rs::ops::quantized_matmul(x, &w.0, &w.1, Some(&w.2), true, 64, 4).unwrap()
        };

        // cache: (recurrent state, conv state, kv-k, kv-v) per layer.
        type Cache = Vec<(Option<Array>, Option<Array>, Option<Array>, Option<Array>)>;
        let conv_k = 4i32;
        let forward = |hidden0: &Array, cache: &mut Cache, do_eval: bool| -> Array {
            let mut hidden = hidden0.clone(); // [1,1,d] bf16
            for l in 0..n_layers {
                let hflat = hidden.reshape(&[b, d]).unwrap();
                if l % 4 == 3 {
                    // "full attention": GROWING KV concat (the realloc churn).
                    let qn = qmm(&hflat, &w_attn_qkv).reshape(&[b, 1, hk * dk]).unwrap();
                    let kc = match cache[l].2.take() {
                        Some(prev) => mlx_rs::ops::concatenate_axis(&[&prev, &qn], 1).unwrap(),
                        None => qn.clone(),
                    };
                    cache[l].2 = Some(kc.clone());
                    // a cheap "attention": mean over the growing cache, project back.
                    let ctx = kc.mean_axes(&[1], false).unwrap(); // [b, hk*dk]
                    let delta = qmm(&ctx, &w_attn_o).reshape(&[b, 1, d]).unwrap();
                    hidden = hidden.add(&delta).unwrap();
                } else {
                    // linear (GatedDeltaNet): conv cache (small concat) + the kernel.
                    let qkv = qmm(&hflat, &wq).reshape(&[b, 1, hk * dk]).unwrap();
                    let prev = match cache[l].1.take() {
                        Some(s) => s,
                        None => zeros::<f32>(&[b, conv_k - 1, hk * dk]).unwrap().as_dtype(bf).unwrap(),
                    };
                    let conv_in = mlx_rs::ops::concatenate_axis(&[&prev, &qkv], 1).unwrap();
                    let tot = conv_in.shape()[1];
                    cache[l].1 = Some(conv_in.index((.., (tot - (conv_k - 1))..tot, ..)));
                    // L6: fast::rms_norm (weightless) on q,k like the real model.
                    let rmsn = |x: &Array| -> Array {
                        let dl = *x.shape().last().unwrap();
                        let w = Array::ones::<f32>(&[dl]).unwrap().as_dtype(x.dtype()).unwrap();
                        mlx_rs::fast::rms_norm(x, &w, 1e-6).unwrap()
                    };
                    let q = rmsn(&qkv.reshape(&[b, 1, hk, dk]).unwrap());
                    let k = rmsn(&qmm(&hflat, &wk).reshape(&[b, 1, hk, dk]).unwrap());
                    let v = qmm(&hflat, &wv).reshape(&[b, 1, hv, dv]).unwrap();
                    let g = mlx_rs::nn::sigmoid(&matmul(&hflat, &wg).unwrap())
                        .unwrap().as_dtype(f32t).unwrap().reshape(&[b, 1, hv]).unwrap();
                    let beta = mlx_rs::nn::sigmoid(&matmul(&hflat, &wb).unwrap())
                        .unwrap().reshape(&[b, 1, hv]).unwrap();
                    let (y, ns) = run_kernel_eval(&q, &k, &v, &g, &beta, cache[l].0.take(), do_eval);
                    cache[l].0 = Some(ns);
                    let yflat = y.reshape(&[b, hv * dv]).unwrap();
                    let delta = qmm(&yflat, &wo).reshape(&[b, 1, d]).unwrap();
                    hidden = hidden.add(&delta).unwrap();
                }
                // MLP (large intermediate every layer).
                let hf = hidden.reshape(&[b, d]).unwrap();
                let up = mlx_rs::nn::silu(&qmm(&hf, &w_up)).unwrap();
                let down = qmm(&up, &w_down).reshape(&[b, 1, d]).unwrap();
                hidden = hidden.add(&down).unwrap();
            }
            hidden
        };

        let run = |do_eval: bool| -> Vec<Array> {
            let mut cache: Cache = (0..n_layers).map(|_| (None, None, None, None)).collect();
            let mut hidden = det(b, d, 1.0).reshape(&[b, 1, d]).unwrap();
            let mut outs = Vec::new();
            for _ in 0..n_tokens {
                hidden = forward(&hidden, &mut cache, do_eval);
                eval([&hidden]).unwrap();
                outs.push(hidden.reshape(&[-1]).unwrap().as_dtype(f32t).unwrap());
            }
            outs
        };

        let refs = run(true); // per-call eval = correct reference
        let cand = run(false); // eval at end = candidate (no per-call eval)

        let mut max_any = 0f32;
        for t in 0..n_tokens {
            eval([&refs[t], &cand[t]]).unwrap();
            let a = refs[t].as_slice::<f32>();
            let c = cand[t].as_slice::<f32>();
            let md = a
                .iter()
                .zip(c.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            max_any = max_any.max(md);
            eprintln!("REPRO token {t}: max|Δ| = {md:.5}");
        }
        eprintln!("REPRO overall max|Δ| = {max_any:.5}  ({})", if max_any > 1e-2 { "REPRODUCED (no-eval diverges)" } else { "no divergence" });
    }

    // Reference values from Python mlx_lm.models.gated_delta.gated_delta_ops
    // (seed 0; B=1,T=3,Hk=1,Hv=2,Dk=4,Dv=4; scalar gating).
    #[test]
    fn gated_delta_matches_python() {
        let q = Array::from_slice(
            &[
                -0.144135, 0.895078, -0.460381, 0.105034, -0.746088, 0.896811, 0.387942, 0.117507,
                1.065939, 0.226569, -2.548788, 1.970043,
            ],
            &[1, 3, 1, 4],
        );
        let k = Array::from_slice(
            &[
                0.367433, -0.794102, -1.320080, 1.225790, 0.240634, 1.170095, -1.265586, 1.147047,
                -1.338933, 0.090507, -0.597340, -0.054466,
            ],
            &[1, 3, 1, 4],
        );
        let v = Array::from_slice(
            &[
                1.385772, 1.859017, -0.075232, 1.567493, 1.493831, -1.831416, -0.911714, -0.028620,
                -1.014434, -0.113476, 1.469374, -1.892237, 0.160349, -0.207623, -0.691551,
                1.089128, 0.202670, 0.253628, 0.392743, -0.495397, 0.468208, 1.596903, -0.751850,
                -0.298933,
            ],
            &[1, 3, 2, 4],
        );
        let g = Array::from_slice(
            &[0.948556, 0.940333, 0.519957, 0.685390, 0.626268, 0.939238],
            &[1, 3, 2],
        );
        let beta = Array::from_slice(
            &[0.561233, 0.449834, 0.808653, 0.165720, 0.884957, 0.066074],
            &[1, 3, 2],
        );
        let expected_y = [
            -0.021197, -0.028435, 0.001151, -0.023976, -0.018314, 0.022453, 0.011177, 0.000351,
            -1.344603, -1.285684, 0.660427, -1.830268, -0.697785, 0.854534, 0.375337, 0.106336,
            -4.314802, -1.995941, 4.537259, -7.150444, 1.778998, -2.190130, -1.641091, 0.986917,
        ];

        let (y, _state) = gated_delta_ops(&q, &k, &v, &g, &beta, None).unwrap();
        let y = y.reshape(&[-1]).unwrap();
        eval([&y]).unwrap();
        let got = y.as_slice::<f32>();
        for (i, (a, b)) in got.iter().zip(expected_y.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "y[{i}] = {a} != {b} (python ref)");
        }
    }

    // The fast Metal kernel must match the ops reference (Dk must be %32; uses
    // Dk=Dv=32, Hk=1, Hv=2, T=4). Deterministic synthetic inputs.
    #[test]
    fn gated_delta_kernel_matches_ops() {
        // Real Qwen3.6 linear dims (Hk=16, Hv=48, rf=3) to exercise hk mapping.
        let (b, t, hk, hv, dk, dv) = (1, 4, 16, 48, 128, 128);
        let gen = |n: usize, scale: f32, off: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32) * 0.137 + off).sin() * scale)
                .collect()
        };
        let q = Array::from_slice(&gen((t * hk * dk) as usize, 0.5, 0.0), &[b, t, hk, dk]);
        let k = Array::from_slice(&gen((t * hk * dk) as usize, 0.5, 1.0), &[b, t, hk, dk]);
        let v = Array::from_slice(&gen((t * hv * dv) as usize, 1.0, 2.0), &[b, t, hv, dv]);
        // g in (0,1), beta in (0,1)
        let gpos = |n: usize, off: f32| -> Vec<f32> {
            (0..n)
                .map(|i| 0.5 + 0.49 * ((i as f32) * 0.7 + off).sin())
                .collect()
        };
        let g = Array::from_slice(&gpos((t * hv) as usize, 0.0), &[b, t, hv]);
        let beta = Array::from_slice(&gpos((t * hv) as usize, 3.0), &[b, t, hv]);

        // Mimic the model dtypes: q/k/v/beta bf16, g/state f32.
        let bf = mlx_rs::Dtype::Bfloat16;
        let q = q.as_dtype(bf).unwrap();
        let k = k.as_dtype(bf).unwrap();
        let v = v.as_dtype(bf).unwrap();
        let beta = beta.as_dtype(bf).unwrap();

        let (y_ops, st_ops) = gated_delta_ops(&q, &k, &v, &g, &beta, None).unwrap();
        let (y_ker, st_ker) = gated_delta_kernel(&q, &k, &v, &g, &beta, None).unwrap();
        // The recurrent state (reused at decode) must match too.
        {
            let so = st_ops.reshape(&[-1]).unwrap();
            let sk = st_ker.reshape(&[-1]).unwrap();
            eval([&so, &sk]).unwrap();
            let (a, bb) = (so.as_slice::<f32>(), sk.as_slice::<f32>());
            assert_eq!(a.len(), bb.len(), "state length mismatch");
            for (i, (x, y)) in a.iter().zip(bb.iter()).enumerate() {
                assert!(
                    (x - y).abs() < 5e-2,
                    "kernel/ops STATE mismatch at {i}: ops={x} kernel={y}"
                );
            }
        }
        let f32 = mlx_rs::Dtype::Float32;
        let yo = y_ops.reshape(&[-1]).unwrap().as_dtype(f32).unwrap();
        let yk = y_ker.reshape(&[-1]).unwrap().as_dtype(f32).unwrap();
        eval([&yo, &yk]).unwrap();
        let (a, bb) = (yo.as_slice::<f32>(), yk.as_slice::<f32>());
        assert_eq!(a.len(), bb.len());
        for (i, (x, y)) in a.iter().zip(bb.iter()).enumerate() {
            assert!(
                (x - y).abs() < 5e-2,
                "kernel/ops mismatch at {i}: ops={x} kernel={y}"
            );
        }
    }
}

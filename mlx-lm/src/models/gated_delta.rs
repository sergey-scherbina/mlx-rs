//! GatedDeltaNet recurrence (ops path) for Qwen3-Next / Qwen3.6 linear-attention
//! layers. This is the pure-ops reference from Python `mlx_lm.models.gated_delta`
//! (`gated_delta_ops`): a sequential per-token delta-rule scan. mlx-rs has no
//! custom-kernel support, so the fast Metal kernel is not portable yet; the ops
//! path is numerically identical and correct (fast for decode T=1, O(T) for
//! prefill). Single-stream (batch 1, no padding) so the SSM mask is always None.

use mlx_rs::{
    error::Exception,
    nn,
    ops::{expand_dims_axes, indexing::IndexOp, repeat_axis, stack_axis, zeros},
    Array,
};

/// `g = exp(-exp(A_log) * softplus(a + dt_bias))`, computed in f32.
/// `a`: `[B, T, Hv]`, `A_log`/`dt_bias`: `[Hv]` -> `g`: `[B, T, Hv]`.
pub fn compute_g(a_log: &Array, a: &Array, dt_bias: &Array) -> Result<Array, Exception> {
    let inner = nn::softplus(&a.add(dt_bias)?)?;
    let a_exp = a_log.exp()?;
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

    let mut state = match state {
        Some(s) => s,
        None => zeros::<f32>(&[b, hv, dv, dk])?,
    };

    // Broadcast K/V grouping: repeat key heads up to value heads.
    let repeat_factor = hv / hk;
    let (q, k) = if repeat_factor > 1 {
        (
            repeat_axis::<f32>(q.clone(), repeat_factor, -2)?,
            repeat_axis::<f32>(k.clone(), repeat_factor, -2)?,
        )
    } else {
        (q.clone(), k.clone())
    };

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
    let y = stack_axis(&ys, 1)?; // [B,T,Hv,Dv]
    Ok((y, state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;

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
}

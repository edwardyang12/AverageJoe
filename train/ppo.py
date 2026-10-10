"""PPO algorithm: GAE, clipped surrogate update, and training loop."""

import os
import time
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

from generals.core.env import GeneralsEnv
from train.rewards import win_lose_reward
from train.rollout_selfplay import collect_rollout as collect_rollout_self
from train.evaluations import periodic_eval, EvalCtx
from evals.ref_eval import load_refs


# ---- GAE ----


@jax.jit
def compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda):
    _, N = rewards.shape

    def scan_fn(last_adv, inputs):
        reward, value, next_value, terminated, truncated = inputs
        done = (terminated | truncated).astype(jnp.float32)

        bootstrap = jnp.where(terminated, 0.0, next_value)
        delta = reward + gamma * bootstrap - value

        nonterminal = 1.0 - done
        adv = delta + gamma * gae_lambda * nonterminal * last_adv
        # Zero carry on truncation: delta is wrong (bootstraps from reset state)
        # and must not leak backwards through GAE chain
        carry = jnp.where(truncated, 0.0, adv)
        return carry, adv

    inputs = (rewards[::-1], values[::-1], next_values[::-1], terminated[::-1], truncated[::-1])
    _, advs_rev = jax.lax.scan(scan_fn, jnp.zeros(N), inputs)
    return advs_rev[::-1]


def compute_mc_returns(rewards, terminated, truncated, gamma):
    """True discounted MC returns (no bootstrapping).

    Also returns a validity mask: 1.0 for timesteps whose episode
    completed (terminated or truncated) within the rollout, 0.0 for
    timesteps in episodes still running at the rollout boundary.
    """
    _, N = rewards.shape

    def scan_fn(carry, inputs):
        next_ret, next_valid = carry
        rew, term, trunc = inputs
        done = (term | trunc).astype(jnp.float32)
        # On done: return is just this reward (episode ended). Mark valid.
        # Otherwise: accumulate and inherit validity from the future.
        ret = rew + gamma * next_ret * (1.0 - done)
        valid = jnp.where(done, 1.0, next_valid)
        return (ret, valid), (ret, valid)

    init = (jnp.zeros(N), jnp.zeros(N))
    _, (mc_rets, valid) = jax.lax.scan(
        scan_fn, init, (rewards[::-1], terminated[::-1], truncated[::-1])
    )
    return mc_rets[::-1], valid[::-1]


# ---- Value loss ----


def make_value_loss_fn(cfg):
    """Return a jittable value loss function based on config.

    MSE: val_aux is a scalar, returns 0.5 * (val - ret)^2
    CE:  val_aux is (num_bins,) logits, returns HL-Gauss cross-entropy
    """
    if cfg.value_loss not in ("mse", "ce"):
        raise ValueError(f"Unknown value_loss '{cfg.value_loss}'. Must be 'mse' or 'ce'.")
    if cfg.value_loss == "ce":
        bin_centers = jnp.linspace(cfg.v_min, cfg.v_max, cfg.num_bins)
        half_width = (cfg.v_max - cfg.v_min) / (cfg.num_bins - 1) / 2.0
        sigma = cfg.hl_sigma

        def value_loss_fn(logits, ret):
            upper = (bin_centers + half_width - ret) / sigma
            lower = (bin_centers - half_width - ret) / sigma
            target_probs = jax.scipy.stats.norm.cdf(upper) - jax.scipy.stats.norm.cdf(lower)
            target_probs = target_probs / jnp.maximum(jnp.sum(target_probs), 1e-8)
            log_probs = jax.nn.log_softmax(logits)
            return -jnp.sum(target_probs * log_probs)

        return value_loss_fn
    else:
        def value_loss_fn(val, ret):
            return 0.5 * (val - ret) ** 2

        return value_loss_fn


# ---- PPO ----


def ppo_update(network, opt_state, batch, optimizer, key, clip_eps, vf_coef, ent_coef, minibatch_size, value_loss_fn, sample_idx, magnet_fn=None):
    """One PPO epoch: shuffle sample indices, then lax.scan gathering minibatches on-the-fly."""
    obs, masks, temporal, actions, old_lps, advs, rets, train_mask = batch
    total = obs.shape[0] * obs.shape[1]

    # Flatten time and env dims (views — no allocation)
    obs_f = obs.reshape(total, *obs.shape[2:])
    masks_f = masks.reshape(total, *masks.shape[2:])
    temporal_f = temporal.reshape(total, *temporal.shape[2:])
    actions_f = actions.reshape(total, -1)
    old_lps_f = old_lps.reshape(-1)
    advs_f = advs.reshape(-1)
    rets_f = rets.reshape(-1)
    mask_f = train_mask.reshape(-1)

    # Shuffle indices only (not data — avoids creating full-size copies)
    n_samples = sample_idx.shape[0]
    num_batches = n_samples // minibatch_size
    perm = jrandom.permutation(key, n_samples)
    shuffled_idx = sample_idx[perm]
    idx_mb = shuffled_idx.reshape(num_batches, minibatch_size)

    def scan_body(carry, mb_idx):
        network, opt_state = carry
        mb_obs = obs_f[mb_idx]
        mb_masks = masks_f[mb_idx]
        mb_temporal = temporal_f[mb_idx]
        mb_actions = actions_f[mb_idx]
        mb_old_lps = old_lps_f[mb_idx]
        mb_advs = advs_f[mb_idx]
        mb_rets = rets_f[mb_idx]
        mb_mask = mask_f[mb_idx]

        def loss_fn(net):
            def single_loss(o, m, td, a, old_lp, adv, ret):
                _, val, lp, ent, val_aux, p_dist = net(o, m, td, None, a)
                log_ratio = lp - old_lp
                ratio = jnp.exp(log_ratio)
                # Standard PPO clipped objective
                pg1 = -adv * ratio
                pg2 = -adv * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                policy_loss = jnp.maximum(pg1, pg2)
                value_loss = value_loss_fn(val_aux, ret)
                clipped = (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32)
                approx_kl = ratio - 1.0 - log_ratio
                # Reverse KL toward magnet: KL(policy || magnet) = -H(p) - sum(p * log(m))
                magnet_dist = magnet_fn(o, m)
                magnet_kl = -ent - jnp.sum(p_dist * jnp.log(magnet_dist + 1e-10))
                total = policy_loss + vf_coef * value_loss + ent_coef * magnet_kl
                return total, {
                    "policy_loss": policy_loss, "value_loss": value_loss, "entropy": ent,
                    "clip_fraction": clipped, "approx_kl": approx_kl, "ratio": ratio,
                    "log_ratio": log_ratio, "lp": lp, "old_lp": old_lp, "magnet_kl": magnet_kl,
                }

            all_losses, s = jax.vmap(single_loss)(
                mb_obs, mb_masks, mb_temporal, mb_actions, mb_old_lps, mb_advs, mb_rets,
            )
            # Mask out truncated steps (their delta/advantage is wrong)
            masked_losses = all_losses * mb_mask
            mean_loss = masked_losses.sum() / jnp.maximum(mb_mask.sum(), 1.0)
            # Per-minibatch summary; key name carries the cross-minibatch
            # reduction (max_*/min_* -> max/min, else mean).
            stats = {
                "total_loss": mean_loss,
                "policy_loss": s["policy_loss"].mean(),
                "value_loss": s["value_loss"].mean(),
                "entropy": s["entropy"].mean(),
                "clip_fraction": s["clip_fraction"].mean(),
                "approx_kl": s["approx_kl"].mean(),
                "mean_ratio": s["ratio"].mean(),
                "magnet_kl": s["magnet_kl"].mean(),
                "max_kl": s["approx_kl"].max(),
                "max_ratio": s["ratio"].max(),
                "min_log_ratio": s["log_ratio"].min(),
                "max_log_ratio": s["log_ratio"].max(),
                "min_lp": s["lp"].min(),
                "max_lp": s["lp"].max(),
                "min_old_lp": s["old_lp"].min(),
                "max_old_lp": s["old_lp"].max(),
            }
            return mean_loss, stats

        (loss, stats), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
        grads = jax.lax.pmean(grads, axis_name='devices')

        # Gradient norms (total, actor, critic)
        grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
        stats["grad_norm"] = jnp.sqrt(sum(jnp.sum(g**2) for g in grad_leaves))
        policy_grads = jax.tree.leaves(eqx.filter(grads.policy_head, eqx.is_array))
        stats["actor_grad_norm"] = jnp.sqrt(sum(jnp.sum(g**2) for g in policy_grads))
        value_grads = jax.tree.leaves(eqx.filter(grads.value_head, eqx.is_array))
        stats["critic_grad_norm"] = jnp.sqrt(sum(jnp.sum(g**2) for g in value_grads))

        updates, opt_state = optimizer.update(grads, opt_state, network)
        network = eqx.apply_updates(network, updates)
        return (network, opt_state), stats

    (network, opt_state), stacked = jax.lax.scan(scan_body, (network, opt_state), idx_mb)

    # Reduce across minibatches; key name carries the reduction.
    def _reduce(k, v):
        if k.startswith("max_"):
            return jnp.max(v)
        if k.startswith("min_"):
            return jnp.min(v)
        return jnp.mean(v)
    result = {k: _reduce(k, v) for k, v in stacked.items()}
    return network, opt_state, result


# ---- Training Loop ----


def train(env, pool, network, optimizer, opt_state, logger, key, cfg, bundle, ckpt_dir, run_name):
    """Main PPO training loop with multi-GPU data parallelism via pmap."""
    num_envs = cfg.num_envs
    num_devices = jax.device_count()
    grid_size = cfg.pad_to
    current_gamma = cfg.gamma
    gamma_anneal = cfg.gamma_anneal_iters > 0 and cfg.gamma_end != cfg.gamma
    iter_offset = getattr(cfg, 'iteration_offset', 0)
    reward_fn = win_lose_reward

    init_obs_state_fn = bundle["init_obs_state"]
    augment_fn = bundle["augment_obs"]
    reset_fn = bundle["reset_obs_state"]
    greedy_fn = bundle["greedy_action"]
    value_loss_fn = make_value_loss_fn(cfg)

    # Partition network for pmap: array leaves are replicated, static captured in closures
    params, static = eqx.partition(network, eqx.is_array)
    params = jax.device_put_replicated(params, jax.devices())
    opt_state = jax.device_put_replicated(opt_state, jax.devices())

    # Per-device environment init
    def _init_envs(key):
        keys = jrandom.split(key, num_envs)
        return jax.vmap(env.init_state)(keys)
    p_init_envs = jax.pmap(_init_envs)

    key, init_key = jrandom.split(key)
    states = p_init_envs(jrandom.split(init_key, num_devices))  # (D, num_envs, ...)

    # Initialize per-network observation state (batched across envs, replicated across devices)
    single_state = init_obs_state_fn(grid_size, cfg.pad_to)
    batched_obs_state = jax.tree.map(lambda x: jnp.tile(x, (num_envs, *([1] * x.ndim))), single_state)
    obs_state_p0 = jax.device_put_replicated(batched_obs_state, jax.devices())
    obs_state_p1 = jax.device_put_replicated(batched_obs_state, jax.devices())

    # Per-device PRNG keys
    keys = jrandom.split(key, num_devices)

    # ---- pmap wrappers (closures over static args) ----

    # Replicate pool across devices for rollout
    pool_rep = jax.device_put_replicated(pool, jax.devices())

    def _rollout_self(params, states, key, osp0, osp1, pool_r, gamma):
        network = eqx.combine(params, static)
        return collect_rollout_self(
            states, env, network, key, cfg.num_steps, osp0, osp1,
            grid_size, reward_fn, augment_fn, reset_fn, gamma, pool=pool_r)
    p_rollout_self = jax.pmap(_rollout_self)

    def _compute_gae(rews, vals, next_vals, terminated, truncated, gamma):
        return compute_gae(rews, vals, next_vals, terminated, truncated, gamma, cfg.gae_lambda)
    p_gae = jax.pmap(_compute_gae)
    p_mc_returns = jax.pmap(compute_mc_returns)

    @partial(jax.pmap, axis_name='devices')
    def _normalize_advs(advs):
        mean = jax.lax.pmean(advs.mean(), axis_name='devices')
        mean_sq = jax.lax.pmean((advs ** 2).mean(), axis_name='devices')
        std = jnp.sqrt(jnp.maximum(mean_sq - mean ** 2, 0.0))
        return (advs - mean) / (std + 1e-8)

    from train.magnet import get_magnet_fn
    magnet_fn = get_magnet_fn(cfg.magnet_policy)

    def _ppo_step(params, opt_state, batch, key, ent_coef, sample_idx):
        network = eqx.combine(params, static)
        network, opt_state, result = ppo_update(
            network, opt_state, batch, optimizer, key,
            cfg.clip_eps, cfg.vf_coef, ent_coef, cfg.minibatch_size, value_loss_fn, sample_idx, magnet_fn=magnet_fn)
        new_params, _ = eqx.partition(network, eqx.is_array)
        result = jax.lax.pmean(result, axis_name='devices')
        return new_params, opt_state, result
    p_ppo_step = jax.pmap(_ppo_step, axis_name='devices')

    p_split_key = jax.pmap(lambda k: jrandom.split(k))  # returns (D, 2, ...)

    def _get_network():
        """Extract single-device network from replicated params (device 0)."""
        return eqx.combine(jax.tree.map(lambda x: x[0], params), static)

    def _get_opt_state():
        """Extract single-device optimizer state from replicated (device 0)."""
        return jax.tree.map(lambda x: x[0], opt_state)

    # Sample filtering: top-k by |advantage|
    per_device_total = cfg.num_steps * 2 * num_envs
    n_keep = int(per_device_total * cfg.adv_top_frac)
    n_keep = (n_keep // cfg.minibatch_size) * cfg.minibatch_size

    @jax.pmap
    def _compute_top_idx(advs):
        flat_advs = advs.reshape(-1)
        _, top_idx = jax.lax.top_k(jnp.abs(flat_advs), n_keep)
        return top_idx

    # Multi-stage curriculum
    curriculum_stages = cfg.curriculum_stages or []
    current_stage_idx = 0 if curriculum_stages else -1  # stage 0 already applied in main.py
    last_eval_wr = 0.0  # most recent eval win-rate vs random (for curriculum gating)

    # EMA network (stored on CPU, updated every iteration, saved alongside regular checkpoints)
    ema_decay = cfg.ema_decay
    if cfg.ema_checkpoint:
        ema_network = eqx.tree_deserialise_leaves(cfg.ema_checkpoint, _get_network())
        ema_params, _ = eqx.partition(ema_network, eqx.is_array)
        print(f"Loaded EMA weights from {cfg.ema_checkpoint}")
    else:
        ema_params = jax.tree.map(lambda x: x[0].copy(), params)  # init from device 0

    # Separate env + pool for eval
    eval_env = GeneralsEnv(
        min_grid_size=env.min_grid_size, max_grid_size=env.max_grid_size,
        pad_to=env.pad_to, min_generals_distance=env.min_generals_distance,
        max_generals_distance=env.max_generals_distance,
        truncation=env.truncation, castle_val_range=env.castle_val_range,
        num_cities_range=env.num_cities_range,
        mountain_density_range=env.mountain_density_range,
        pool_size=1000,
    )
    key, eval_env_key = jrandom.split(key)
    eval_pool, _ = eval_env.reset(eval_env_key)

    # Reference ELO eval setup
    ref_agents = None
    ref_h2h = None
    ref_eval_env = None
    ref_eval_pool = None
    if cfg.ref_eval_every > 0 and cfg.ref_eval_dir:
        import copy, json as _json
        ref_agents = load_refs(cfg.ref_eval_dir)
        matrix_path = cfg.ref_eval_matrix or os.path.join(cfg.ref_eval_dir, "h2h.json")
        if os.path.exists(matrix_path):
            ref_h2h = _json.load(open(matrix_path))["h2h"]
        else:
            print(f"WARNING: No ref matrix at {matrix_path}, ref-vs-ref results will be missing")
            ref_h2h = {}
        _ref_cfg = copy.copy(cfg)
        _ref_cfg.truncation = cfg.ref_eval_truncation or cfg.truncation
        ref_eval_env = GeneralsEnv(
            min_grid_size=cfg.ref_eval_min_grid_size or cfg.min_grid_size,
            max_grid_size=cfg.ref_eval_max_grid_size or cfg.max_grid_size,
            pad_to=cfg.pad_to,
            min_generals_distance=cfg.ref_eval_min_generals_distance,
            max_generals_distance=cfg.ref_eval_max_generals_distance,
            truncation=_ref_cfg.truncation,
            num_cities_range=(cfg.ref_eval_num_cities_min, cfg.ref_eval_num_cities_max),
            castle_val_range=(cfg.ref_eval_castle_val_min, cfg.ref_eval_castle_val_max),
        )
        key, _rpool_key = jrandom.split(key)
        ref_eval_pool, _ = ref_eval_env.reset(_rpool_key)
        print(f"REF_EVAL: {len(ref_agents)} refs, {cfg.ref_eval_games} games/side")

    ev = EvalCtx(
        bundle=bundle, single_state=single_state,
        augment_fn=augment_fn, reset_fn=reset_fn, greedy_fn=greedy_fn,
        ref_agents=ref_agents, ref_h2h=ref_h2h,
        ref_eval_env=ref_eval_env, ref_eval_pool=ref_eval_pool,
    )

    mode_str = "self-play"
    print(f"Training ({mode_str}, {num_devices} device(s))...")
    train_start = time.time()
    for it in range(cfg.num_iters):
        # Eval (before training so it==0 gives a baseline)
        network = _get_network()
        on_last_stage = curriculum_stages and current_stage_idx >= len(curriculum_stages) - 1
        eval_freq = cfg.eval_every_after if (cfg.eval_every_after and on_last_stage) else cfg.eval_every
        eval_ran, last_eval_wr, key = periodic_eval(
            it, cfg, eval_freq, network, ema_params, static,
            eval_env, eval_pool, ev, logger, key, last_eval_wr)

        t0 = time.time()

        # Curriculum: advance to next stage when win-rate threshold is met (only on eval iters)
        if eval_ran and curriculum_stages and current_stage_idx < len(curriculum_stages) - 1:
            next_stage = curriculum_stages[current_stage_idx + 1]
            if last_eval_wr >= next_stage.win_rate_threshold:
                current_stage_idx += 1
                stage = next_stage
                print(f"CURRICULUM stage {current_stage_idx}/{len(curriculum_stages)-1} (iter {it}, wr={last_eval_wr:.0%}>={stage.win_rate_threshold:.0%}): "
                      f"dist={stage.min_generals_distance}-{stage.max_generals_distance}")
                # Update env attributes — pool is traced so no rollout recompile needed
                env.min_generals_distance = stage.min_generals_distance
                env.max_generals_distance = stage.max_generals_distance
                if stage.castle_val_min is not None:
                    env.castle_val_range = (stage.castle_val_min, stage.castle_val_max)
                if stage.num_cities_min is not None:
                    env.num_cities_range = (stage.num_cities_min, stage.num_cities_max)
                # Regenerate pool with new params (pool generator recompiles, rollout does NOT)
                key, pool_key = jrandom.split(key)
                pool, _ = env.reset(pool_key)
                pool_rep = jax.device_put_replicated(pool, jax.devices())
                # New eval env (new object forces JIT retrace of evaluate())
                castle = (stage.castle_val_min, stage.castle_val_max) if stage.castle_val_min is not None else eval_env.castle_val_range
                cities = (stage.num_cities_min, stage.num_cities_max) if stage.num_cities_min is not None else eval_env.num_cities_range
                eval_env = GeneralsEnv(
                    min_grid_size=eval_env.min_grid_size, max_grid_size=eval_env.max_grid_size,
                    pad_to=eval_env.pad_to, min_generals_distance=stage.min_generals_distance,
                    max_generals_distance=stage.max_generals_distance,
                    truncation=eval_env.truncation, castle_val_range=castle,
                    num_cities_range=cities,
                    mountain_density_range=eval_env.mountain_density_range,
                    pool_size=eval_env.pool_size,
                )
                key, eval_pool_key = jrandom.split(key)
                eval_pool, _ = eval_env.reset(eval_pool_key)
                # Re-init env states from new pool
                key, reinit_key = jrandom.split(key)
                states = p_init_envs(jrandom.split(reinit_key, num_devices))
                obs_state_p0 = jax.device_put_replicated(batched_obs_state, jax.devices())
                obs_state_p1 = jax.device_put_replicated(batched_obs_state, jax.devices())
                # Update gamma if specified (traced, no recompile needed)
                if stage.gamma is not None:
                    current_gamma = stage.gamma
                print(f"CURRICULUM: regenerated pool")

        # Periodically regenerate the map pool for diversity (no recompile — pool is traced)
        if cfg.reset_pool_every > 0 and it > 0 and it % cfg.reset_pool_every == 0:
            key, pool_key = jrandom.split(key)
            pool, _ = env.reset(pool_key)
            pool_rep = jax.device_put_replicated(pool, jax.devices())

        # Gamma annealing: update current_gamma (traced through rollout + GAE, no recompile)
        if gamma_anneal:
            frac = min((it + iter_offset) / cfg.gamma_anneal_iters, 1.0)
            current_gamma = cfg.gamma + (cfg.gamma_end - cfg.gamma) * frac

        # Collect rollout — pmapped across devices
        t_rollout = time.time()
        gamma_rep = jnp.full(num_devices, current_gamma)
        states, rollout_data, keys, obs_state_p0, obs_state_p1 = p_rollout_self(
            params, states, keys, obs_state_p0, obs_state_p1, pool_rep, gamma_rep)
        jax.block_until_ready(states)
        t_rollout = time.time() - t_rollout

        # rollout_data shapes: (D, num_steps, N, ...) where N = 2*num_envs (self) or num_envs
        obs, masks, temporal, actions, lps, vals, next_vals, rews, terminated, truncated, winners, owned_cities = rollout_data
        del rollout_data
        dones = terminated | truncated

        # Episode stats from p0 perspective (summed across all devices)
        dones_p0 = dones[:, :, :num_envs]
        terminated_p0 = terminated[:, :, :num_envs]
        winners_p0 = winners[:, :, :num_envs]

        # GAE advantages (per-device) — gamma_rep already computed before rollout
        advs = p_gae(rews, vals, next_vals, terminated, truncated, gamma_rep)
        rets = advs + vals
        adv_std_raw = float(advs.std())
        advs = _normalize_advs(advs)

        # Rollout-level diagnostics (before any filtering)
        mean_owned_cities = float(owned_cities.mean()) / num_envs
        mean_val = float(vals.mean())
        ret_mean = float(rets.mean())
        ret_std = float(rets.std())
        var_returns = float(jnp.var(rets))
        explained_var = 1.0 - float(jnp.var(rets - vals)) / max(var_returns, 1e-8)
        value_bias = float(jnp.mean(vals - rets))

        # MC explained variance: V(s) vs true discounted returns (no bootstrap)
        mc_rets, mc_valid = p_mc_returns(rews, terminated, truncated, gamma_rep)
        mc_valid_count = float(mc_valid.sum())
        mc_valid_frac = mc_valid_count / max(float(mc_valid.size), 1.0)
        if mc_valid_count > 100:
            mc_diff = (vals - mc_rets) * mc_valid
            mc_mean_ret = float(jnp.sum(mc_rets * mc_valid)) / mc_valid_count
            mc_var_ret = float(jnp.sum(mc_valid * (mc_rets - mc_mean_ret) ** 2) / mc_valid_count)
            mc_ev = 1.0 - float(jnp.sum(mc_diff ** 2) / mc_valid_count) / max(mc_var_ret, 1e-8)
            mc_value_bias = float(jnp.sum(mc_diff) / mc_valid_count)
        else:
            mc_ev = float('nan')
            mc_value_bias = float('nan')

        # PPO update (minibatched, lax.scan over gradient steps)
        t_ppo = time.time()
        train_mask = 1.0 - truncated.astype(jnp.float32)

        # Compute sample indices (no data copies — just indices)
        sample_idx = _compute_top_idx(advs)
        advs_flat = advs.reshape(num_devices, -1)
        filtered_advs = jnp.take_along_axis(advs_flat, sample_idx, axis=1)
        filtered_adv_std = float(filtered_advs.std())
        filtered_adv_mean = float(jnp.abs(filtered_advs).mean())

        batch = (obs, masks, temporal, actions, lps, advs, rets, train_mask)
        metrics = {}
        epochs_used = 0

        # Entropy coefficient schedule
        sched_it = it + iter_offset
        if cfg.ent_schedule == "power_law":
            current_ent_coef = max(cfg.ent_coef_start / (sched_it + 1) ** cfg.ent_power, cfg.ent_coef_min)
        else:  # linear
            t = min(sched_it / max(cfg.ent_coef_decay_iters, 1), 1.0)
            current_ent_coef = cfg.ent_coef_start + t * (cfg.ent_coef_end - cfg.ent_coef_start)

        # Learning rate
        if getattr(cfg, 'lr_schedule', 'linear') == 'power_law':
            iteration = sched_it + 1.0
            raw = cfg.lr_power_law_numerator / (iteration ** cfg.lr_power_law_exponent)
            current_lr = max(min(raw, cfg.lr_power_law_max), cfg.lr_power_law_min)
        elif cfg.lr_decay_iters > 0:
            t_lr = min(sched_it / cfg.lr_decay_iters, 1.0)
            current_lr = cfg.lr + t_lr * (cfg.final_lr - cfg.lr)
        else:
            current_lr = cfg.lr
        ent_coef_arr = jnp.full(num_devices, current_ent_coef)

        for _ in range(cfg.num_epochs):
            split = p_split_key(keys)  # (D, 2, ...)
            keys, epoch_keys = split[:, 0], split[:, 1]
            params, opt_state, metrics = p_ppo_step(
                params, opt_state, batch, epoch_keys, ent_coef_arr, sample_idx)
            epochs_used += 1
            if cfg.target_kl is not None and float(metrics["approx_kl"][0]) > cfg.target_kl:
                break
        jax.block_until_ready(params)
        t_ppo = time.time() - t_ppo

        # Metrics from device 0 (identical across devices due to pmean)
        m = jax.tree.map(lambda x: x[0], metrics)

        elapsed = time.time() - t0
        eps = int(dones_p0.sum())
        wins = int(jnp.sum(terminated_p0 & (winners_p0 == 0)))
        losses = int(jnp.sum(terminated_p0 & (winners_p0 == 1)))
        draws = eps - wins - losses
        wr = wins / max(eps, 1)
        lr = losses / max(eps, 1)
        dr = draws / max(eps, 1)
        samples_per_iter = num_devices * 2 * num_envs * cfg.num_steps
        sps = samples_per_iter / elapsed

        total_steps_p0 = float(dones_p0.size)
        mean_ep_len = total_steps_p0 / max(eps, 1)

        wall = int(time.time() - train_start)
        hh, mm, ss = wall // 3600, wall % 3600 // 60, wall % 60
        print(
            f"[{hh:02d}:{mm:02d}:{ss:02d}] Iter {it + 1:3d}/{cfg.num_iters} | Loss: {float(m['total_loss']):.4f} | "
            f"PG: {float(m['policy_loss']):.4f} | "
            f"VF: {float(m['value_loss']):.4f} | "
            f"Ent: {float(m['entropy']):.3f} | "
            f"KL: {float(m['approx_kl']):.4f} | "
            f"Clip: {float(m['clip_fraction']):.2f} | "
            f"GNorm: {float(m['grad_norm']):.2f} | "
            f"EV: {explained_var:.2f} | "
            f"Reward: {float(rews.mean()):+.4f} | "
            f"Eps: {eps:3d} | W/L/D: {wins}/{losses}/{draws} ({wr * 100:.0f}%/{lr * 100:.0f}%/{dr * 100:.0f}%) | "
            f"EpLen: {mean_ep_len:.0f} | "
            f"Cities: {mean_owned_cities:.2f} | "
            f"Epochs: {epochs_used}/{cfg.num_epochs} | "
            f"LR: {current_lr:.1e} | "
            f"SPS: {sps:.0f} | {elapsed:.2f}s (rollout {t_rollout:.2f}s, ppo {t_ppo:.2f}s)"
        )

        log_metrics = {
            "train/total_loss": m["total_loss"],
            "train/policy_loss": m["policy_loss"],
            "train/value_loss": m["value_loss"],
            "train/entropy": m["entropy"],
            "train/magnet_kl": m["magnet_kl"],
            "train/n_actions": jnp.exp(m["entropy"]),
            "train/clip_fraction": m["clip_fraction"],
            "train/approx_kl": m["approx_kl"],
            "train/mean_ratio": m["mean_ratio"],
            "train/max_ratio": m["max_ratio"],
            "train/grad_norm": m["grad_norm"],
            "train/actor_grad_norm": m["actor_grad_norm"],
            "train/critic_grad_norm": m["critic_grad_norm"],
            "train/explained_variance": explained_var,
            "train/mc_explained_variance": mc_ev,
            "train/value_bias": value_bias,
            "train/mc_value_bias": mc_value_bias,
            "train/mc_valid_frac": mc_valid_frac,
            "train/mean_value": mean_val,
            "train/return_mean": ret_mean,
            "train/return_std": ret_std,
            "train/adv_std_raw": adv_std_raw,
            "train/filtered_adv_std": filtered_adv_std,
            "train/filtered_adv_mean_abs": filtered_adv_mean,
            "train/mean_ep_length": mean_ep_len,
            "train/mean_reward": float(rews.mean()),
            "train/win_rate": wr,
            "train/loss_rate": lr,
            "train/draw_rate": dr,
            "train/mean_owned_cities": mean_owned_cities,
            "train/sps": sps,
            "train/ent_coef": current_ent_coef,
            "train/lr": current_lr,
            "train/gamma": current_gamma,
        }
        if cfg.debug:
            log_metrics.update({
                "train/max_kl": m["max_kl"],
                "train/min_log_ratio": m["min_log_ratio"],
                "train/max_log_ratio": m["max_log_ratio"],
                "train/min_lp": m["min_lp"],
                "train/max_lp": m["max_lp"],
                "train/min_old_lp": m["min_old_lp"],
                "train/max_old_lp": m["max_old_lp"],
                "train/epochs_used": epochs_used,
            })
        logger.log(it + 1, log_metrics)

        # Update EMA params
        current_params = jax.tree.map(lambda x: x[0], params)
        ema_params = jax.tree.map(
            lambda e, c: ema_decay * e + (1 - ema_decay) * c,
            ema_params, current_params)

        # Extract single-device network for checkpointing
        network = _get_network()

        if (it + 1) % cfg.ckpt_every == 0:
            ema_ckpt_path = os.path.join(ckpt_dir, f"{run_name}_ema_{it + 1}.eqx")
            eqx.tree_serialise_leaves(ema_ckpt_path, eqx.combine(ema_params, static))

        if (it + 1) % cfg.save_every == 0:
            path = os.path.join(ckpt_dir, f"{run_name}_{it + 1}.eqx")
            eqx.tree_serialise_leaves(path, (network, _get_opt_state()))
            ema_network = eqx.combine(ema_params, static)
            ema_path = os.path.join(ckpt_dir, f"{run_name}_ema_{it + 1}.eqx")
            eqx.tree_serialise_leaves(ema_path, ema_network)
            ema_latest = os.path.join(ckpt_dir, f"{run_name}_ema.eqx")
            eqx.tree_serialise_leaves(ema_latest, ema_network)
            print(f"  SAVED: {path} + EMA: {ema_path}")

        # Free large arrays to prevent BFC allocator fragmentation on next rollout
        del obs, masks, temporal, actions, lps, advs, rets, train_mask, batch, sample_idx
        del vals, next_vals, rews, terminated, truncated, winners, owned_cities
        del dones, dones_p0, terminated_p0, winners_p0

    return _get_network(), _get_opt_state()

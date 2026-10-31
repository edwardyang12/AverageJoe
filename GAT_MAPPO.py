"""
Generals.io Cell-as-Agent MARL-GAT Trainer (PyTorch + JAX Environment Bridge)
Combines the CleanRL MAPPO structure of `ppo_vmas_navigation_radius.py` with the 
exact self-play PPO recipe, HL-Gauss critic, Top-25% Advantage Filtering, and 
truncation-safe GAE from Straka et al. (`ppo.py`).
"""

import os
# CRITICAL: Prevent JAX from pre-allocating 75% of GPU VRAM or fragmenting memory via BFC
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Enable headless Pygame video & audio drivers for screenless Docker containers
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import copy
import gc
import inspect
import math
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# Live JAX + generals-bots (`pip install git+https://github.com/strakam/generals-bots.git`) imports
try:
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import jax.dlpack as j_dlpack
    from generals.core.env import GeneralsEnv
    from generals.core.game import get_observation
    from generals.core.action import compute_valid_move_mask
    JAX_ENV_AVAILABLE = True
except ImportError:
    JAX_ENV_AVAILABLE = False

# Optional AverageJoe repo helpers (if running inside https://github.com/strakam/AverageJoe)
try:
    from networks import get_network_bundle, obs_to_array, reset_done_envs
    from networks.common import encode_action, decode_action, prepare_action_mask, normalize_observations
    import train.rewards as aj_rewards
    from train.rewards import win_lose_reward
    COMPOSITE_REWARD_FN = getattr(aj_rewards, "composite_reward", None)
    AVERAGEJOE_BUNDLE_AVAILABLE = True
except ImportError:
    COMPOSITE_REWARD_FN = None
    AVERAGEJOE_BUNDLE_AVAILABLE = False


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml",
        help="path to AverageJoe YAML config file (automatically syncs env, curriculum, PPO, and LR settings)")
    parser.add_argument("--exp-name", type=str, default="generals_marl_gat",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=43,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="generals-marl-gat",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")

    # Environment & Curriculum arguments (aligned with your YAML config defaults)
    parser.add_argument("--grid-size", type=int, default=15,
        help="padded HxW board dimension (default 15 to match your config)")
    parser.add_argument("--num-envs", type=int, default=640,
        help="number of parallel 1v1 game environments (640 envs * 2 = 1280 player views)")
    parser.add_argument("--num-steps", type=int, default=256,
        help="rollout length T per iteration (256 in config)")
    parser.add_argument("--num-iters", type=int, default=25000,
        help="total PPO self-play iterations")
    parser.add_argument("--use-synthetic-env", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, forces SyntheticGeneralsVecEnv even if generals-bots is installed")

    # GeneralsEnv Map & Spawn-Distance Curriculum arguments (synced with YAML config)
    parser.add_argument("--min-grid-size", type=int, default=12)
    parser.add_argument("--max-grid-size", type=int, default=15)
    parser.add_argument("--pad-to", type=int, default=15)
    parser.add_argument("--min-generals-distance", type=int, default=4)
    parser.add_argument("--max-generals-distance", type=int, default=9)
    parser.add_argument("--castle-val-min", type=int, default=40)
    parser.add_argument("--castle-val-max", type=int, default=51)
    parser.add_argument("--truncation", type=int, default=1024,
        help="episode step truncation limit (1024 in config)")
    parser.add_argument("--allow-pass", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to allow voluntary pass when valid moves exist")
    parser.add_argument("--pool-size", type=int, default=50000,
        help="pre-generated map pool size on GPU")
    parser.add_argument("--reset-pool-every", type=int, default=10,
        help="regenerate the JAX map pool every N iterations")
    parser.add_argument("--eval-every", type=int, default=10,
        help="evaluate greedy policy vs random bot every N iterations")
    parser.add_argument("--eval-games", type=int, default=512,
        help="number of evaluation games vs random bot")
    parser.add_argument("--elo-eval-every", type=int, default=50,
        help="run Elo league evaluation against frozen historical checkpoints every N iterations")
    parser.add_argument("--max-league-snapshots", type=int, default=5,
        help="maximum number of historical model snapshots kept in the Elo league pool")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to record headless MP4 game replays using SDL dummy driver (works in screenless Docker)")
    parser.add_argument("--video-every", type=int, default=100,
        help="record a replay MP4 video every N iterations during training")
    parser.add_argument("--video-fps", type=int, default=10,
        help="FPS for recorded MP4 replay videos")
    parser.add_argument("--eval-checkpoint", type=str, default=None,
        help="optional path to a saved .pt checkpoint to immediately record replay videos and exit")
    parser.add_argument("--eval-mode", type=str, default="selfplay", choices=["selfplay", "random"],
        help="whether recorded replay videos pit the agent against itself ('selfplay') or random ('random')")
    parser.add_argument("--num-videos", type=int, default=3,
        help="number of replay videos to record when running in --eval-checkpoint mode")

    # Architecture arguments (matching MARL-GAT Spec & config embed_dim=256, n_head=4, ff_factor=2)
    parser.add_argument("--d-model", type=int, default=256,
        help="hidden embedding dimension across GAT layers")
    parser.add_argument("--num-layers", type=int, default=6,
        help="number of 5x5 GAT layers (6 layers covers 15x15 board completely + global relay)")
    parser.add_argument("--heads", type=int, default=4,
        help="number of attention heads in GAT and Critic Cross-Attention")
    parser.add_argument("--ffn-mult", type=int, default=2,
        help="FFN expansion factor (2 matches ff_factor: 2 in config)")
    parser.add_argument("--compile", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="use torch.compile on the GAT agent for fused CUDA kernels")

    # PPO & Optimization arguments (matching your YAML config)
    parser.add_argument("--minibatch-size", type=int, default=4096,
        help="minibatch size (4096 utilizes 48GB A40 VRAM to cut PPO time in half)")
    parser.add_argument("--update-epochs", type=int, default=1,
        help="epochs per iteration (num_epochs: 1)")
    parser.add_argument("--target-kl", type=float, default=0.02,
        help="target KL divergence threshold for early epoch stopping")
    parser.add_argument("--gamma", type=float, default=0.999,
        help="discount factor gamma (0.999 in config)")
    parser.add_argument("--gae-lambda", type=float, default=0.9,
        help="lambda for Generalized Advantage Estimation")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="PPO surrogate clipping coefficient epsilon")
    parser.add_argument("--vf-coef", type=float, default=1.0,
        help="value loss coefficient beta (1.0 in config)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="maximum gradient norm clipping")
    parser.add_argument("--adv-top-frac", type=float, default=0.25,
        help="top-advantage filtering fraction (0.25 doubles PPO speed and focuses on high-leverage steps)")
    parser.add_argument("--ema-decay", type=float, default=0.999,
        help="exponential moving average decay tau for deployment policy")

    # Optimizer Warmup + Cosine Decay Schedule (matching your YAML config!)
    parser.add_argument("--lr", type=float, default=1.0e-4,
        help="peak learning rate after warmup")
    parser.add_argument("--warmup-lr", type=float, default=1.0e-5,
        help="initial learning rate at iter 0")
    parser.add_argument("--lr-warmup-iters", type=int, default=200,
        help="linear warmup iterations from warmup_lr to lr")
    parser.add_argument("--final-lr", type=float, default=1.0e-5,
        help="final learning rate after cosine decay")
    parser.add_argument("--lr-decay-iters", type=int, default=8000,
        help="cosine decay iterations from peak lr to final_lr")
    parser.add_argument("--ent-coef", type=float, default=0.013,
        help="constant entropy bonus coefficient (0.013 in config)")

    # Value Loss arguments ("mse" or "ce", matching your YAML config!)
    parser.add_argument("--value-loss", type=str, default="mse", choices=["mse", "ce"],
        help="critic loss type: 'mse' (default in config, unclipped for composite_reward) or 'ce' (HL-Gauss)")
    parser.add_argument("--num-bins", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-10.0)
    parser.add_argument("--v-max", type=float, default=10.0)
    parser.add_argument("--hl-sigma", type=float, default=0.75)

    args = parser.parse_args()

    # Automatically load YAML config if it exists so all settings stay in sync
    args.curriculum = [
        {"iter": 0, "min_generals_distance": 4, "max_generals_distance": 9, "reward_fn": "composite_reward"},
        {"iter": 500, "min_generals_distance": 9, "max_generals_distance": 17, "reward_fn": "composite_reward"},
        {"iter": 2000, "min_generals_distance": 17, "max_generals_distance": 25, "reward_fn": "win_lose_reward"},
        {"iter": 5000, "min_generals_distance": 17, "max_generals_distance": 35, "reward_fn": "win_lose_reward"},
    ]
    if args.config and os.path.exists(args.config):
        try:
            import sys
            from ruamel.yaml import YAML
            yaml = YAML(typ="safe")
            with open(args.config, "r") as f:
                ycfg = yaml.load(f) or {}
            for k in [
                "pad_to", "min_grid_size", "max_grid_size", "reset_pool_every",
                "castle_val_min", "castle_val_max", "truncation", "num_iters",
                "num_steps", "minibatch_size", "seed", "gamma", "gae_lambda",
                "vf_coef", "ent_coef", "target_kl", "adv_top_frac", "lr",
                "warmup_lr", "lr_warmup_iters", "final_lr", "lr_decay_iters",
                "eval_every", "eval_games", "value_loss", "num_bins", "v_min",
                "v_max", "hl_sigma",
            ]:
                if k in ycfg and ycfg[k] is not None:
                    # ONLY override if the user didn't explicitly pass it via CLI
                    cli_flag = f"--{k.replace('_', '-')}"
                    if cli_flag not in sys.argv:
                        setattr(args, k, ycfg[k])
            
            if "clip_eps" in ycfg and "--clip-coef" not in sys.argv:
                args.clip_coef = float(ycfg["clip_eps"])
            if "num_epochs" in ycfg and "--update-epochs" not in sys.argv:
                args.update_epochs = int(ycfg["num_epochs"])
            if "embed_dim" in ycfg and "--d-model" not in sys.argv:
                args.d_model = int(ycfg["embed_dim"])
            if "n_head" in ycfg and "--heads" not in sys.argv:
                args.heads = int(ycfg["n_head"])
            if "ff_factor" in ycfg and "--ffn-mult" not in sys.argv:
                args.ffn_mult = int(ycfg["ff_factor"])
            
            if "curriculum" in ycfg and ycfg["curriculum"]:
                # Inherit reward_fn from previous stage if not specified in a stage
                cur_list = []
                last_rew = "composite_reward"
                for st in ycfg["curriculum"]:
                    last_rew = st.get("reward_fn", last_rew)
                    cur_list.append({
                        "iter": int(st["iter"]),
                        "min_generals_distance": int(st["min_generals_distance"]),
                        "max_generals_distance": int(st["max_generals_distance"]),
                        "reward_fn": last_rew,
                    })
                args.curriculum = cur_list
            print(f"Synced hyperparameters & {len(args.curriculum)}-stage iter curriculum from {args.config}")
        except Exception as e:
            print(f"Note: Could not parse {args.config} via ruamel.yaml ({e}), using built-in config defaults.")

    args.grid_size = args.pad_to
    args.num_trajectories = 2 * args.num_envs
    args.batch_size = int(args.num_steps * args.num_trajectories)
    args.n_keep = int(args.batch_size * args.adv_top_frac)
    args.n_keep = max((args.n_keep // args.minibatch_size) * args.minibatch_size, args.minibatch_size)
    # fmt: on
    return args


def compute_scheduled_lr(it: int, args) -> float:
    """
    Exact match to your YAML config's LR schedule:
      - Linear warmup from `warmup_lr` to `lr` over `lr_warmup_iters`
      - Cosine decay from `lr` to `final_lr` over `lr_decay_iters`
    """
    if args.lr_warmup_iters > 0 and it < args.lr_warmup_iters:
        frac = float(it + 1) / float(args.lr_warmup_iters)
        return args.warmup_lr + frac * (args.lr - args.warmup_lr)
    elif args.lr_decay_iters > 0:
        decay_it = it - max(args.lr_warmup_iters, 0)
        progress = min(max(float(decay_it) / float(args.lr_decay_iters), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.final_lr + (args.lr - args.final_lr) * cosine
    return args.lr


# =====================================================================
# 1. FEATURE PARTITIONING & PREPROCESSING (Section 3 of Spec)
# =====================================================================

NODE_CHANNEL_INDICES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    20, 21,
    22, 23,
    *range(24, 38)
]
GLOBAL_CHANNEL_INDICES = [14, 15, 16, 17, 18, 19]

NODE_SYMLOG_INDICES = [0, 1, 2, 3, 14, *range(18, 32)]
GLOBAL_SYMLOG_INDICES = [3, 5]


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def split_and_preprocess_obs(obs_38: torch.Tensor, already_normalized: bool = False):
    global_s = obs_38[:, GLOBAL_CHANNEL_INDICES, 0, 0].to(torch.float32)
    node_x = obs_38[:, NODE_CHANNEL_INDICES, :, :].to(torch.float32)

    if not already_normalized:
        global_s = global_s.clone()
        node_x = node_x.clone()
        global_s[:, GLOBAL_SYMLOG_INDICES] = symlog(global_s[:, GLOBAL_SYMLOG_INDICES])
        node_x[:, NODE_SYMLOG_INDICES, :, :] = symlog(node_x[:, NODE_SYMLOG_INDICES, :, :])

    return node_x, global_s


# =====================================================================
# 2. FAST 5x5 GRAPH ATTENTION + GLOBAL RELAY ARCHITECTURE (Section 5)
# =====================================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Fast5x5GATBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_mult: int = 3):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        self.norm1 = nn.LayerNorm(d_model)
        self.local_v_proj = layer_init(nn.Linear(d_model, d_model, bias=False))
        # Depthwise + Pointwise 5x5 Graph Convolution with zero directional bias
        self.local_5x5_conv = nn.Conv2d(
            d_model, d_model, kernel_size=5, padding=2, groups=heads, bias=False
        )
        self.local_gate_proj = layer_init(nn.Linear(d_model * 2, d_model))
        self.local_out_proj = layer_init(nn.Linear(d_model, d_model))

        # Fast O(N) Global Relay Conditioning (replaces slow nn.MultiheadAttention fallback)
        self.relay_proj = layer_init(nn.Linear(d_model * 3, d_model))
        self.relay_gate = layer_init(nn.Linear(d_model * 2, d_model))

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            layer_init(nn.Linear(d_model, d_model * ffn_mult)),
            nn.GELU(),
            layer_init(nn.Linear(d_model * ffn_mult, d_model)),
        )

    def forward(self, x: torch.Tensor, global_tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x: (B, N, d_model)
        global_tokens: (B, 3, d_model)
        """
        B, N, D = x.shape
        h = self.norm1(x)

        # 1. 5x5 Multi-Head Local Graph Message Passing (Hardware-fused 5x5 kernel)
        v_2d = self.local_v_proj(h).transpose(1, 2).reshape(B, D, H, W)
        nb_msg = self.local_5x5_conv(v_2d).reshape(B, D, N).transpose(1, 2)  # (B, N, D)

        # Attention gate between center node u and its 5x5 aggregated neighborhood
        attn_gate = torch.sigmoid(self.local_gate_proj(torch.cat([h, nb_msg], dim=-1)))
        local_out = self.local_out_proj(attn_gate * nb_msg)

        # 2. Fast O(N) Global Relay Conditioning (no MQA/MHA kernel overhead!)
        g_flat = global_tokens.reshape(B, 1, D * 3)
        g_Broadcast = self.relay_proj(g_flat).expand(B, N, D)
        r_gate = torch.sigmoid(self.relay_gate(torch.cat([h, g_Broadcast], dim=-1)))
        relay_out = r_gate * g_Broadcast

        x = x + local_out + relay_out
        x = x + self.ffn(self.norm2(x))
        return x


class HLGaussCriticHead(nn.Module):
    """
    Centralized Critic using Cross-Attention Pooling over all N node embeddings + 3 global tokens.
    Supports BOTH:
      - `value_loss == "mse"`: Scalar value projection (unbounded, ideal for `composite_reward`)
      - `value_loss == "ce"`: Categorical HL-Gauss bins over `[v_min, v_max]`
    """
    def __init__(
        self,
        d_model: int,
        heads: int,
        value_loss: str = "mse",
        num_bins: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        sigma: float = 0.75,
    ):
        super().__init__()
        self.value_loss = value_loss
        self.num_bins = num_bins if value_loss == "ce" else 1
        self.v_min = v_min
        self.v_max = v_max
        self.sigma = sigma
        self.half_width = (v_max - v_min) / max(num_bins - 1, 1) / 2.0

        self.register_buffer("bin_centers", torch.linspace(v_min, v_max, max(num_bins, 2)))

        # Learned value query token q_val in R^(1 x 1 x d_model)
        self.q_val = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.out_norm = nn.LayerNorm(d_model)
        self.val_proj = layer_init(nn.Linear(d_model, self.num_bins), std=0.01)

    def forward(self, node_h: torch.Tensor, global_tokens: torch.Tensor):
        B = node_h.shape[0]
        kv = self.kv_norm(torch.cat([node_h, global_tokens], dim=1))  # (B, N + 3, d_model)
        q = self.q_val.expand(B, 1, -1)                               # (B, 1, d_model)

        pooled, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        pooled = self.out_norm(q + pooled).squeeze(1)                 # (B, d_model)

        val_raw = self.val_proj(pooled)                               # (B, 1) or (B, num_bins)
        if self.value_loss == "ce":
            probs = F.softmax(val_raw.float(), dim=-1)
            expected_val = torch.sum(probs * self.bin_centers, dim=-1)
            return expected_val, val_raw
        else:
            expected_val = val_raw.squeeze(-1).float()
            return expected_val, expected_val

    def compute_loss(self, val_aux: torch.Tensor, target_returns: torch.Tensor) -> torch.Tensor:
        """
        Exact PyTorch equivalent of `make_value_loss_fn` ("mse" or "ce") in `ppo.py`.
        """
        if self.value_loss == "mse":
            return 0.5 * (val_aux.float() - target_returns.float()).pow(2)

        ret = target_returns.unsqueeze(-1).clamp(self.v_min, self.v_max)
        centers = self.bin_centers.unsqueeze(0)

        upper = (centers + self.half_width - ret) / (self.sigma * math.sqrt(2.0))
        lower = (centers - self.half_width - ret) / (self.sigma * math.sqrt(2.0))

        target_probs = 0.5 * (torch.erf(upper) - torch.erf(lower))
        target_probs = target_probs / torch.clamp(target_probs.sum(dim=-1, keepdim=True), min=1e-8)

        log_probs = F.log_softmax(val_aux.float(), dim=-1)
        return -torch.sum(target_probs * log_probs, dim=-1)


class GeneralsMARLGATAgent(nn.Module):
    """
    Full CTDE Cell-as-Agent 5x5 GAT Network for Generals.io.
    """
    def __init__(self, args, already_normalized_obs: bool = False):
        super().__init__()
        self.args = args
        self.already_normalized_obs = already_normalized_obs
        self.H = args.grid_size
        self.W = args.grid_size
        self.N = self.H * self.W
        self.d_model = args.d_model

        # 1. Node Projection
        self.node_proj = nn.Sequential(
            layer_init(nn.Linear(32, args.d_model)),
            nn.LayerNorm(args.d_model),
        )

        # 2. Global Context MLPs (Channels 14-19 scalar MLP + 2 Time-Series MLPs, matching TemporalEncoder in transformer.py)
        self.scalar_mlp = nn.Sequential(
            layer_init(nn.Linear(6, args.d_model)),
            nn.SiLU(),
            layer_init(nn.Linear(args.d_model, args.d_model)),
            nn.LayerNorm(args.d_model),
        )
        self.army_ts_mlp = nn.Sequential(
            layer_init(nn.Linear(512, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, args.d_model)),
            nn.LayerNorm(args.d_model),
        )
        self.land_ts_mlp = nn.Sequential(
            layer_init(nn.Linear(512, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, args.d_model)),
            nn.LayerNorm(args.d_model),
        )

        # 3. Shared 1-Pass Fast 5x5 GAT Torso
        ffn_mult = getattr(args, "ffn_mult", 2)
        self.blocks = nn.ModuleList([
            Fast5x5GATBlock(args.d_model, args.heads, ffn_mult=ffn_mult)
            for _ in range(args.num_layers)
        ])
        self.final_norm = nn.LayerNorm(args.d_model)

        # 4. Decentralized Per-Node Actor Head (9 logits per node: pass + 4 dirs x {all, half})
        self.actor_head = layer_init(nn.Linear(args.d_model, 9), std=0.01)

        # 5. Centralized Cross-Attention Critic Head ("mse" or "ce")
        self.critic_head = HLGaussCriticHead(
            d_model=args.d_model,
            heads=args.heads,
            value_loss=getattr(args, "value_loss", "mse"),
            num_bins=args.num_bins,
            v_min=args.v_min,
            v_max=args.v_max,
            sigma=args.hl_sigma,
        )

    def _encode_torso(self, obs_38: torch.Tensor, temporal: torch.Tensor):
        """
        obs_38: (B, 38, H, W)
        temporal: (B, 2, 512) -> [:, 0] is opponent army history, [:, 1] is opponent land history
        """
        B = obs_38.shape[0]
        node_x_2d, global_s = split_and_preprocess_obs(obs_38, already_normalized=self.already_normalized_obs)
        node_x = node_x_2d.permute(0, 2, 3, 1).reshape(B, self.N, 32)  # (B, N, 32)
        temporal_f32 = temporal.to(torch.float32)

        # Global Context Embeddings (matching `temporal_data / 50.0` in transformer.py)
        g_scalar = self.scalar_mlp(global_s)                             # (B, d_model)
        g_army = self.army_ts_mlp(temporal_f32[:, 0, :] / 50.0)          # (B, d_model)
        g_land = self.land_ts_mlp(temporal_f32[:, 1, :] / 50.0)          # (B, d_model)

        g_ctx = (g_scalar + g_army + g_land).unsqueeze(1)                # (B, 1, d_model)
        global_tokens = torch.stack([g_scalar, g_army, g_land], dim=1)   # (B, 3, d_model)

        h = self.node_proj(node_x) + g_ctx                               # (B, N, d_model)

        for block in self.blocks:
            h = block(h, global_tokens, self.H, self.W)

        node_h = self.final_norm(h)
        return node_h, global_tokens

    def get_value(self, obs_38: torch.Tensor, temporal: torch.Tensor):
        node_h, global_tokens = self._encode_torso(obs_38, temporal)
        expected_val, _ = self.critic_head(node_h, global_tokens)
        return expected_val

    def _compute_masked_logits(self, node_h: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Produces flat action logits in `(B, 9 * H * W)` order, matching `networks/transformer.py`
        (`action_logits` shape `(9, p, p)` -> `.reshape(-1)`) so `decode_action` maps
        flat indices to the exact `[pass, row, col, dir, is_half]` action vector!

        CRITICAL FIX for the 529-Pass-Action Trap:
        In `prepare_action_mask`, Channel 0 (`pass`) is True across all H*W = 529 cells!
        If all 529 pass cells are left unmasked, 99.2% of random initial actions are `pass`
        and the agent never moves! We restrict `pass` to at most cell (0,0) (index 0), and
        if `allow_pass=False`, index 0 is only enabled when the player has 0 valid moves.
        """
        B = node_h.shape[0]
        node_logits = self.actor_head(node_h).view(B, self.H, self.W, 9).permute(0, 3, 1, 2)  # (B, 9, H, W)
        mask_9hw = masks.view(B, 9, self.H, self.W).bool().clone()

        # Zero out duplicate pass actions on cells 1..N-1 so there is only 1 pass action (at [0, 0, 0])
        mask_9hw[:, 0, :, :] = False
        has_valid_move = mask_9hw[:, 1:, :, :].reshape(B, -1).any(dim=-1)

        if self.args.allow_pass:
            # Enable single pass logit at (0, 0, 0)
            mask_9hw[:, 0, 0, 0] = True
        else:
            # Only allow pass when the player has no valid moves on the board!
            mask_9hw[:, 0, 0, 0] = ~has_valid_move

        flat_logits = node_logits.reshape(B, -1).float()
        flat_mask = mask_9hw.reshape(B, -1)
        return torch.where(flat_mask, flat_logits, -1e9)

    def get_action_and_value(
        self,
        obs_38: torch.Tensor,
        masks: torch.Tensor,
        temporal: torch.Tensor,
        action: torch.Tensor = None,
    ):
        node_h, global_tokens = self._encode_torso(obs_38, temporal)
        masked_logits = self._compute_masked_logits(node_h, masks)
        dist = Categorical(logits=masked_logits)

        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        expected_val, val_logits = self.critic_head(node_h, global_tokens)

        return action, logprob, entropy, expected_val, val_logits

    @torch.no_grad()
    def get_greedy_action(self, obs_38: torch.Tensor, masks: torch.Tensor, temporal: torch.Tensor):
        """Selects greedy (argmax) action for evaluation vs random bot."""
        node_h, _ = self._encode_torso(obs_38, temporal)
        masked_logits = self._compute_masked_logits(node_h, masks)
        return torch.argmax(masked_logits, dim=-1)


# =====================================================================
# 3. TRUNCATION-SAFE GAE & MC RETURNS (Exact PyTorch port of `ppo.py`)
# =====================================================================

@torch.no_grad()
def compute_gae_generals(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    T, N_traj = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(N_traj, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        term_t = terminated[t].bool()
        trunc_t = truncated[t].bool()
        done_t = (term_t | trunc_t).float()

        bootstrap = torch.where(term_t, torch.zeros_like(next_values[t]), next_values[t])
        delta = rewards[t] + gamma * bootstrap - values[t]

        nonterminal = 1.0 - done_t
        adv = delta + gamma * gae_lambda * nonterminal * last_adv
        advantages[t] = adv

        # Zero carry on truncation: delta is invalid and must not leak backward
        last_adv = torch.where(trunc_t, torch.zeros_like(adv), adv)

    return advantages


@torch.no_grad()
def compute_mc_returns_generals(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
):
    T, N_traj = rewards.shape
    mc_rets = torch.zeros_like(rewards)
    valid = torch.zeros_like(rewards)

    next_ret = torch.zeros(N_traj, device=rewards.device, dtype=rewards.dtype)
    next_valid = torch.zeros(N_traj, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        done_t = (terminated[t].bool() | truncated[t].bool()).float()
        ret = rewards[t] + gamma * next_ret * (1.0 - done_t)
        val_flag = torch.where(done_t > 0.5, torch.ones_like(next_valid), next_valid)

        mc_rets[t] = ret
        valid[t] = val_flag
        next_ret = ret
        next_valid = val_flag

    return mc_rets, valid


def compute_component_grad_norms(agent: nn.Module):
    total_sq = 0.0
    actor_sq = 0.0
    critic_sq = 0.0
    for name, p in agent.named_parameters():
        if p.grad is not None:
            g_sq = float(p.grad.detach().pow(2).sum().item())
            total_sq += g_sq
            if "actor_head" in name:
                actor_sq += g_sq
            elif "critic_head" in name:
                critic_sq += g_sq
    return math.sqrt(total_sq), math.sqrt(actor_sq), math.sqrt(critic_sq)


# =====================================================================
# 4. ZERO-COPY DLPACK BRIDGE & SYNTHETIC FALLBACK ENV
# =====================================================================

def jax_to_torch(j_arr) -> torch.Tensor:
    return torch.from_dlpack(j_arr)


def torch_to_jax(t_arr: torch.Tensor):
    return j_dlpack.from_dlpack(t_arr.contiguous())


class JAXGeneralsSelfPlayVecEnv:
    def __init__(self, args, device: torch.device):
        if not JAX_ENV_AVAILABLE:
            raise ImportError(
                "generals-bots is not installed! Install via:\n"
                "  pip install git+https://github.com/strakam/generals-bots.git"
            )
        jax.config.update("jax_default_matmul_precision", "tensorfloat32")
        self.args = args
        self.num_envs = args.num_envs
        self.num_trajectories = 2 * args.num_envs
        self.H = args.pad_to
        self.W = args.pad_to
        self.device = device
        self.key = jrandom.PRNGKey(args.seed)

        # Initialize Stage 0 from iter-based curriculum in args
        self.curriculum_stages = args.curriculum
        self.current_stage_idx = 0
        s0 = self.curriculum_stages[0]
        args.min_generals_distance = s0["min_generals_distance"]
        args.max_generals_distance = s0["max_generals_distance"]
        self.use_composite_reward = (s0.get("reward_fn", "composite_reward") == "composite_reward")

        env_sig = inspect.signature(GeneralsEnv.__init__).parameters
        cities_kwarg = (
            {"num_castles_range": (9, 11)}
            if "num_castles_range" in env_sig
            else {"num_cities_range": (9, 11)}
        )

        self.env = GeneralsEnv(
            min_grid_size=args.min_grid_size,
            max_grid_size=args.max_grid_size,
            pad_to=self.H,
            min_generals_distance=args.min_generals_distance,
            max_generals_distance=args.max_generals_distance,
            truncation=args.truncation,
            pool_size=args.pool_size,
            castle_val_range=(args.castle_val_min, args.castle_val_max),
            mountain_density_range=(0.18, 0.22),
            **cities_kwarg,
        )

        if AVERAGEJOE_BUNDLE_AVAILABLE:
            bundle = get_network_bundle("history_transformer")
            self.init_obs_state_fn = bundle["init_obs_state"]
            self.augment_fn = bundle["augment_obs"]
            self.reset_obs_fn = bundle["reset_obs_state"]
        else:
            self.init_obs_state_fn = self._init_obs_state_builtin
            self.augment_fn = self._augment_obs_builtin
            self.reset_obs_fn = self._reset_obs_state_builtin

        self._vmap_init = jax.jit(jax.vmap(self.env.init_state))
        self._jit_extract_obs = self._build_jit_extract_obs()
        self._jit_step_and_obs = self._build_jit_step_and_obs()
        self.pool = None
        self.states = None
        self.obs_state_p0 = None
        self.obs_state_p1 = None

    def _build_jit_extract_obs(self):
        H, W, num_envs = self.H, self.W, self.num_envs
        augment_fn = self.augment_fn

        if AVERAGEJOE_BUNDLE_AVAILABLE:
            @jax.jit
            def _extract(states, obs_state_p0, obs_state_p1):
                raw_p0 = jax.vmap(lambda s: get_observation(s, 0))(states)
                raw_p1 = jax.vmap(lambda s: get_observation(s, 1))(states)
                arr_p0 = jax.vmap(obs_to_array)(raw_p0)
                arr_p1 = jax.vmap(obs_to_array)(raw_p1)
                mask_p0 = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(raw_p0)
                mask_p1 = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(raw_p1)

                obs_p0, new_osp0 = jax.vmap(augment_fn)(arr_p0, obs_state_p0)
                obs_p1, new_osp1 = jax.vmap(augment_fn)(arr_p1, obs_state_p1)

                # Apply AverageJoe's exact normalize_observations!
                obs_p0 = jax.vmap(normalize_observations)(obs_p0)
                obs_p1 = jax.vmap(normalize_observations)(obs_p1)

                temp_p0 = jnp.stack([new_osp0.opponent_army_history, new_osp0.opponent_land_history], axis=1)
                temp_p1 = jnp.stack([new_osp1.opponent_army_history, new_osp1.opponent_land_history], axis=1)

                prep_p0 = jax.vmap(lambda m: prepare_action_mask(m, H, allow_pass=False) > -1e5)(mask_p0)
                prep_p1 = jax.vmap(lambda m: prepare_action_mask(m, H, allow_pass=False) > -1e5)(mask_p1)

                obs_both = jnp.concatenate([obs_p0, obs_p1], axis=0)
                masks_both = jnp.concatenate([prep_p0, prep_p1], axis=0)
                temp_both = jnp.concatenate([temp_p0, temp_p1], axis=0)
                return obs_both, masks_both, temp_both, new_osp0, new_osp1
            return _extract
        else:
            @jax.jit
            def _extract(states, obs_state_p0, obs_state_p1):
                raw_p0 = jax.vmap(lambda s: get_observation(s, 0))(states)
                raw_p1 = jax.vmap(lambda s: get_observation(s, 1))(states)
                mask_p0 = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(raw_p0)
                mask_p1 = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(raw_p1)

                obs_p0, temp_p0, new_osp0 = augment_fn(raw_p0, obs_state_p0)
                obs_p1, temp_p1, new_osp1 = augment_fn(raw_p1, obs_state_p1)

                pass_p0 = jnp.zeros((num_envs, 1, H, W), dtype=jnp.bool_)
                move_p0 = jnp.transpose(mask_p0.astype(jnp.bool_), (0, 3, 1, 2))
                prep_p0 = jnp.concatenate([pass_p0, move_p0, move_p0], axis=1)
                move_p1 = jnp.transpose(mask_p1.astype(jnp.bool_), (0, 3, 1, 2))
                prep_p1 = jnp.concatenate([pass_p0, move_p1, move_p1], axis=1)

                obs_both = jnp.concatenate([obs_p0, obs_p1], axis=0)
                masks_both = jnp.concatenate([prep_p0, prep_p1], axis=0)
                temp_both = jnp.concatenate([temp_p0, temp_p1], axis=0)
                return obs_both, masks_both, temp_both, new_osp0, new_osp1
            return _extract

    def _build_jit_step_and_obs(self):
        H, W, num_envs = self.H, self.W, self.num_envs
        env = self.env
        extract_fn = self._jit_extract_obs
        reset_fn = reset_done_envs if AVERAGEJOE_BUNDLE_AVAILABLE else self.reset_obs_fn

        @jax.jit
        def _step_and_obs(states, obs_state_p0, obs_state_p1, pool, act_jax, use_composite_flag):
            act_p0_flat = act_jax[:num_envs]
            act_p1_flat = act_jax[num_envs:]

            if AVERAGEJOE_BUNDLE_AVAILABLE:
                act_p0 = jax.vmap(lambda idx: decode_action(idx, H))(act_p0_flat)
                act_p1 = jax.vmap(lambda idx: decode_action(idx, H))(act_p1_flat)
            else:
                def _decode(idx):
                    act_type = idx // (H * W)
                    rem = idx % (H * W)
                    r = rem // W
                    c = rem % W
                    is_pass = jnp.where(act_type == 0, 1, 0)
                    direction = jnp.clip((act_type - 1) % 4, 0, 3)
                    is_half = jnp.where(act_type >= 5, 1, 0)
                    return jnp.stack([is_pass, r, c, direction, is_half], axis=-1).astype(jnp.int32)
                act_p0 = jax.vmap(_decode)(act_p0_flat)
                act_p1 = jax.vmap(_decode)(act_p1_flat)

            # Capture pre-step stats for dense composite_reward shaping (cities, land, army)
            prev_raw_p0 = jax.vmap(lambda s: get_observation(s, 0))(states)
            prev_castles_p0 = getattr(prev_raw_p0, "cities", getattr(prev_raw_p0, "castles", jnp.zeros_like(prev_raw_p0.owned_cells)))
            prev_cities_p0 = jnp.sum(prev_raw_p0.owned_cells * prev_castles_p0, axis=(1, 2)).astype(jnp.float32)
            prev_cities_p1 = jnp.sum(prev_raw_p0.opponent_cells * prev_castles_p0, axis=(1, 2)).astype(jnp.float32)
            prev_land_diff = (prev_raw_p0.owned_land_count - prev_raw_p0.opponent_land_count).astype(jnp.float32)
            prev_army_diff = (prev_raw_p0.owned_army_count - prev_raw_p0.opponent_army_count).astype(jnp.float32)

            actions = jnp.stack([act_p0, act_p1], axis=1)
            timesteps, new_states = jax.vmap(lambda s, a: env.step(s, a, pool))(states, actions)

            terminated = timesteps.terminated
            truncated = timesteps.truncated
            winners = timesteps.info.winner
            dones = terminated | truncated

            # Use `last_state` before auto-reset so terminal step city/land deltas are exact!
            end_states = getattr(timesteps, "last_state", new_states)
            post_raw_p0 = jax.vmap(lambda s: get_observation(s, 0))(end_states)
            post_castles_p0 = getattr(post_raw_p0, "cities", getattr(post_raw_p0, "castles", jnp.zeros_like(post_raw_p0.owned_cells)))
            owned_cities_p0 = jnp.sum(post_raw_p0.owned_cells * post_castles_p0, axis=(1, 2)).astype(jnp.float32)
            owned_cities_p1 = jnp.sum(post_raw_p0.opponent_cells * post_castles_p0, axis=(1, 2)).astype(jnp.float32)
            post_land_diff = (post_raw_p0.owned_land_count - post_raw_p0.opponent_land_count).astype(jnp.float32)
            post_army_diff = (post_raw_p0.owned_army_count - post_raw_p0.opponent_army_count).astype(jnp.float32)

            obs_state_p0 = reset_fn(obs_state_p0, dones)
            obs_state_p1 = reset_fn(obs_state_p1, dones)

            # Terminal win/lose reward (+1 / -1)
            r_term_p0 = jnp.where(terminated, jnp.where(winners == 0, 1.0, -1.0), 0.0)

            # Fixed Dense Composite Reward Shaping:
            # 1. Strong positive reward (+0.35) for capturing a castle + continuous holding bonus (+0.005/step per castle)
            # 2. NEVER penalize `d_army < 0` when capturing neutral tiles/castles (which prevented castle captures!)
            city_gain_p0 = jnp.maximum(owned_cities_p0 - prev_cities_p0, 0.0)
            city_gain_p1 = jnp.maximum(owned_cities_p1 - prev_cities_p1, 0.0)
            d_cities = (city_gain_p0 - city_gain_p1)
            hold_cities = (owned_cities_p0 - owned_cities_p1) * 0.005

            d_land = jnp.clip((post_land_diff - prev_land_diff) / 20.0, -0.1, 0.1)
            d_army = jnp.clip((post_army_diff - prev_army_diff) / 200.0, 0.0, 0.05)  # Only reward net army growth, don't punish castle investment!
            r_shaping_p0 = jnp.where(dones, 0.0, 0.35 * d_cities + hold_cities + 0.10 * d_land + d_army)

            r_p0 = r_term_p0 + jnp.where(use_composite_flag, r_shaping_p0, 0.0)
            r_p1 = -r_p0

            rewards = jnp.concatenate([r_p0, r_p1], axis=0)
            term_both = jnp.concatenate([terminated, terminated], axis=0)
            trunc_both = jnp.concatenate([truncated, truncated], axis=0)
            win_both = jnp.concatenate([winners, 1 - winners], axis=0)

            obs_both, masks_both, temp_both, obs_state_p0, obs_state_p1 = extract_fn(
                new_states, obs_state_p0, obs_state_p1
            )
            return (
                new_states, obs_state_p0, obs_state_p1,
                obs_both, masks_both, temp_both,
                rewards, term_both, trunc_both, win_both, owned_cities_p0
            )

        return _step_and_obs

    def _init_obs_state_builtin(self, grid_size: int, pad_to: int):
        H = W = pad_to
        return {
            "seen": jnp.zeros((H, W), dtype=jnp.float32),
            "enemy_seen": jnp.zeros((H, W), dtype=jnp.float32),
            "generals": jnp.zeros((H, W), dtype=jnp.float32),
            "castles": jnp.zeros((H, W), dtype=jnp.float32),
            "mountains": jnp.zeros((H, W), dtype=jnp.float32),
            "last_enemy_army_seen": jnp.zeros((H, W), dtype=jnp.float32),
            "last_enemy_army_age": jnp.zeros((H, W), dtype=jnp.float32),
            "prev_own_army": jnp.zeros((H, W), dtype=jnp.float32),
            "prev_enemy_army": jnp.zeros((H, W), dtype=jnp.float32),
            "own_army_deltas": jnp.zeros((7, H, W), dtype=jnp.float32),
            "enemy_army_deltas": jnp.zeros((7, H, W), dtype=jnp.float32),
            "temporal": jnp.zeros((2, 512), dtype=jnp.float32),
        }

    @staticmethod
    def _reset_obs_state_builtin(obs_state, done_mask):
        def _zero_where(arr):
            view = done_mask.reshape((-1,) + (1,) * (arr.ndim - 1))
            return jnp.where(view, jnp.zeros_like(arr), arr)
        return jax.tree.map(_zero_where, obs_state)

    def _augment_obs_builtin(self, raw_obs, obs_state):
        H, W = self.H, self.W
        armies = getattr(raw_obs, "armies").astype(jnp.float32)
        owned = getattr(raw_obs, "owned_cells").astype(jnp.float32)
        opp = getattr(raw_obs, "opponent_cells").astype(jnp.float32)
        neutral = getattr(raw_obs, "neutral_cells").astype(jnp.float32)
        fog = getattr(raw_obs, "fog_cells").astype(jnp.float32)
        struct_fog = getattr(raw_obs, "structures_in_fog").astype(jnp.float32)
        mountains_vis = getattr(raw_obs, "mountains").astype(jnp.float32)
        generals_vis = getattr(raw_obs, "generals").astype(jnp.float32)
        castles_vis = getattr(raw_obs, "cities", getattr(raw_obs, "castles", jnp.zeros_like(armies))).astype(jnp.float32)
        timestep = getattr(raw_obs, "timestep").astype(jnp.float32)
        own_land = getattr(raw_obs, "owned_land_count").astype(jnp.float32)
        own_army_tot = getattr(raw_obs, "owned_army_count").astype(jnp.float32)
        opp_land = getattr(raw_obs, "opponent_land_count").astype(jnp.float32)
        opp_army_tot = getattr(raw_obs, "opponent_army_count").astype(jnp.float32)

        vis = 1.0 - fog
        own_army = armies * owned
        enemy_army = armies * opp
        neutral_army = armies * neutral

        seen = jnp.maximum(obs_state["seen"], vis)
        enemy_seen = jnp.maximum(obs_state["enemy_seen"], opp)
        generals = jnp.maximum(obs_state["generals"], generals_vis)
        castles = jnp.maximum(obs_state["castles"], castles_vis)
        mountains = jnp.maximum(obs_state["mountains"], mountains_vis)

        last_enemy_army = jnp.where(opp > 0.5, enemy_army, obs_state["last_enemy_army_seen"])
        raw_age = jnp.where(opp > 0.5, 0.0, obs_state["last_enemy_army_age"] + 1.0)
        log_age = jnp.log1p(raw_age) / 5.0

        d_own = own_army - obs_state["prev_own_army"]
        d_enemy = enemy_army - obs_state["prev_enemy_army"]
        own_deltas = jnp.concatenate([d_own[:, None], obs_state["own_army_deltas"][:, :-1]], axis=1)
        enemy_deltas = jnp.concatenate([d_enemy[:, None], obs_state["enemy_army_deltas"][:, :-1]], axis=1)

        ones_hw = jnp.ones((1, H, W), dtype=jnp.float32)
        ch14 = (timestep[:, None, None] / 1000.0) * ones_hw
        ch15 = ((timestep[:, None, None] % 50.0) / 50.0) * ones_hw
        ch16 = (own_land[:, None, None] / float(H * W)) * ones_hw
        ch17 = own_army_tot[:, None, None] * ones_hw
        ch18 = (opp_land[:, None, None] / float(H * W)) * ones_hw
        ch19 = opp_army_tot[:, None, None] * ones_hw

        B = armies.shape[0]
        cx = jnp.broadcast_to(jnp.linspace(0.0, 1.0, W)[None, None, :], (B, H, W))
        cy = jnp.broadcast_to(jnp.linspace(0.0, 1.0, H)[None, :, None], (B, H, W))

        base_channels = jnp.stack([
            armies, own_army, enemy_army, neutral_army,
            seen, enemy_seen, generals, castles, mountains,
            neutral, owned, opp, fog, struct_fog,
            ch14, ch15, ch16, ch17, ch18, ch19,
            last_enemy_army, log_age, cx, cy
        ], axis=1)

        obs_38 = jnp.concatenate([base_channels, own_deltas, enemy_deltas], axis=1)

        new_temp = jnp.roll(obs_state["temporal"], shift=-1, axis=-1)
        new_temp = new_temp.at[:, 0, -1].set(opp_army_tot)
        new_temp = new_temp.at[:, 1, -1].set(opp_land)

        new_obs_state = {
            "seen": seen,
            "enemy_seen": enemy_seen,
            "generals": generals,
            "castles": castles,
            "mountains": mountains,
            "last_enemy_army_seen": last_enemy_army,
            "last_enemy_army_age": raw_age,
            "prev_own_army": own_army,
            "prev_enemy_army": enemy_army,
            "own_army_deltas": own_deltas,
            "enemy_army_deltas": enemy_deltas,
            "temporal": new_temp,
        }
        return obs_38, new_temp, new_obs_state

    def update_curriculum_by_iter(self, it: int):
        """
        Advances curriculum stage based on the exact `iter:` schedule in your YAML config:
          - iter 0: dist 4-9, composite_reward
          - iter 500: dist 9-17, composite_reward
          - iter 2000: dist 17-25, win_lose_reward
          - iter 5000: dist 17-35, win_lose_reward
        """
        target_idx = self.current_stage_idx
        for idx, st in enumerate(self.curriculum_stages):
            if it >= st["iter"]:
                target_idx = idx
        if target_idx != self.current_stage_idx:
            self.current_stage_idx = target_idx
            st = self.curriculum_stages[target_idx]
            min_d, max_d = st["min_generals_distance"], st["max_generals_distance"]
            rew_name = st.get("reward_fn", "win_lose_reward")
            self.use_composite_reward = (rew_name == "composite_reward")
            self.env.min_generals_distance = min_d
            self.env.max_generals_distance = max_d
            self.regenerate_pool()
            print(
                f"--> CURRICULUM STAGE {target_idx} (iter {it}): "
                f"spawn distance [{min_d}, {max_d}], reward_fn={rew_name}"
            )

    def regenerate_pool(self):
        self.key, pool_key = jrandom.split(self.key)
        self.pool, _ = self.env.reset(pool_key)

    def reset(self):
        if self.pool is None:
            self.regenerate_pool()
        self.key, init_key = jrandom.split(self.key)
        keys = jrandom.split(init_key, self.num_envs)
        self.states = self._vmap_init(keys)

        single_state = self.init_obs_state_fn(self.H, self.W)
        self.obs_state_p0 = jax.tree.map(lambda x: jnp.tile(x, (self.num_envs, *([1] * x.ndim))), single_state)
        self.obs_state_p1 = jax.tree.map(lambda x: jnp.tile(x, (self.num_envs, *([1] * x.ndim))), single_state)

        return self._extract_torch_obs()

    def _extract_torch_obs(self):
        obs_both, masks_both, temp_both, self.obs_state_p0, self.obs_state_p1 = self._jit_extract_obs(
            self.states, self.obs_state_p0, self.obs_state_p1
        )
        return jax_to_torch(obs_both), jax_to_torch(masks_both), jax_to_torch(temp_both)

    def step(self, flat_actions: torch.Tensor):
        act_jax = torch_to_jax(flat_actions)
        use_comp = jnp.bool_(self.use_composite_reward)
        (
            self.states, self.obs_state_p0, self.obs_state_p1,
            obs_both, masks_both, temp_both,
            rewards, term_both, trunc_both, win_both, owned_cities
        ) = self._jit_step_and_obs(
            self.states, self.obs_state_p0, self.obs_state_p1, self.pool, act_jax, use_comp
        )
        return (
            jax_to_torch(obs_both),
            jax_to_torch(masks_both),
            jax_to_torch(temp_both),
            jax_to_torch(rewards),
            jax_to_torch(term_both),
            jax_to_torch(trunc_both),
            jax_to_torch(win_both),
            jax_to_torch(owned_cities),
        )


class SyntheticGeneralsVecEnv:
    def __init__(self, num_envs: int, grid_size: int, device: torch.device):
        self.num_envs = num_envs
        self.num_trajectories = 2 * num_envs
        self.H = grid_size
        self.W = grid_size
        self.device = device
        self.step_counter = torch.zeros(num_envs, device=device, dtype=torch.long)

    def reset(self):
        B = self.num_trajectories
        obs = torch.randn(B, 38, self.H, self.W, device=self.device) * 0.1
        masks = torch.randint(0, 2, (B, 9, self.H, self.W), device=self.device, dtype=torch.bool)
        masks[:, 1, 0, 0] = True
        temporal = torch.abs(torch.randn(B, 2, 512, device=self.device))
        return obs, masks, temporal

    def step(self, flat_actions: torch.Tensor):
        self.step_counter += 1
        obs, masks, temporal = self.reset()

        term_games = (torch.rand(self.num_envs, device=self.device) < 0.02)
        trunc_games = (self.step_counter >= 500) & (~term_games)

        p0_wins = torch.randint(0, 2, (self.num_envs,), device=self.device).bool()
        r_p0 = torch.where(term_games, torch.where(p0_wins, 1.0, -1.0), 0.0)
        r_p1 = -r_p0

        rewards = torch.cat([r_p0, r_p1], dim=0)
        terminated = torch.cat([term_games, term_games], dim=0)
        truncated = torch.cat([trunc_games, trunc_games], dim=0)
        winners = torch.cat([torch.where(p0_wins, 0, 1), torch.where(p0_wins, 1, 0)], dim=0)
        owned_cities = torch.ones(self.num_envs, device=self.device, dtype=torch.float32) * 1.5

        self.step_counter = torch.where(term_games | trunc_games, torch.zeros_like(self.step_counter), self.step_counter)
        return obs, masks, temporal, rewards, terminated, truncated, winners, owned_cities


@torch.no_grad()
def evaluate_vs_random(agent: GeneralsMARLGATAgent, envs, n_games: int = 64):
    """
    Symmetric double-sided evaluation (matches `evaluate()` in `AverageJoe/train/evaluations.py`):
    Plays `n_games // 2` with GAT as Player 0 and `n_games // 2` with GAT as Player 1
    against a masked random bot so any directional/spawn bias is measured fairly.
    """
    agent.eval()
    num_envs = envs.num_envs
    n_half = min(max(n_games // 2, 1), num_envs)
    H, W = envs.H, envs.W
    max_steps = getattr(envs.args, "truncation", 500) if hasattr(envs, "args") else 100

    total_w, total_l, total_d, total_fin = 0, 0, 0, 0

    for gat_seat in (0, 1):
        obs, masks, temporal = envs.reset()
        finished = torch.zeros(n_half, device=obs.device, dtype=torch.bool)
        won = torch.zeros(n_half, device=obs.device, dtype=torch.bool)
        lost = torch.zeros(n_half, device=obs.device, dtype=torch.bool)
        drew = torch.zeros(n_half, device=obs.device, dtype=torch.bool)

        for _ in range(max_steps):
            if finished.all():
                break
            # Compute random actions for the random seat
            rand_seat_slice = slice(num_envs, 2 * num_envs) if gat_seat == 0 else slice(0, num_envs)
            gat_seat_slice = slice(0, num_envs) if gat_seat == 0 else slice(num_envs, 2 * num_envs)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(obs.device.type == "cuda")):
                a_gat = agent.get_greedy_action(obs[gat_seat_slice], masks[gat_seat_slice], temporal[gat_seat_slice])

            mask_rnd = masks[rand_seat_slice].view(num_envs, 9, H, W).bool().clone()
            mask_rnd[:, 0, :, :] = False
            has_move_rnd = mask_rnd[:, 1:, :, :].reshape(num_envs, -1).any(dim=-1)
            mask_rnd[:, 0, 0, 0] = ~has_move_rnd
            flat_mask_rnd = mask_rnd.reshape(num_envs, -1)
            rand_logits = torch.where(flat_mask_rnd, 0.0, -1e9)
            a_rnd = Categorical(logits=rand_logits).sample()

            actions = torch.cat([a_gat, a_rnd], dim=0) if gat_seat == 0 else torch.cat([a_rnd, a_gat], dim=0)
            obs, masks, temporal, _, terminated, truncated, winners, _ = envs.step(actions)

            term_g = terminated[:n_half]
            trunc_g = truncated[:n_half]
            win_p0 = winners[:n_half]
            dones = term_g | trunc_g
            new_done = dones & (~finished)

            won |= new_done & term_g & (win_p0 == gat_seat)
            lost |= new_done & term_g & (win_p0 == (1 - gat_seat))
            drew |= new_done & trunc_g & (~term_g)
            finished |= dones

        total_w += int(won.sum().item())
        total_l += int(lost.sum().item())
        total_d += int(drew.sum().item())
        total_fin += int(finished.sum().item())

    agent.train()
    wr = total_w / max(total_fin, 1)
    return total_w, total_l, total_d, total_fin, wr


@torch.no_grad()
def evaluate_head_to_head(
    agent_a: GeneralsMARLGATAgent,
    agent_b: GeneralsMARLGATAgent,
    envs,
    n_games: int = 64,
):
    """
    Plays `n_games` symmetric head-to-head games between `agent_a` and `agent_b`
    (half with A as Player 0, half with A as Player 1).
    Returns (a_wins, b_wins, draws, total_finished).
    """
    agent_a.eval()
    agent_b.eval()
    num_envs = envs.num_envs
    n_half = min(max(n_games // 2, 1), num_envs)
    max_steps = getattr(envs.args, "truncation", 500) if hasattr(envs, "args") else 100

    a_wins, b_wins, draws, total_fin = 0, 0, 0, 0

    for a_seat in (0, 1):
        obs, masks, temporal = envs.reset()
        finished = torch.zeros(n_half, device=obs.device, dtype=torch.bool)
        won = torch.zeros(n_half, device=obs.device, dtype=torch.bool)
        lost = torch.zeros(n_half, device=obs.device, dtype=torch.bool)
        drew = torch.zeros(n_half, device=obs.device, dtype=torch.bool)

        for _ in range(max_steps):
            if finished.all():
                break
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(obs.device.type == "cuda")):
                if a_seat == 0:
                    a_p0 = agent_a.get_greedy_action(obs[:num_envs], masks[:num_envs], temporal[:num_envs])
                    a_p1 = agent_b.get_greedy_action(obs[num_envs:], masks[num_envs:], temporal[num_envs:])
                else:
                    a_p0 = agent_b.get_greedy_action(obs[:num_envs], masks[:num_envs], temporal[:num_envs])
                    a_p1 = agent_a.get_greedy_action(obs[num_envs:], masks[num_envs:], temporal[num_envs:])

            actions = torch.cat([a_p0, a_p1], dim=0)
            obs, masks, temporal, _, terminated, truncated, winners, _ = envs.step(actions)

            term_g = terminated[:n_half]
            trunc_g = truncated[:n_half]
            win_p0 = winners[:n_half]
            dones = term_g | trunc_g
            new_done = dones & (~finished)

            won |= new_done & term_g & (win_p0 == a_seat)
            lost |= new_done & term_g & (win_p0 == (1 - a_seat))
            drew |= new_done & trunc_g & (~term_g)
            finished |= dones

        a_wins += int(won.sum().item())
        b_wins += int(lost.sum().item())
        draws += int(drew.sum().item())
        total_fin += int(finished.sum().item())

    agent_a.train()
    return a_wins, b_wins, draws, total_fin


class EloLeagueTracker:
    """
    Self-contained Historical Checkpoint Elo League (analogous to `ref_eval` in `AverageJoe`).
    Maintains a pool of frozen historical snapshots of the agent (stored on CPU RAM to use 0 VRAM),
    evaluates the current EMA agent against them + RandomBot, and computes exact Elo ratings.
    """
    def __init__(self, template_agent: GeneralsMARLGATAgent, device: torch.device, max_snapshots: int = 6):
        self.device = device
        self.max_snapshots = max_snapshots
        self.opponent_model = copy.deepcopy(template_agent).to(device)
        for p in self.opponent_model.parameters():
            p.requires_grad_(False)
        self.opponent_model.eval()

        # Snapshots: list of (name, cpu_state_dict, elo_rating)
        self.snapshots = []
        self.random_elo = 1000.0
        self.current_elo = 1000.0

    def add_snapshot(self, name: str, agent: GeneralsMARLGATAgent, elo: float):
        cpu_state = {k.replace("_orig_mod.", ""): v.detach().cpu().clone() for k, v in agent.state_dict().items()}
        self.snapshots.append({"name": name, "state_dict": cpu_state, "elo": float(elo)})
        if len(self.snapshots) > self.max_snapshots:
            # Keep the earliest baseline (index 0) and the most recent snapshots
            self.snapshots.pop(1)

    def evaluate_and_update_elo(
        self,
        candidate_agent: GeneralsMARLGATAgent,
        envs,
        rand_w: int,
        rand_l: int,
        rand_d: int,
        n_games_per_opp: int = 64,
    ):
        """
        Plays `candidate_agent` against historical snapshots in `self.snapshots`,
        combines with vs-Random results, and solves for the maximum-likelihood Elo rating.
        """
        results = []
        # 1. Include vs-Random match result (anchored at 1000 Elo)
        results.append(("RandomBot", self.random_elo, rand_w, rand_l, rand_d))

        # 2. Play against frozen historical checkpoints
        last_ckpt_wr = 0.5
        for snap in self.snapshots:
            self.opponent_model.load_state_dict(snap["state_dict"], strict=False)
            w, l, d, fin = evaluate_head_to_head(
                candidate_agent, self.opponent_model, envs, n_games=n_games_per_opp
            )
            results.append((snap["name"], snap["elo"], w, l, d))
            last_ckpt_wr = (w + 0.5 * d) / max(w + l + d, 1)

        # 3. Solve for Candidate Elo via Newton-Raphson / Bradley-Terry MLE anchored on opponent Elos
        elo = self.current_elo
        for _ in range(50):
            grad = 0.0
            hess = 0.0
            ln10_400 = math.log(10.0) / 400.0
            for _, opp_elo, w, l, d in results:
                n = w + l + d
                if n == 0:
                    continue
                score = w + 0.5 * d
                expected = 1.0 / (1.0 + math.pow(10.0, (opp_elo - elo) / 400.0))
                grad += ln10_400 * (score - n * expected)
                hess += (ln10_400 ** 2) * n * expected * (1.0 - expected)
            if hess > 1e-8:
                elo = elo + max(min(grad / hess, 100.0), -100.0)

        self.current_elo = elo
        return self.current_elo, last_ckpt_wr, results


@torch.no_grad()
def record_replay_video_headless(
    agent: GeneralsMARLGATAgent,
    envs,
    out_path: str,
    mode: str = "selfplay",
    fps: int = 10,
):
    """
    Records a single full game replay to MP4 (or GIF fallback) inside a screenless
    Docker container using `SDL_VIDEODRIVER="dummy"` and `ReplayGUI`.
    """
    if not isinstance(envs, JAXGeneralsSelfPlayVecEnv):
        return
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    try:
        import pygame
        import imageio
        from generals.gui import ReplayGUI
    except Exception as e:
        print(f"  [Video Skipped: {e}]")
        return

    agent.eval()
    H, W = envs.H, envs.W
    device = envs.device

    envs.key, init_key = jrandom.split(envs.key)
    state = envs.env.init_state(init_key)
    single_obs_st = envs.init_obs_state_fn(H, W)
    osp0 = jax.tree.map(lambda x: x[None, ...], single_obs_st)
    osp1 = jax.tree.map(lambda x: x[None, ...], single_obs_st)

    labels = ["GAT (Red)", "GAT (Blue)" if mode == "selfplay" else "Random (Blue)"]
    gui = ReplayGUI(state, agent_ids=labels)
    frames = []

    def _grab():
        surf = pygame.display.get_surface()
        arr = np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
        # Crop 1px if height or width is odd (e.g. 1086x751 -> 1086x750) so libx264 yuv420p succeeds
        h_even = arr.shape[0] - (arr.shape[0] % 2)
        w_even = arr.shape[1] - (arr.shape[1] % 2)
        return arr[:h_even, :w_even, :].copy()

    gui.tick(fps=None)
    frames.append(_grab())

    max_steps = envs.args.truncation
    winner = -1
    for step in range(max_steps):
        states_b = jax.tree.map(lambda x: x[None, ...], state)
        raw_p0 = jax.vmap(lambda s: get_observation(s, 0))(states_b)
        raw_p1 = jax.vmap(lambda s: get_observation(s, 1))(states_b)
        mask_p0 = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(raw_p0)
        mask_p1 = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(raw_p1)

        if AVERAGEJOE_BUNDLE_AVAILABLE:
            arr_p0 = jax.vmap(obs_to_array)(raw_p0)
            arr_p1 = jax.vmap(obs_to_array)(raw_p1)
            obs_p0, osp0 = jax.vmap(envs.augment_fn)(arr_p0, osp0)
            obs_p1, osp1 = jax.vmap(envs.augment_fn)(arr_p1, osp1)
            obs_p0 = jax.vmap(normalize_observations)(obs_p0)
            obs_p1 = jax.vmap(normalize_observations)(obs_p1)
            temp_p0 = jnp.stack([osp0.opponent_army_history, osp0.opponent_land_history], axis=1)
            temp_p1 = jnp.stack([osp1.opponent_army_history, osp1.opponent_land_history], axis=1)
            prep_p0 = jax.vmap(lambda m: prepare_action_mask(m, H, allow_pass=False) > -1e5)(mask_p0)
            prep_p1 = jax.vmap(lambda m: prepare_action_mask(m, H, allow_pass=False) > -1e5)(mask_p1)
        else:
            obs_p0, temp_p0, osp0 = envs.augment_fn(raw_p0, osp0)
            obs_p1, temp_p1, osp1 = envs.augment_fn(raw_p1, osp1)
            pass_z = jnp.zeros((1, 1, H, W), dtype=jnp.bool_)
            m0 = jnp.transpose(mask_p0.astype(jnp.bool_), (0, 3, 1, 2))
            m1 = jnp.transpose(mask_p1.astype(jnp.bool_), (0, 3, 1, 2))
            prep_p0 = jnp.concatenate([pass_z, m0, m0], axis=1)
            prep_p1 = jnp.concatenate([pass_z, m1, m1], axis=1)

        t_obs0, t_mask0, t_temp0 = jax_to_torch(obs_p0), jax_to_torch(prep_p0), jax_to_torch(temp_p0)
        t_obs1, t_mask1, t_temp1 = jax_to_torch(obs_p1), jax_to_torch(prep_p1), jax_to_torch(temp_p1)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            a0_idx = int(agent.get_greedy_action(t_obs0, t_mask0, t_temp0).item())
            if mode == "selfplay":
                a1_idx = int(agent.get_greedy_action(t_obs1, t_mask1, t_temp1).item())
            else:
                m_rnd = t_mask1.view(1, 9, H, W).bool().clone()
                m_rnd[:, 0, :, :] = False
                m_rnd[:, 0, 0, 0] = ~m_rnd[:, 1:, :, :].reshape(1, -1).any(dim=-1)
                a1_idx = int(Categorical(logits=torch.where(m_rnd.reshape(1, -1), 0.0, -1e9)).sample().item())

        act_p0 = decode_action(jnp.int32(a0_idx), H)
        act_p1 = decode_action(jnp.int32(a1_idx), H)
        actions = jnp.stack([act_p0, act_p1], axis=0)
        timestep, state = envs.env.step(state, actions, envs.pool)

        disp_state = timestep.last_state if (timestep.terminated or timestep.truncated) else state
        gui.update(disp_state, timestep.info)
        gui.tick(fps=None)
        frames.append(_grab())

        if timestep.terminated or timestep.truncated:
            winner = int(timestep.info.winner)
            break

    for _ in range(fps):
        frames.append(frames[-1])
    gui.close()
    agent.train()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    outcome = "P0Win" if winner == 0 else "P1Win" if winner == 1 else "Draw"
    final_path = out_path.replace(".mp4", f"_{outcome}_{step + 1}steps.mp4")
    try:
        imageio.mimsave(final_path, frames, fps=fps, macro_block_size=1)
        print(f"  [Replay Video Saved] -> {final_path} ({len(frames)} frames)")
    except Exception as e:
        gif_path = final_path.replace(".mp4", ".gif")
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"  [Replay GIF Saved ({e})] -> {gif_path} ({len(frames)} frames)")


# =====================================================================
# 5. MAIN TRAINING LOOP (CleanRL + Straka et al. PPO Recipe)
# =====================================================================

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 1. Initialize Environment first (syncs grid_size/pad_to from config)
    if JAX_ENV_AVAILABLE and not args.use_synthetic_env:
        print("Initializing live JAX GeneralsEnv (`generals-bots`) with zero-copy DLPack bridge...")
        envs = JAXGeneralsSelfPlayVecEnv(args=args, device=device)
    else:
        print(
            "Initializing SyntheticGeneralsVecEnv (Install `generals-bots` via "
            "`pip install git+https://github.com/strakam/generals-bots.git` for live JAX simulation)."
        )
        envs = SyntheticGeneralsVecEnv(num_envs=args.num_envs, grid_size=args.grid_size, device=device)

    H = W = envs.H
    args.grid_size = H

    # 2. Initialize Fast 5x5 GAT Agent & EMA Shadow Agent
    agent = GeneralsMARLGATAgent(args, already_normalized_obs=AVERAGEJOE_BUNDLE_AVAILABLE).to(device)
    ema_agent = copy.deepcopy(agent).to(device)
    for p in ema_agent.parameters():
        p.requires_grad_(False)

    # Initialize Historical Checkpoint Elo League Tracker (uses uncompiled ema_agent template)
    elo_league = EloLeagueTracker(ema_agent, device=device, max_snapshots=args.max_league_snapshots)
    elo_league.add_snapshot("iter_0", ema_agent, elo=1000.0)

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Initialized GeneralsMARLGATAgent ({H}x{W} grid) with {total_params / 1e6:.2f}M parameters on {device}.")

    # Optional Standalone Video Replay Evaluation Mode
    if args.eval_checkpoint is not None:
        print(f"Loading checkpoint from {args.eval_checkpoint} for headless video recording...")
        ckpt_state = torch.load(args.eval_checkpoint, map_location=device)
        # Strip `_orig_mod.` prefix if checkpoint was saved after torch.compile
        clean_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt_state.items()}
        agent.load_state_dict(clean_state, strict=False)
        for v_i in range(args.num_videos):
            vid_path = f"videos/eval_replay/game_{v_i + 1:03d}_{args.eval_mode}.mp4"
            record_replay_video_headless(
                agent, envs, out_path=vid_path, mode=args.eval_mode, fps=args.video_fps
            )
        writer.close()
        raise SystemExit(0)

    if args.compile and device.type == "cuda":
        print("Compiling GAT torso with torch.compile for fast fused CUDA kernels...")
        agent._encode_torso = torch.compile(agent._encode_torso)

    optimizer = optim.Adam(agent.parameters(), lr=args.warmup_lr, eps=1e-5)

    # 3. Rollout Storage Setup (bfloat16)
    T = args.num_steps
    B_traj = args.num_trajectories

    buf_obs = torch.zeros((T, B_traj, 38, H, W), device=device, dtype=torch.bfloat16)
    buf_masks = torch.zeros((T, B_traj, 9, H, W), device=device, dtype=torch.bool)
    buf_temporal = torch.zeros((T, B_traj, 2, 512), device=device, dtype=torch.bfloat16)
    buf_actions = torch.zeros((T, B_traj), device=device, dtype=torch.long)
    buf_logprobs = torch.zeros((T, B_traj), device=device, dtype=torch.float32)
    buf_values = torch.zeros((T, B_traj), device=device, dtype=torch.float32)
    buf_next_values = torch.zeros((T, B_traj), device=device, dtype=torch.float32)
    buf_rewards = torch.zeros((T, B_traj), device=device, dtype=torch.float32)
    buf_terminated = torch.zeros((T, B_traj), device=device, dtype=torch.bool)
    buf_truncated = torch.zeros((T, B_traj), device=device, dtype=torch.bool)
    buf_winners = torch.zeros((T, args.num_envs), device=device, dtype=torch.int32)
    buf_cities = torch.zeros(T, device=device, dtype=torch.float32)

    next_obs, next_masks, next_temporal = envs.reset()

    global_step = 0
    train_start = time.time()

    for it in range(args.num_iters):
        # Advance curriculum strictly by iteration schedule (`iter: 0, 500, 2000, 5000` in config)
        if isinstance(envs, JAXGeneralsSelfPlayVecEnv):
            envs.update_curriculum_by_iter(it)

        # ---- Periodic Evaluation vs Random Bot & Historical Elo League ----
        if args.eval_every > 0 and (it == 0 or (it + 1) % args.eval_every == 0):
            ew, el, ed, efin, eval_wr = evaluate_vs_random(agent, envs, n_games=args.eval_games)
            print(f"  EVAL (iter {it}): {ew}W/{el}L/{ed}D ({eval_wr * 100:.0f}%) greedy vs random (double-sided)")
            writer.add_scalar("eval/win_rate", eval_wr, it)
            writer.add_scalar("eval/wins", ew, it)
            writer.add_scalar("eval/losses", el, it)
            writer.add_scalar("eval/draws", ed, it)

            # Run Historical Elo League evaluation every `elo_eval_every` iterations
            if args.elo_eval_every > 0 and (it + 1) % args.elo_eval_every == 0:
                cur_elo, vs_prev_wr, h2h_summary = elo_league.evaluate_and_update_elo(
                    ema_agent, envs, rand_w=ew, rand_l=el, rand_d=ed, n_games_per_opp=min(args.eval_games, 128)
                )
                opp_strs = " | ".join(
                    [f"vs {name}({o_elo:.0f}): {w}W/{l}L/{d}D" for name, o_elo, w, l, d in h2h_summary]
                )
                print(f"  ELO LEAGUE (iter {it + 1}): EMA Elo = {cur_elo:.0f} | {opp_strs}")
                writer.add_scalar("eval/elo", cur_elo, it + 1)
                writer.add_scalar("eval/vs_prev_ckpt_wr", vs_prev_wr, it + 1)
                elo_league.add_snapshot(f"iter_{it + 1}", ema_agent, elo=cur_elo)

            next_obs, next_masks, next_temporal = envs.reset()

        # ---- Periodic Headless Replay Video Recording ----
        if args.capture_video and args.video_every > 0 and (it + 1) % args.video_every == 0:
            vid_path = f"videos/{run_name}/iter_{it + 1:05d}_{args.eval_mode}.mp4"
            record_replay_video_headless(
                ema_agent, envs, out_path=vid_path, mode=args.eval_mode, fps=args.video_fps
            )
            next_obs, next_masks, next_temporal = envs.reset()

        t0 = time.time()

        if isinstance(envs, JAXGeneralsSelfPlayVecEnv) and args.reset_pool_every > 0 and it > 0 and it % args.reset_pool_every == 0:
            envs.regenerate_pool()

        current_lr = compute_scheduled_lr(it, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        current_ent_coef = args.ent_coef

        # ---- PHASE 1: Rollout Collection ----
        t_rollout = time.time()

        for step in range(T):
            global_step += B_traj
            buf_obs[step] = next_obs.to(torch.bfloat16)
            buf_masks[step] = next_masks
            buf_temporal[step] = next_temporal.to(torch.bfloat16)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.cuda):
                action, logprob, _, value, _ = agent.get_action_and_value(
                    next_obs, next_masks, next_temporal
                )

            buf_actions[step] = action
            buf_logprobs[step] = logprob
            buf_values[step] = value.to(torch.float32)

            next_obs, next_masks, next_temporal, reward, terminated, truncated, winners, owned_cities = envs.step(action)

            buf_rewards[step] = reward
            buf_terminated[step] = terminated
            buf_truncated[step] = truncated
            buf_winners[step] = winners[:args.num_envs]
            buf_cities[step] = owned_cities.mean()

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.cuda):
            buf_next_values[:-1] = buf_values[1:]
            buf_next_values[-1] = agent.get_value(next_obs, next_temporal).to(torch.float32)

        term_p0_all = buf_terminated[:, :args.num_envs]
        dones_p0_all = term_p0_all | buf_truncated[:, :args.num_envs]
        eps_p0 = int(dones_p0_all.sum().item())
        wins_p0 = int((term_p0_all & (buf_winners == 0)).sum().item())
        losses_p0 = int((term_p0_all & (buf_winners == 1)).sum().item())
        total_owned_cities = float(buf_cities.sum().item())

        t_rollout = time.time() - t_rollout

        # ---- PHASE 2: Truncation-Safe GAE, MC Returns & Top-25% Advantage Filtering ----
        t_ppo = time.time()
        advs = compute_gae_generals(
            buf_rewards,
            buf_values,
            buf_next_values,
            buf_terminated,
            buf_truncated,
            args.gamma,
            args.gae_lambda,
        )
        rets = advs + buf_values
        adv_std_raw = float(advs.std().item())

        mean_owned_cities = total_owned_cities / float(T)
        mean_val = float(buf_values.mean().item())
        ret_mean = float(rets.mean().item())
        ret_std = float(rets.std().item())
        var_returns = float(rets.var().item())
        explained_var = 1.0 - float((rets - buf_values).var().item()) / max(var_returns, 1e-8)
        value_bias = float((buf_values - rets).mean().item())

        mc_rets, mc_valid = compute_mc_returns_generals(
            buf_rewards, buf_terminated, buf_truncated, args.gamma
        )
        mc_valid_count = float(mc_valid.sum().item())
        mc_valid_frac = mc_valid_count / max(float(mc_valid.numel()), 1.0)
        if mc_valid_count > 100:
            mc_diff = (buf_values - mc_rets) * mc_valid
            mc_mean_ret = float((mc_rets * mc_valid).sum().item()) / mc_valid_count
            mc_var_ret = float((mc_valid * (mc_rets - mc_mean_ret).pow(2)).sum().item()) / mc_valid_count
            mc_ev = 1.0 - (float(mc_diff.pow(2).sum().item()) / mc_valid_count) / max(mc_var_ret, 1e-8)
            mc_value_bias = float(mc_diff.sum().item()) / mc_valid_count
        else:
            mc_ev = float("nan")
            mc_value_bias = float("nan")

        advs_norm = (advs - advs.mean()) / (advs.std() + 1e-8)
        train_mask = 1.0 - buf_truncated.float()

        b_obs = buf_obs.reshape(-1, 38, H, W)
        b_masks = buf_masks.reshape(-1, 9, H, W)
        b_temporal = buf_temporal.reshape(-1, 2, 512)
        b_actions = buf_actions.reshape(-1)
        b_old_lps = buf_logprobs.reshape(-1)
        b_advs = advs_norm.reshape(-1)
        b_rets = rets.reshape(-1)
        b_train_mask = train_mask.reshape(-1)

        _, top_idx = torch.topk(b_advs.abs(), k=args.n_keep, largest=True, sorted=False)
        filtered_advs = b_advs[top_idx]
        filtered_adv_std = float(filtered_advs.std().item())
        filtered_adv_mean = float(filtered_advs.abs().mean().item())

        # ---- PHASE 3: Minibatch PPO + HL-Gauss Update ----
        num_mbs = args.n_keep // args.minibatch_size
        tot_losses_t = torch.zeros(num_mbs, device=device)
        pg_losses_t = torch.zeros(num_mbs, device=device)
        v_losses_t = torch.zeros(num_mbs, device=device)
        ent_losses_t = torch.zeros(num_mbs, device=device)
        approx_kls_t = torch.zeros(num_mbs, device=device)
        clipfracs_t = torch.zeros(num_mbs, device=device)
        mean_ratios_t = torch.zeros(num_mbs, device=device)
        max_ratios_t = torch.zeros(num_mbs, device=device)
        grad_norms_t = torch.zeros(num_mbs, device=device)
        epochs_used = 0

        for epoch in range(args.update_epochs):
            perm = torch.randperm(args.n_keep, device=device)
            shuffled_idx = top_idx[perm]

            for mb_i, start in enumerate(range(0, args.n_keep, args.minibatch_size)):
                mb_idx = shuffled_idx[start : start + args.minibatch_size]

                mb_obs = b_obs[mb_idx]
                mb_masks = b_masks[mb_idx]
                mb_temporal = b_temporal[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_lps = b_old_lps[mb_idx]
                mb_advs = b_advs[mb_idx]
                mb_rets = b_rets[mb_idx]
                mb_mask = b_train_mask[mb_idx]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.cuda):
                    _, new_lp, entropy, _, val_logits = agent.get_action_and_value(
                        mb_obs, mb_masks, mb_temporal, mb_actions
                    )

                    log_ratio = new_lp.to(torch.float32) - mb_old_lps
                    ratio = torch.exp(log_ratio)

                    pg1 = -mb_advs * ratio
                    pg2 = -mb_advs * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                    pg_loss_per_sample = torch.max(pg1, pg2)

                    v_loss_per_sample = agent.critic_head.compute_loss(val_logits.to(torch.float32), mb_rets)

                    total_per_sample = pg_loss_per_sample + args.vf_coef * v_loss_per_sample - current_ent_coef * entropy.to(torch.float32)
                    valid_denom = torch.clamp(mb_mask.sum(), min=1.0)
                    loss = (total_per_sample * mb_mask).sum() / valid_denom

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    tot_losses_t[mb_i] = loss.detach()
                    clipfracs_t[mb_i] = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
                    pg_losses_t[mb_i] = pg_loss_per_sample.mean()
                    v_losses_t[mb_i] = v_loss_per_sample.mean()
                    ent_losses_t[mb_i] = entropy.mean()
                    approx_kls_t[mb_i] = ((ratio - 1.0) - log_ratio).mean()
                    mean_ratios_t[mb_i] = ratio.mean()
                    max_ratios_t[mb_i] = ratio.max()
                    grad_norms_t[mb_i] = grad_norm

            epochs_used += 1
            if args.target_kl is not None and float(approx_kls_t.mean().item()) > args.target_kl:
                break

        _, act_gn_last, crit_gn_last = compute_component_grad_norms(agent)

        # ---- PHASE 4: Update EMA Deployment Weights ----
        with torch.no_grad():
            for ema_p, online_p in zip(ema_agent.parameters(), agent.parameters()):
                ema_p.data.mul_(args.ema_decay).add_(online_p.data, alpha=1.0 - args.ema_decay)

        t_ppo = time.time() - t_ppo
        elapsed = time.time() - t0
        sps = int((T * B_traj) / max(elapsed, 1e-6))

        draws_p0 = eps_p0 - wins_p0 - losses_p0
        wr = wins_p0 / max(eps_p0, 1)
        lr_stat = losses_p0 / max(eps_p0, 1)
        dr = draws_p0 / max(eps_p0, 1)
        total_steps_p0 = float(T * args.num_envs)
        mean_ep_len = total_steps_p0 / max(eps_p0, 1)
        mean_ent = float(ent_losses_t.mean().item())

        log_metrics = {
            "train/total_loss": float(tot_losses_t.mean().item()),
            "train/policy_loss": float(pg_losses_t.mean().item()),
            "train/value_loss": float(v_losses_t.mean().item()),
            "train/entropy": mean_ent,
            "train/n_actions": float(math.exp(min(mean_ent, 20.0))),
            "train/clip_fraction": float(clipfracs_t.mean().item()),
            "train/approx_kl": float(approx_kls_t.mean().item()),
            "train/mean_ratio": float(mean_ratios_t.mean().item()),
            "train/max_ratio": float(max_ratios_t.max().item()),
            "train/grad_norm": float(grad_norms_t.mean().item()),
            "train/actor_grad_norm": act_gn_last,
            "train/critic_grad_norm": crit_gn_last,
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
            "train/mean_reward": float(buf_rewards.mean().item()),
            "train/win_rate": wr,
            "train/loss_rate": lr_stat,
            "train/draw_rate": dr,
            "train/mean_owned_cities": mean_owned_cities,
            "train/sps": float(sps),
            "train/ent_coef": current_ent_coef,
            "train/lr": current_lr,
            "train/gamma": args.gamma,
        }
        for k, v in log_metrics.items():
            writer.add_scalar(k, v, it + 1)

        wall = int(time.time() - train_start)
        hh, mm, ss = wall // 3600, (wall % 3600) // 60, wall % 60
        print(
            f"[{hh:02d}:{mm:02d}:{ss:02d}] Iter {it + 1:3d}/{args.num_iters} | Loss: {log_metrics['train/total_loss']:.4f} | "
            f"PG: {log_metrics['train/policy_loss']:.4f} | "
            f"VF: {log_metrics['train/value_loss']:.4f} | "
            f"Ent: {log_metrics['train/entropy']:.3f} | "
            f"KL: {log_metrics['train/approx_kl']:.4f} | "
            f"Clip: {log_metrics['train/clip_fraction']:.2f} | "
            f"GNorm: {log_metrics['train/grad_norm']:.2f} | "
            f"EV: {explained_var:.2f} | "
            f"Reward: {log_metrics['train/mean_reward']:+.4f} | "
            f"Eps: {eps_p0:3d} | W/L/D: {wins_p0}/{losses_p0}/{draws_p0} ({wr * 100:.0f}%/{lr_stat * 100:.0f}%/{dr * 100:.0f}%) | "
            f"EpLen: {mean_ep_len:.0f} | "
            f"Cities: {mean_owned_cities:.2f} | "
            f"Epochs: {epochs_used}/{args.update_epochs} | "
            f"LR: {current_lr:.1e} | "
            f"SPS: {sps:.0f} | {elapsed:.2f}s (rollout {t_rollout:.2f}s, ppo {t_ppo:.2f}s)"
        )

        if (it + 1) % 250 == 0 or (it + 1) == args.num_iters:
            save_dir = f"checkpoints/{run_name}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(agent.state_dict(), f"{save_dir}/online_{it + 1}.pt")
            torch.save(ema_agent.state_dict(), f"{save_dir}/ema_{it + 1}.pt")
            gc.collect()

    writer.close()
"""Training configuration loaded from YAML with CLI overrides."""

from dataclasses import dataclass, fields
from typing import Optional, List
from ruamel.yaml import YAML


@dataclass
class CurriculumStage:
    """A single curriculum stage. Advances to the next stage when eval
    win-rate vs random >= win_rate_threshold of the *next* stage."""
    min_generals_distance: int
    max_generals_distance: int
    win_rate_threshold: float = 0.85  # win-rate needed to advance *to* this stage
    castle_val_min: Optional[int] = None   # None = inherit from top-level config
    castle_val_max: Optional[int] = None
    num_cities_min: Optional[int] = None
    num_cities_max: Optional[int] = None
    gamma: Optional[float] = None


@dataclass
class Config:
    # Run
    run_name: str = "default"

    # Environment
    pad_to: int = 15
    min_grid_size: int = 15
    max_grid_size: int = 15
    min_generals_distance: int = 4
    max_generals_distance: int = 8
    truncation: int = 512
    mountain_density_min: float = 0.18
    mountain_density_max: float = 0.26

    # Network
    network: str = "history_transformer"
    init_checkpoint: str = ""  # path to .eqx checkpoint to initialize network weights
    ema_checkpoint: str = ""   # path to .eqx checkpoint to initialize EMA weights (default: copy from init_checkpoint)
    ema_decay: float = 0.999

    # Transformer-specific (ignored by other networks)
    depth: int = 6
    embed_dim: int = 256
    n_head: int = 8
    ff_factor: int = 4
    patch_size: int = 1
    conv_dim: int = 256
    use_bf16: bool = False
    magnet_policy: str = "uniform"  # Uniform KL anchoring is plain entropy up to a constant.

    # Rollouts
    num_envs: int = 2048
    num_steps: int = 512
    num_iters: int = 800
    minibatch_size: int = 2048
    seed: int = 42

    # PPO
    num_epochs: int = 2
    lr: float = 1e-4
    final_lr: float = 0.0  # final LR after cosine decay (0 = no schedule)
    lr_decay_iters: int = 0  # cosine decay over N training iters (0 = no decay)
    lr_schedule: str = "linear"  # "linear" or "power_law"
    lr_power_law_numerator: float = 0.5  # numerator in: clip(num / iter^exp, min, max)
    lr_power_law_exponent: float = 1.1  # exponent in power law decay
    lr_power_law_min: float = 5e-6  # floor for power law LR
    lr_power_law_max: float = 1e-4  # ceiling for power law LR
    max_grad_norm: float = 0.267  # gradient clipping via optax.clip_by_global_norm
    gamma: float = 1.0
    gamma_end: float = 1.0  # final gamma for linear anneal (if == gamma, no anneal)
    gamma_anneal_iters: int = 0  # anneal gamma linearly over N iters (0 = no anneal)
    iteration_offset: int = 0  # shift LR/entropy/gamma schedules (for checkpoint resumption)
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef_start: float = 0.01
    ent_coef_end: float = 0.001
    ent_coef_decay_iters: int = 5000
    ent_schedule: str = "linear"  # "linear" or "power_law"
    ent_power: float = 0.3        # exponent for power_law schedule: start / (t+1)^power
    ent_coef_min: float = 0.001   # floor for power_law schedule
    target_kl: float = 0.02
    adv_top_frac: float = 0.25  # fraction of samples to keep by |advantage|

    # Value loss
    value_loss: str = "mse"    # "mse" or "ce" (HL-Gauss categorical)
    num_bins: int = 128          # number of bins for CE value head
    v_min: float = -1.0        # min value for bin range
    v_max: float = 1.0         # max value for bin range
    hl_sigma: float = 0.75      # sigma for HL-Gauss soft labels

    # Evaluation
    eval_every: int = 5
    eval_every_after: int = 0    # switch to this eval frequency on last curriculum stage (0 = no switch)
    eval_games: int = 128
    eval_ema_only: bool = False  # if True, eval/ref_eval use only EMA weights (skip current)
    ckpt_every: int = 10

    # Reference ELO eval during training
    ref_eval_every: int = 0          # 0 = disabled; run ref ELO eval every N iters
    ref_eval_games: int = 500        # games per side vs each reference
    ref_eval_dir: str = ""           # path to references folder (e.g. checkpoints/references/S)
    ref_eval_matrix: str = ""        # path to precomputed ref-ref JSON (auto: <ref_eval_dir>/ref_matrix.json)
    # Ref eval environment settings (explicit — no guessing from curriculum or ref configs)
    ref_eval_min_grid_size: Optional[int] = None
    ref_eval_max_grid_size: Optional[int] = None
    ref_eval_min_generals_distance: Optional[int] = None
    ref_eval_max_generals_distance: Optional[int] = None
    ref_eval_num_cities_min: Optional[int] = None
    ref_eval_num_cities_max: Optional[int] = None
    ref_eval_castle_val_min: Optional[int] = None
    ref_eval_castle_val_max: Optional[int] = None
    ref_eval_truncation: Optional[int] = None

    pool_size: int = 10_000
    reset_pool_every: int = 10
    save_every: int = 1000
    debug: bool = False

    # Curriculum (list of stage dicts, or None for no curriculum)
    curriculum: Optional[list] = None
    castle_val_min: int = 40
    castle_val_max: int = 51
    num_cities_min: int = 9
    num_cities_max: int = 11

    @property
    def num_actions(self):
        return 9 * self.pad_to * self.pad_to

    @property
    def curriculum_stages(self) -> Optional[List[CurriculumStage]]:
        """Parse curriculum list from YAML into CurriculumStage objects (order preserved)."""
        if not self.curriculum:
            return None
        known = {f.name for f in fields(CurriculumStage)}
        return [CurriculumStage(**{k: v for k, v in s.items() if k in known}) for s in self.curriculum]

    def __post_init__(self):
        # Coerce types — ruamel.yaml uses ScalarFloat/ScalarInt subtypes
        for f in fields(self):
            if f.name in ("curriculum",):
                continue
            val = getattr(self, f.name)
            if val is None:
                continue
            if f.type in (float, Optional[float]) and not isinstance(val, float):
                object.__setattr__(self, f.name, float(val))
            elif f.type in (int, Optional[int]) and not isinstance(val, int):
                object.__setattr__(self, f.name, int(val))

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        yaml = YAML(typ="safe")
        with open(path) as f:
            data = yaml.load(f)
        known = {f.name for f in fields(cls)}
        unknown = [k for k in data if k not in known]
        if unknown:
            print(f"WARNING: ignoring unknown config keys in {path}: {unknown}")
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

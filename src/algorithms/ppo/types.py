from dataclasses import dataclass


@dataclass
class Params:

    # Training Params
    n_epochs: int
    n_total_steps: int
    n_minibatches: int
    batch_size: int
    random_seeds: list

    # PPO Params
    lr: float
    gamma: float
    lmbda: float
    grad_clip: float
    eps_clip: float
    ent_coef: float
    val_coef: float
    std_coef: float


@dataclass
class Experiment:
    device: str = ""
    model: str = ""
    params: Params = None

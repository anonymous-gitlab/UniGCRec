from dataclasses import dataclass


@dataclass
class ModelConfig:
    sem_dim: int = 768

    collab_dim: int = 64
    raw_collab_dim: int = 256

    hidden_dim: int = 256

    num_domains: int = 5

    code_num: int = 256
    code_depth: int = 3

    e_dim: int = 32

    sem_floor: float = 0.35


DEFAULT_MODEL = ModelConfig()

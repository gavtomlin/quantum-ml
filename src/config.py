from dataclasses import dataclass 
from pathlib import Path 
import yaml

@dataclass
class DataConfig: 
    path:str
    output_dir: str

@dataclass
class SplitConfig:
    test_size: float
    random_state: int

@dataclass
class ClassicalConfig:
    max_iter: int
    n_estimators: int
    random_state: int


@dataclass
class QuantumConfig:
    n_qubits: int
    n_layers: int
    encoding: str
    n_steps: int
    step_size: float
    max_train_samples: int
    optimizer: str

@dataclass
class PipelineConfig:
    data: DataConfig
    split: SplitConfig
    classical: ClassicalConfig
    quantum: QuantumConfig

def load_config(path: str | Path = "config/pipeline.yaml") -> PipelineConfig:
    with open(path) as f: 
        raw = yaml.safe_load(f)
    return PipelineConfig(
            data=DataConfig(**raw["data"]),
            split=SplitConfig(**raw["split"]),
            classical=ClassicalConfig(**raw["classical"]),
            quantum=QuantumConfig(**raw["quantum"]),
    )
 

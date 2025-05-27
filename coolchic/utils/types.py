import itertools
import random
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Type, TypeVar

import yaml
from pydantic import BaseModel, BeforeValidator, Field, computed_field

from coolchic.enc.training.presets import TrainerPhase, Warmup, WarmupPhase
from coolchic.utils.paths import COOLCHIC_REPO_ROOT

PRESET_NAMES = Literal["c3x", "debug"]
preset_configs_dir = COOLCHIC_REPO_ROOT / "preset_cfg"
PRESET_PATHS: dict[PRESET_NAMES, Path] = {
    "c3x": preset_configs_dir / "c3x.yaml",
    "debug": preset_configs_dir / "debug.yaml",
}


class PresetConfig(BaseModel):
    preset_name: str
    warmup: Warmup
    all_phases: list[TrainerPhase]

    def model_post_init(self, __context: Any) -> None:
        if "hnet" in self.preset_name:
            # Skip the quantization check when training a hypernet.
            return
        # Check that we do quantize the model at least once during the training
        flag_quantize_model = False
        for training_phase in self.all_phases:
            if training_phase.quantize_model:
                flag_quantize_model = True

        # Ignore this assertion if there is no self.all_phases described
        assert flag_quantize_model or len(self.all_phases) == 0, (
            f"The selected preset ({self.preset_name}) does not include "
            f" a training phase with neural network quantization.\n"
            f"{self.pretty_string()}"
        )

    def _get_total_training_iterations(self) -> int:
        """Return the total number of iterations for the whole warm-up."""
        return sum([phase.max_itr for phase in self.all_phases])

    def pretty_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""
        s = f"Preset: {self.preset_name:<10}\n"
        s += "-------\n"
        s += "\nWarm-up\n"
        s += "-------\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        s += WarmupPhase._pretty_string_column_name() + "\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        for warmup_phase in self.warmup.phases:
            s += warmup_phase.pretty_string() + "\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"

        s += "\nMain training\n"
        s += "-------------\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        s += f'|{"Phase index":^14}|{TrainerPhase._pretty_string_column_name()}\n'
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        for idx, training_phase in enumerate(self.all_phases):
            s += f"|{idx:^14}|{training_phase.pretty_string()}\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"

        s += "\nMaximum number of iterations (warm-up / training / total):"
        warmup_max_itr = self.warmup._get_total_warmup_iterations()
        training_max_itr = self._get_total_training_iterations()
        total_max_itr = warmup_max_itr + training_max_itr
        s += f"{warmup_max_itr:^8} / {training_max_itr:^8} / {total_max_itr:^8}\n\n"
        return s


class EncoderConfig(BaseModel):
    intra_period: int = 0
    p_period: int = 0
    start_lr: float = 1e-2
    n_itr: int | None = None
    n_train_loops: int = 1
    # The recipe parameters are mutually exclusive.
    recipe: PresetConfig | None = None
    std_recipe_name: PRESET_NAMES | None = (
        None  # Declares a standardised preset by its name.
    )

    def model_post_init(self, __context: Any) -> None:
        # Check that at least one of the 2 recipe parameters is given.
        if not self.recipe and not self.std_recipe_name:
            raise ValueError("One of 'recipe' or 'std_recipe_name' must be provided.")
        # If std_recipe_name was provided, assign the right recipe preset.
        if self.std_recipe_name:
            if self.recipe:
                # Check that both are not given at the same time.
                raise ValueError(
                    "Only one of 'recipe' or 'std_recipe_name' must be provided, not both."
                )
            with open(PRESET_PATHS[self.std_recipe_name], "r") as stream:
                self.recipe = PresetConfig(**yaml.safe_load(stream))

        # At this point, recipe exists:
        assert self.recipe is not None, (
            "Training recipe was found to be None, which was unexpected. "
            "Check the code for possible errors."
        )

        # There is a convention in the original cool-chic repo where iterations in the first loop of the recipe have to be n_itr + 600.
        # Here we avoid that, using the number found in the params.
        if self.n_itr:
            self.recipe.all_phases[0].max_itr = self.n_itr


class DecoderConfig(BaseModel):
    config_name: str | None = Field(
        default=None,
        description="When we have more than one decoder config, we can give them distinct names.",
    )
    layers_synthesis: str = Field(
        default="40-1-linear-relu,X-1-linear-none,X-3-residual-relu,X-3-residual-none",
        description=(
            "Syntax example for the synthesis:"
            " 12-1-linear-relu,12-1-residual-relu,X-1-linear-relu,X-3-residual-none"
            "This is a 4 layers synthesis. Now the output layer (computing the final RGB"
            "values) must be specified i.e. a 12,12 should now be called a 12,12,3. Each layer"
            "is described using the following syntax:"
            "<output_dim>-<kernel_size>-<type>-<non_linearity>. "
            "<output_dim> is the number of output features. If set to X, this is replaced by the"
            "number of required output features i.e. 3 for a RGB or YUV frame."
            "<kernel_size> is the spatial dimension of the kernel. Use 1 to mimic an MLP."
            "<type> is either 'linear' for a standard conv or 'residual' for a residual"
            " block i.e. layer(x) = x + conv(x). <non_linearity> Can be'none' for no"
            " non-linearity, 'relu' for a ReLU"
        ),
    )
    arm: str = Field(
        default="24,2",
        description=(
            "<arm_context_and_layer_dimension>,<number_of_hidden_layers>"
            "First number indicates both the context size **AND** the hidden layer dimension."
            "Second number indicates the number of hidden layer(s). 0 gives a linear ARM module."
        ),
    )
    ups_k_size: int = Field(
        default=8,
        description=(
            "Upsampling kernel size for the transposed convolutions. "
            "Must be even and >= 4."
        ),
    )
    ups_preconcat_k_size: int = Field(
        default=7,
        description=(
            "Upsampling kernel size for the pre-concatenation convolutions. "
            "Must be odd."
        ),
    )
    n_ft_per_res: str = Field(
        default="1,1,1,1,1,1,1",
        description=(
            "Number of feature for each latent resolution. e.g. --n_ft_per_res=1,2,2,2,3,3,3"
            " for 7 latent grids with variable resolutions."
        ),
    )
    encoder_gain: int = Field(
        default=16,
        description=(
            "Encoder gain for the encoder. "
            "This is used to scale the latents before quantization."
        ),
    )

    @computed_field
    @property
    def dim_arm(self) -> int:
        assert len(self.arm.split(",")) == 2, (
            f"--arm format should be X,Y." f" Found {self.arm}"
        )
        return int(self.arm.split(",")[0])

    @computed_field
    @property
    def n_hidden_layers_arm(self) -> int:
        assert len(self.arm.split(",")) == 2, (
            f"--arm format should be X,Y." f" Found {self.arm}"
        )
        return int(self.arm.split(",")[1])

    @computed_field
    @property
    def parsed_layers_synthesis(self) -> list[str]:
        # Parsing the synthesis layers.
        parsed_layers_synthesis = [
            x for x in self.layers_synthesis.split(",") if x != ""
        ]
        # NOTE: We replace the X in the number of channels by the number of output channels,
        # which will always be 3, as we are working with RGB images.
        parsed_layers_synthesis = [
            lay.replace("X", str(3)) for lay in parsed_layers_synthesis
        ]
        assert parsed_layers_synthesis, (
            "Synthesis should have at least one layer, found nothing.\n"
            "Try something like 32-1-linear-relu,X-1-linear-none,"
            "X-3-residual-relu,X-3-residual-none"
        )
        return parsed_layers_synthesis

    @computed_field
    @property
    def parsed_n_ft_per_res(self) -> list[int]:
        parsed_n_ft_per_res = [int(x) for x in self.n_ft_per_res.split(",") if x != ""]
        assert set(parsed_n_ft_per_res) == {
            1
        }, f"--n_ft_per_res should only contain 1. Found {self.n_ft_per_res}"
        return parsed_n_ft_per_res


def single_element_to_list(elem: Any) -> list[Any]:
    if not isinstance(elem, list):
        return [elem]
    return elem


def get_run_uid(index: int | None = None):
    if not index:
        # if an index number is not provided, we generate a random number to avoid collisions.
        index = random.randint(100, 999)
    return f"{datetime.now().strftime('%H%M%S')}_{index:03}"  # Timestamp + number


class RunConfig(BaseModel):
    input: Path
    output: Path | None = None
    workdir: Path | None = None
    lmbda: float = 1e-3
    job_duration_min: int = -1
    enc_cfg: EncoderConfig
    dec_cfg: DecoderConfig
    disable_wandb: bool = False
    load_models: bool = True
    unique_id: str = get_run_uid()
    user_tag: str | None


class UserConfig(BaseModel):
    input: Annotated[Path | list[Path], BeforeValidator(single_element_to_list)]
    output: Path | None = None
    workdir: Path | None = None
    lmbda: Annotated[float | list[float], BeforeValidator(single_element_to_list)] = [
        1e-3
    ]
    job_duration_min: int = -1
    enc_cfg: EncoderConfig
    dec_cfg: Annotated[
        DecoderConfig | list[DecoderConfig], BeforeValidator(single_element_to_list)
    ]
    disable_wandb: bool = False
    load_models: bool = True
    mult_attributes: list[str] = ["input", "lmbda", "dec_cfg"]
    user_tag: str | None = None

    def get_run_configs(self) -> list[RunConfig]:
        configs = []
        for input, lmbda, dec_cfg in itertools.product(
            *[self.__getattribute__(attr) for attr in self.mult_attributes]
        ):  # All combinations of elements in the lists.
            cur_config = self.model_copy(deep=True)
            cur_config.input = input
            cur_config.lmbda = lmbda
            cur_config.dec_cfg = dec_cfg
            if cur_config.enc_cfg.std_recipe_name:
                # We do this because when RunConfig is built, it will look at
                # whether there is a recipe name and fetch the according preset.
                cur_config.enc_cfg.recipe = None
            cur_config = RunConfig(**cur_config.model_dump())
            cur_config.unique_id = get_run_uid(len(configs))
            configs.append(cur_config)
        return configs


class HyperNetParams(BaseModel):
    hidden_dim: int
    n_layers: int
    # Whether or not the hypernet should output bias values
    # (as well as matrices or conv kernels).
    biases: bool = True
    # Whether the hypernet should only output biases.
    only_biases: bool = False
    output_activation: str | None = "tanh"


RESNET_OPTIONS = Literal["resnet18", "resnet50", "resnet101"]


class HyperNetConfig(BaseModel):
    dec_cfg: DecoderConfig

    synthesis: HyperNetParams = HyperNetParams(hidden_dim=1024, n_layers=3)
    arm: HyperNetParams = HyperNetParams(hidden_dim=1024, n_layers=3)
    backbone_arch: RESNET_OPTIONS = "resnet18"
    double_backbone: bool = False
    n_hidden_channels: int = 64

    patch_size: tuple[int, int] = (256, 256)

    @computed_field
    @property
    def n_latents(self) -> int:
        return len(self.dec_cfg.parsed_n_ft_per_res)


class HypernetRunConfig(BaseModel):
    n_samples: int
    batch_size: int = 1
    lmbda: float = 1e-3

    recipe: PresetConfig
    unfreeze_backbone: int

    hypernet_cfg: HyperNetConfig
    workdir: Path | None = None
    model_weights: Path | None = None
    checkpoint: Path | None = None

    disable_wandb: bool = False
    unique_id: str = get_run_uid()
    user_tag: str | None


T = TypeVar("T", bound=BaseModel)


def load_config(config_path: Path, config_class: Type[T]) -> T:
    with open(config_path, "r") as stream:
        return config_class(**yaml.safe_load(stream))

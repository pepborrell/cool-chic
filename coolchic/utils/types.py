from pathlib import Path
from typing import Any, Literal
from typing_extensions import Annotated

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic.functional_validators import BeforeValidator

from enc.training.presets import TrainerPhase, Warmup, WarmupPhase
from utils.paths import COOLCHIC_REPO_ROOT

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

        # If n_itr is provided, the iterations in the first loop of the recipe have to be n_itr + 600.
        # This is a convention we take over from the original cool-chic repo.
        if self.n_itr:
            self.recipe.all_phases[0].max_itr = self.n_itr + 600


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


def single_element_to_list(elem: Any) -> list[Any]:
    if not isinstance(elem, list):
        return [elem]
    return elem


class Config(BaseModel):
    input: Path
    output: Path = Path("")
    workdir: Path | None = None
    lmbda: float = 1e-3
    job_duration_min: int = -1
    enc_cfg: EncoderConfig
    dec_cfgs: Annotated[list[DecoderConfig], BeforeValidator(single_element_to_list)]
    dec_cfg: DecoderConfig = Field(
        default=DecoderConfig(),
        description="This field should never be set at init. "
        "It is used all over the place later on, but we assign values to it only via code.",
    )
    disable_wandb: bool = False
    load_models: bool = True

    @model_validator(mode="before")
    def check_no_dec_cfg(cls, values):
        if "dec_cfg" in values:
            raise ValueError(
                "dec_cfg cannot be initialized. Provide a list of decoder configs to dec_cfgs instead."
            )
        return values

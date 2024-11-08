from pathlib import Path
import yaml
from typing import Any, Literal, override

from pydantic import BaseModel, Field, model_validator

from coolchic.enc.utils.presets import TrainerPhase, Warmup, WarmupPhase
from coolchic.utils.paths import COOLCHIC_PYTHON_ROOT

PRESET_NAMES = Literal["c3x", "debug"]
preset_configs_dir = COOLCHIC_PYTHON_ROOT / "preset_cfg"
PRESET_PATHS: dict[PRESET_NAMES, Path] = {
    "c3x": preset_configs_dir / "c3x.yaml",
    "debug": preset_configs_dir / "debug.yaml",
}


class PresetConfig(BaseModel):
    preset_name: str
    warmup: Warmup
    all_phases: list[TrainerPhase]

    @override
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
    n_itr: int = int(1e4)
    n_train_loops: int = 1
    # The recipe parameters are mutually exclusive.
    recipe: PresetConfig | None
    std_recipe_name: PRESET_NAMES | None  # Declares a standardised preset by its name.

    def model_post_init(self, __context: Any) -> None:
        # If std_recipe_name was provided, assign the right recipe preset.
        if self.std_recipe_name:
            if self.recipe:
                # Check that both are not given at the same time.
                raise ValueError(
                    "Only one of 'recipe' or 'std_recipe_name' must be provided, not both."
                )
            with open(PRESET_PATHS[self.std_recipe_name], "r") as stream:
                self.recipe = PresetConfig(**yaml.safe_load(stream))

    @model_validator(mode="after")
    def check_mutually_exclusive_recipes(self):
        """Checks that only one of recipe and std_recipe are provided."""
        # Check if both fields are provided
        if self.recipe and self.std_recipe_name:
            raise ValueError(
                "Only one of 'recipe' or 'std_recipe_name' must be provided, not both."
            )
        # Check if neither field is provided
        if not self.recipe and not self.std_recipe_name:
            raise ValueError("One of 'recipe' or 'std_recipe_name' must be provided.")
        return self


class DecoderConfig(BaseModel):
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
    n_ft_per_res: str = Field(
        default="1,1,1,1,1,1,1",
        description=(
            "Number of feature for each latent resolution. e.g. --n_ft_per_res=1,2,2,2,3,3,3"
            " for 7 latent grids with variable resolutions."
        ),
    )
    upsampling_kernel_size: int = 8
    static_upsampling_kernel: bool = False


class Config(BaseModel):
    input: Path
    output: Path = Path("")
    workdir: Path | None = None
    lmbda: float = 1e-3
    job_duration_min: int = -1
    enc_cfg: EncoderConfig
    dec_cfg: DecoderConfig
    disable_wandb: bool = False
    load_models: bool = True

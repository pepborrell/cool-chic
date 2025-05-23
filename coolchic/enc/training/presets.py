# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

"""Gather the different encoding presets here."""

import typing
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

from pydantic import BaseModel, Field

from coolchic.enc.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
)

MODULE_TO_OPTIMIZE = Literal["all", "arm", "upsampling", "synthesis", "latent"]


class TrainerPhase(BaseModel):
    """Dataclass representing one phase of an encoding preset.

    Args:
        lr (float): Initial learning rate of the phase. Can vary if
            ``schedule_lr`` is True. Defaults to 0.01.
        max_itr (int): Maximum number of iterations for the phase. The actual
            number of iterations can be made smaller through the patience
            mechanism. Defaults to 10000.
        freq_valid: Check (and print) the performance
            each ``frequency_validation`` iterations. This drives the patience
            mechanism. Defaults to 100.
        patience: After ``patience`` iterations without any
            improvement to the results, exit the training. Patience is disabled
            by setting ``patience = max_iterations``. If patience is used alongside
            cosine_scheduling_lr, then it does not end the training. Instead,
            we simply reload the best model so far once we reach the patience,
            and the training continue. Defaults to 1000.
        quantize_model (bool): If ``True``, quantize the neural networks
            parameters at the end of the training phase. Defaults to ``False``.
        schedule_lr (bool): If ``True``, the learning rate is no longer
            constant. instead, it varies with a cosine scheduling, as suggested
            in  `C3: High-performance and low-complexity neural compression from
            a single image or video, Kim et al.
            <https://arxiv.org/abs/2312.02753>`_. Defaults to False.
        softround_temperature (Tuple[float, float]). Start, end temperature of
            the :doc:`softround function <../component/core/quantizer>`. It is
            used in the forward / backward if ``quantizer_type`` is set to
            ``"softround"`` or ``"softround_alone"``. It is also used in the
            backward pass if ``quantizer_type`` is set to ``"ste"``.
            The softround temperature is linearly scheduled
            during the training. At iteration n° 0 it is equal to
            ``softround_temperature[0]`` while at iteration n° ``max_itr`` it is
            equal to ``softround_temperature[1]``. Note that the patience might
            interrupt the training before it reaches this last value.
            Defaults to (0.3, 0.3).
        noise_parameter (Tuple[float, float]): The random noise temperature is
            linearly scheduled during the training. At iteration n° 0 it is equal
            to ``noise_parameter[0]`` while at iteration n° ``max_itr`` it is equal
            to ``noise_parameter[1]``. Note that the patience might interrupt
            the training before it reaches this last value. Defaults to (2.0,
            1.0).
        quantizer_noise_type (POSSIBLE_QUANTIZATION_NOISE_TYPE): The random noise
            used by the quantizer. More information available in
            :doc:`encoder/component/core/quantizer.py <../component/core/quantizer>`.
            Defaults to ``"kumaraswamy"``.
        quantizer_type (POSSIBLE_QUANTIZER_TYPE): What quantizer to
            use during training. See
            :doc:`encoder/component/core/quantizer.py <../component/core/quantizer>`
            for more information. Defaults to ``"softround"``.
        optimized_module (List[MODULE_TO_OPTIMIZE]): List of modules to be
            optimized. Most often you'd want to use ``optimized_module = ['all']``.
            Defaults to ``['all']``.
    """

    lr: float = 1e-2
    max_itr: int = 5000
    freq_valid: int = 100
    patience: int = 10000
    checkpointing_freq: int = 10000
    gradient_accumulation: int = 1
    quantize_model: bool = False
    schedule_lr: bool = False
    end_lr: float | None = None
    softround_temperature: Tuple[float, float] = (0.3, 0.3)
    noise_parameter: Tuple[float, float] = (1.0, 1.0)
    quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy"
    quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround"
    optimized_module: List[MODULE_TO_OPTIMIZE] = Field(default_factory=lambda: ["all"])

    def __post_init__(self):
        # If all is present in the list of modules to be optimized, alongside something else,
        # it overrides everything, leaving the list of modules to be optimized to just ['all'].
        if "all" in self.optimized_module:
            self.optimized_module == ["all"]

        # Some checks about quantization options mismatch. They are done here
        # to avoid doing it each time we do a forward pass inside the quantize
        # function. Additionally, torch.compile messes up the assertion in the
        # quantize function anyway.
        assert self.quantizer_noise_type in typing.get_args(
            POSSIBLE_QUANTIZATION_NOISE_TYPE
        ), (
            f"quantizer_noise_type must be in {POSSIBLE_QUANTIZATION_NOISE_TYPE}"
            f" found {self.quantizer_noise_type}"
        )

        assert self.quantizer_type in typing.get_args(POSSIBLE_QUANTIZER_TYPE), (
            f"quantizer_type must be in {POSSIBLE_QUANTIZER_TYPE}"
            f" found {self.quantizer_type}"
        )

        # If we use only the softround **alone**, or hardround we do not need
        # any noise addition. Otherwise, we need a type of noise, i.e. either
        # kumaraswamy or gaussian noise.
        if self.quantizer_type in ["softround_alone", "hardround", "ste", "none"]:
            assert self.quantizer_noise_type == "none", (
                f"Using quantizer type {self.quantizer_type} does not require"
                "to have any random noise.\n Switching the "
                f"quantizer_noise_type from {self.quantizer_noise_type} to none."
            )
        else:
            assert self.quantizer_noise_type != "none", (
                "Using quantizer_noise_type = 'none' is only possible with "
                "quantizer_type = 'softround_alone', 'ste' or 'hardround'.\n"
                f"Trying to use {self.quantizer_type} quantizer which do require "
                "some kind of random noise such as 'gaussian' or 'kumaraswamy'."
            )

    def pretty_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""

        s = f'{f"{self.lr:1.2e}":^{14}}|'
        s += f"{self.max_itr:^{9}}|"
        s += f"{self.patience:^{16}}|"
        s += f"{self.freq_valid:^{13}}|"
        s += f"{self.quantize_model:^{13}}|"
        s += f"{self.schedule_lr:^{13}}|"

        softround_str = ", ".join([f"{x:1.1e}" for x in self.softround_temperature])
        s += f'{f"{softround_str}":^{18}}|'

        noise_str = ", ".join([f"{x:1.2f}" for x in self.noise_parameter])
        s += f'{f"{noise_str}":^{14}}|'
        return s

    @staticmethod
    def _pretty_string_column_name() -> str:
        """Return the name of the column aligned with the pretty_string function"""
        s = f'{"Learn rate":^{14}}|'
        s += f'{"Max itr":^{9}}|'
        s += f'{"Patience [itr]":^{16}}|'
        s += f'{"Valid [itr]":^{13}}|'
        s += f'{"Quantize NN":^{13}}|'
        s += f'{"Schedule lr":^{13}}|'
        s += f'{"Softround Temp":^{18}}|'
        s += f'{"Noise":^{14}}|'
        return s

    @staticmethod
    def _vertical_line_array() -> str:
        """Return a string made of "-" and "+" matching the columns
        of the print detailed above"""
        s = "-" * 14 + "+"
        s += "-" * 9 + "+"
        s += "-" * 16 + "+"
        s += "-" * 13 + "+"
        s += "-" * 13 + "+"
        s += "-" * 13 + "+"
        s += "-" * 18 + "+"
        s += "-" * 14 + "+"
        return s


class WarmupPhase(BaseModel):
    """Describe one phase of the :doc:`warm-up <../training/warmup>`. At the
    beginning of each warm-up phase, we start by keeping the best ``candidates``
    systems. We then perform a short training, and we go to the next phase.

    Args:
        candidates (int): How many candidates are kept at the beginning of the phase.
        training_phase (TrainerPhase): Describe how the candidates are trained.
    """

    candidates: int  # Keep the first <candidates> best systems at the beginning of this warmup phase
    training_phase: TrainerPhase

    def pretty_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""
        s = f"|{self.candidates:^{14}}|"
        s += f"{self.training_phase.pretty_string()}"
        return s

    @staticmethod
    def _pretty_string_column_name() -> str:
        """Return the name of the column aligned with the pretty_string function"""
        s = f'|{"Candidates":^{14}}|'
        s += f"{TrainerPhase._pretty_string_column_name()}"
        return s


class Warmup(BaseModel):
    """A :doc:`warm-up <../training/warmup>` is composed of different phases
    where the worse candidates are successively eliminated.

    Args:
        phase (List[WarmupPhase]): The successive phases of the Warmup.
            Defaults to ``[]``.
    """

    phases: List[WarmupPhase] = Field(default_factory=lambda: [])

    def _get_total_warmup_iterations(self) -> int:
        """Return the total number of iterations for the whole warm-up."""
        return sum(
            [phase.candidates * phase.training_phase.max_itr for phase in self.phases]
        )


@dataclass
class Preset:
    """Dummy parent (abstract) class of all encoder presets. An actual preset
    should inherit from this class.

    Encoding preset defines how we encode each frame. They are similar to
    conventional codecs presets *e.g* x264 ``--slow`` preset offers better
    compression performance at the expense of a longer encoding.

    Here a preset defines two things: how the :doc:`warm-up <../training/warmup>`
    is done, and how the subsequent :doc:`training <../training/train>` is done.

    Args:
        preset_name (str): Name of the preset.
        all_phases (List[TrainerPhase]): The successive (post warm-up) training
            phase. Defaults to ``[]``.
        warmup (Warmup): The warm-up parameters. Defaults to ``Warmup()``.
    """

    preset_name: str
    # Dummy empty training phases and warm-up
    all_phases: List[TrainerPhase] = field(
        default_factory=lambda: []
    )  # All the post-warm-up training phases
    warmup: Warmup = field(default_factory=lambda: Warmup())  # All the warm-up phases

    def __post_init__(self):
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


class PresetC3x(Preset):
    def __init__(self, start_lr: float = 1e-2, n_itr_per_phase: int = 100000):
        super().__init__(preset_name="c3x")
        # 1st stage: with soft round and quantization noise
        self.all_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=start_lr,
                max_itr=n_itr_per_phase,
                patience=5000,
                optimized_module=["all"],
                schedule_lr=True,
                quantizer_type="softround",
                quantizer_noise_type="gaussian",
                softround_temperature=(0.3, 0.1),
                noise_parameter=(0.25, 0.1),
            ),
            # Stage with STE then network quantization
            TrainerPhase(
                lr=1.0e-4,
                max_itr=1500,
                patience=1500,
                optimized_module=["all"],
                schedule_lr=True,
                quantizer_type="ste",
                quantizer_noise_type="none",
                # This is only used to parameterize the backward of the quantization
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(1.0, 1.0),  # not used since quantizer type is "ste"
                quantize_model=True,  # ! This is an important parameter
            ),
            # Re-tune the latent
            TrainerPhase(
                lr=1.0e-4,
                max_itr=1000,
                patience=50,
                quantizer_type="ste",
                quantizer_noise_type="none",
                optimized_module=["latent"],  # ! Only fine tune the latent
                freq_valid=10,
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(1.0, 1.0),  # not used since quantizer type is "ste"
            ),
        ]

        self.warmup = Warmup(
            phases=[
                WarmupPhase(
                    candidates=5,
                    training_phase=TrainerPhase(
                        lr=start_lr,
                        max_itr=400,
                        freq_valid=400,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(2.0, 2.0),
                        quantizer_noise_type="kumaraswamy",
                        quantizer_type="softround",
                        optimized_module=["all"],
                    ),
                ),
                WarmupPhase(
                    candidates=2,
                    training_phase=TrainerPhase(
                        lr=start_lr,
                        max_itr=400,
                        freq_valid=400,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(2.0, 2.0),
                        quantizer_noise_type="kumaraswamy",
                        quantizer_type="softround",
                        optimized_module=["all"],
                    ),
                ),
            ]
        )


class PresetDebug(Preset):
    """Very fast training schedule, should only be used to ensure that the code works properly!"""

    def __init__(self, start_lr: float = 1e-2, n_itr_per_phase: int = 100000):
        super().__init__(preset_name="debug")
        self.all_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=start_lr,
                max_itr=50,
                patience=100000,
                optimized_module=["all"],
                schedule_lr=True,
                quantizer_type="softround",
                quantizer_noise_type="gaussian",
                softround_temperature=(0.3, 0.1),
                noise_parameter=(0.25, 0.1),
            )
        ]

        self.all_phases.append(
            TrainerPhase(
                lr=1e-4,
                max_itr=10,
                patience=10,
                optimized_module=["all"],
                quantizer_type="ste",
                quantizer_noise_type="none",
                quantize_model=True,
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(1.0, 1.0),  # not used since quantizer type is "ste"
            )
        )

        self.all_phases.append(
            TrainerPhase(
                lr=1e-4,
                max_itr=10,
                patience=50,
                optimized_module=["latent"],
                freq_valid=5,
                quantizer_type="ste",
                quantizer_noise_type="none",
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(1.0, 1.0),  # not used since quantizer type is "ste"
            )
        )

        self.warmup = Warmup(
            [
                WarmupPhase(candidates=3, training_phase=TrainerPhase(max_itr=10)),
                WarmupPhase(candidates=2, training_phase=TrainerPhase(max_itr=10)),
            ]
        )


class PresetMeasureSpeed(Preset):
    def __init__(self, start_lr: float = 1e-2, n_itr_per_phase: int = 100000):
        super().__init__(preset_name="c3x")

        # Single stage model with the shortest warm-up ever!
        self.all_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=start_lr,
                max_itr=n_itr_per_phase,
                patience=5000,
                optimized_module=["all"],
                schedule_lr=True,
                quantizer_type="softround",
                quantizer_noise_type="gaussian",
                softround_temperature=(0.3, 0.1),
                noise_parameter=(0.25, 0.1),
                quantize_model=True,  # ! This is an important parameter
            ),
        ]

        self.warmup = Warmup(
            [
                WarmupPhase(
                    candidates=1,
                    training_phase=TrainerPhase(
                        lr=start_lr,
                        max_itr=1,
                        freq_valid=1,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(2.0, 2.0),
                        quantizer_noise_type="kumaraswamy",
                        quantizer_type="softround",
                        optimized_module=["all"],
                    ),
                )
            ]
        )


AVAILABLE_PRESETS: Dict[str, type[Preset]] = {
    "c3x": PresetC3x,
    "debug": PresetDebug,
    "measure_speed": PresetMeasureSpeed,
}

from typing import OrderedDict

import torch

from coolchic.enc.component.coolchic import (
    CoolChicEncoder,
    CoolChicEncoderOutput,
    CoolChicEncoderParameter,
)


class CCMetaLearningEncoder(CoolChicEncoder):
    """Wrapper around the CoolChic Encoder for one frame,
    adapted so that its API helps implement MAML easily.
    """

    def __init__(self, param: CoolChicEncoderParameter):
        self.param = param
        super().__init__(param)

    def forward(  # pyright: ignore (to ignore the different signature on override)
        self,
        weights: OrderedDict[str, torch.Tensor],
        training: bool,
        num_step: int,
    ) -> CoolChicEncoderOutput:
        assert any(
            weight.requires_grad for weight in weights.values()
        ), "At least one weight should require grad to perform MAML."
        # Override params.
        self.set_param(weights)
        return super().forward(
            quantizer_noise_type="kumaraswamy",
            quantizer_type="softround",
            soft_round_temperature=0.3,
            noise_parameter=1.0,
            AC_MAX_VAL=-1,
            flag_additional_outputs=False,
        )

    def zero_grad(self, set_to_none: bool = True, params: dict | None = None):
        if params is None:
            super().zero_grad(set_to_none)
        else:
            for name, param in params.items():
                if (
                    param.requires_grad
                    and param.grad is not None
                    and torch.sum(param.grad) > 0
                ):
                    print(param.grad)
                    param.grad.zero_()
                    params[name].grad = None

    def restore_backup_stats(self) -> None:
        pass

    def get_flops(self) -> None:
        """Overriding this method is necessary. Otherwise the one from the original
        encoder gets called and causes issues (because forward signature changed).
        """
        self.total_flops = -1

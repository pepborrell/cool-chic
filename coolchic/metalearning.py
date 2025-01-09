import torch
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader

from coolchic.enc.component.coolchic import (
    CoolChicEncoderParameter,
)
from coolchic.enc.training.loss import loss_function
from coolchic.enc.utils.parsecli import get_coolchic_param_from_args
from coolchic.metalearning.data import PATCH_SIZE, OpenImagesDataset
from coolchic.metalearning.inner_loop import LSLRGradientDescentLearningRule
from coolchic.metalearning.model import CCMetaLearningEncoder
from coolchic.utils.types import DecoderConfig

# CoolChic-specific configuration.
hop_config = DecoderConfig(
    arm="16,2",
    layers_synthesis="48-1-linear-relu,X-1-linear-none,X-3-residual-relu,X-3-residual-none",
    n_ft_per_res="1,1,1,1,1,1,1",
    ups_k_size=8,
    ups_preconcat_k_size=7,
)
hop_params = CoolChicEncoderParameter(**get_coolchic_param_from_args(hop_config))
hop_params.set_image_size(PATCH_SIZE)
# Taken from coolchic/enc/component/video.py
# The number of output channels is changed depending on frame type.
# All images are I frames.
# Change the number of channels for the synthesis output
hop_params.layers_synthesis = [
    lay.replace("X", str(3)) for lay in hop_params.layers_synthesis
]
lmbda = 1e-3


class Args(BaseModel):
    total_epochs: int = 100
    use_multi_step_loss_optimization: bool = True
    multi_step_loss_num_epochs: int = 10
    number_of_training_steps_per_iter: int = 5
    first_order_to_second_order_epoch: int = -1
    learnable_per_layer_per_step_inner_loop_learning_rate: bool = True
    min_learning_rate: float = 0.00001
    task_learning_rate: float = 0.1
    meta_learning_rate: float = 0.001
    second_order: bool = True


class MAMLTrainer(nn.Module):
    def __init__(self, im_shape, device, args: Args):
        super().__init__()

        self.im_shape = im_shape
        self.device = device
        self.args = args

        self.lmbda = lmbda

        self.task_learning_rate = args.task_learning_rate
        self.encoder = CCMetaLearningEncoder(param=hop_params)
        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(
            device=self.device,
            init_learning_rate=self.task_learning_rate,
            total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
            use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate,
        )

        self.inner_loop_optimizer.initialise(
            names_weights_dict={k: v for k, v in self.encoder.named_parameters()}
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=args.meta_learning_rate, amsgrad=False
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.args.total_epochs,
            eta_min=self.args.min_learning_rate,
        )
        self.current_epoch = 0
        self.to(self.device)

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        All steps (but the last) are weighted equal, with decreasing importance as the optimization progresses.
        The last step is given the most importance, making sure that the sum of all weights adds up to one.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        n_steps = self.args.number_of_training_steps_per_iter
        decay_rate = 1.0 / n_steps / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / n_steps

        # Importance weight decays as optimization progresses.
        current_epoch_weight = max(
            1 / n_steps - self.current_epoch * decay_rate,
            min_value_for_non_final_losses,
        )
        loss_weights = torch.full((n_steps,), fill_value=current_epoch_weight)
        # Adapt last step's weight to ensure the sum of all weights is 1.
        loss_weights[-1] = 1 - torch.sum(loss_weights[:-1])

        loss_weights = loss_weights.to(device=self.device)
        return loss_weights

    def apply_inner_loop_update(
        self, loss, names_weights_copy, use_second_order, current_step_idx
    ):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        self.encoder.zero_grad(params=names_weights_copy)

        grads = torch.autograd.grad(
            loss,
            names_weights_copy.values(),
            create_graph=use_second_order,
            allow_unused=True,
        )
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        # names_weights_copy = {
        #     key: value[0] for key, value in names_weights_copy.items()
        # }

        print("HOLAAAAA")
        for key, grad in names_grads_copy.items():
            if grad is None:
                print(f"{names_weights_copy[key]=}")
                print(f"{names_weights_copy[key].is_leaf=}")
                print(f"{names_weights_copy[key].grad_fn=}")
                print(f"{names_weights_copy[key].grad_fn.next_functions=}")
                print("Grads not found for inner loop parameter", key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

        names_weights_copy = self.inner_loop_optimizer.update_params(
            names_weights_dict=names_weights_copy,
            names_grads_wrt_params_dict=names_grads_copy,
            num_step=current_step_idx,
        )

        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses):
        losses = {"loss": torch.mean(torch.stack(total_losses))}
        return losses

    def forward(
        self,
        data_batch,
        epoch,
        use_second_order,
        use_multi_step_loss_optimization,
        num_steps,
        training_phase,
    ):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """

        total_losses = []
        per_task_target_preds = [[] for _ in range(len(data_batch))]
        per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
        self.encoder.zero_grad()
        for patch_id, patch in enumerate(data_batch):
            # New patch means we train from scratch again.
            # This means latents have to be reset.
            self.encoder.initialize_latent_grids()
            task_losses = []
            # names_weights_copy = self.get_inner_loop_parameter_dict(
            #     self.encoder.named_parameters()
            # )
            names_weights_copy = {k: v for k, v in self.encoder.named_parameters()}

            for num_step in range(num_steps):
                support_loss, support_preds = self.net_forward(
                    x=patch,
                    weights=names_weights_copy,
                    training=True,
                    num_step=num_step,
                )

                names_weights_copy = self.apply_inner_loop_update(
                    loss=support_loss,
                    names_weights_copy=names_weights_copy,
                    use_second_order=use_second_order,
                    current_step_idx=num_step,
                )

                if (
                    use_multi_step_loss_optimization
                    and training_phase
                    and epoch < self.args.multi_step_loss_num_epochs
                ):
                    target_loss, target_preds = self.net_forward(
                        x=patch,
                        weights=names_weights_copy,
                        training=True,
                        num_step=num_step,
                    )

                    task_losses.append(
                        per_step_loss_importance_vectors[num_step] * target_loss
                    )
                elif num_step == (self.args.number_of_training_steps_per_iter - 1):
                    target_loss, target_preds = self.net_forward(
                        x=patch,
                        weights=names_weights_copy,
                        training=True,
                        num_step=num_step,
                    )
                    task_losses.append(target_loss)

            per_task_target_preds[patch_id] = target_preds.detach().cpu().numpy()  # pyright: ignore (target_preds will always be bound)

            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)

            if not training_phase:
                self.encoder.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses)

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses[f"loss_importance_vector_{idx}"] = item.detach().cpu()

        return losses, per_task_target_preds

    def net_forward(self, x, weights, training, num_step):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary.
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape c, h, w
        :param weights: A dictionary containing the weights to pass to the network.
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """
        result = self.encoder.forward(
            weights=weights,
            training=training,
            num_step=num_step,
        )

        pred = result.raw_out
        rate = result.rate
        loss = self.loss_fn(pred, rate, target=x)

        return loss, pred

    def loss_fn(
        self, pred: torch.Tensor, latent_rate: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        loss_output = loss_function(
            pred,
            latent_rate,
            target,
            lmbda=self.lmbda,
            rate_mlp_bit=0.0,  # MLP rates not relevant for training.
            compute_logs=False,
        )
        assert isinstance(
            loss_output.loss, torch.Tensor
        ), f"Expected loss to be computed as a Tensor. Got {type(loss_output.loss)} instead."
        return loss_output.loss

    def train_forward_prop(self, data_batch: torch.Tensor, epoch: int):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(
            data_batch=data_batch,
            epoch=epoch,
            use_second_order=self.args.second_order
            and epoch > self.args.first_order_to_second_order_epoch,
            use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
            num_steps=self.args.number_of_training_steps_per_iter,
            training_phase=True,
        )
        return losses, per_task_target_preds

    def meta_update(self, loss: torch.Tensor):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run_train_iter(self, data_batch: torch.Tensor, epoch: int):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        losses, per_task_target_preds = self.train_forward_prop(
            data_batch=data_batch, epoch=epoch
        )

        self.meta_update(loss=losses["loss"])
        losses["learning_rate"] = self.scheduler.get_lr()[0]  # pyright: ignore (scheduler.get_lr() is not a tensor but we don't care)
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds


def train():
    training_data = OpenImagesDataset(n_images=10)
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = Args()

    trainer = MAMLTrainer(training_data[0].shape[1:], device, args)
    for epoch in range(args.total_epochs):
        for img in train_dataloader:
            losses, _ = trainer.run_train_iter(img, epoch)
            print(losses)


if __name__ == "__main__":
    train()

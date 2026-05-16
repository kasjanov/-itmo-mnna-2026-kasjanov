import math

import torch
import lightning.pytorch as pl

from src.models.gpt import (
    GPTConfig,
    GPTLikeModel,
    compute_packed_lm_loss,
)


class LitGPTLanguageModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters(config)

        model_config = GPTConfig(**config["model"])
        self.model = GPTLikeModel(model_config)

        self.learning_rate = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]
        self.betas = tuple(config["training"]["betas"])
        self.warmup_steps = config["training"]["warmup_steps"]
        self.min_lr_ratio = config["training"]["min_lr_ratio"]

    def forward(self, input_ids, packed_mask):
        return self.model(
            input_ids=input_ids,
            packed_mask=packed_mask,
        )

    def shared_step(self, batch, stage: str):
        input_ids = batch["input_ids"]
        packed_mask = batch["packed_mask"]

        logits = self(
            input_ids=input_ids,
            packed_mask=packed_mask,
        )

        loss, loss_mask = compute_packed_lm_loss(
            logits=logits,
            input_ids=input_ids,
            packed_mask=packed_mask,
        )

        perplexity = torch.exp(loss.detach().clamp(max=20))

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=input_ids.size(0),
        )

        self.log(
            f"{stage}/perplexity",
            perplexity,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=input_ids.size(0),
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step + 1) / float(max(1, self.warmup_steps))

            progress = float(current_step - self.warmup_steps) / float(
                max(1, total_steps - self.warmup_steps)
            )

            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]

        self.log(
            "train/lr",
            current_lr,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )
	
    # Логирование глобальной и локальных норм градиентов.
    # Если активен ClearML, эти значения также попадут в ClearML,
    # так как ClearML автоматически перехватывает TensorBoard logs.
    def on_before_optimizer_step(self, optimizer):
        total_norm_sq = 0.0

        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm_sq += param_norm.item() ** 2

        total_norm = total_norm_sq ** 0.5

        self.log(
            "grad_norm/global",
            total_norm,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )

        for i, layer in enumerate(self.model.layers):
            layer_norm_sq = 0.0

            for p in layer.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    layer_norm_sq += param_norm.item() ** 2

            layer_norm = layer_norm_sq ** 0.5

            self.log(
                f"grad_norm/layer_{i}",
                layer_norm,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
import torch
import lightning as L

from x_clip import CLIP
from lightly.utils.benchmarking import OnlineLinearClassifier


class MultimodalCollapse(L.LightningModule):
    def __init__(
        self,
        image_encoder,
        image_preprocessing,
        text_encoder,
        text_preprocessing,
        hidden_dim_image, 
        hidden_dim_text,
        projected_dim,
        freeze_image_encoder=False,
        n_classes=10, 
        learning_rate=1e-3
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.image_preprocessing = image_preprocessing
        self.text_encoder = text_encoder
        self.text_preprocessing = text_preprocessing
        self.hidden_dim_img = hidden_dim_image
        self.hidden_dim_text = hidden_dim_text
        self.projected_dim = projected_dim
        self.learning_rate = learning_rate
        self.freeze_image_encoder = freeze_image_encoder

        self.clip = CLIP(
            image_encoder=self.image_encoder,
            text_encoder=self.text_encoder,
            dim_image=hidden_dim_image,
            dim_text=hidden_dim_text,
            dim_latent=projected_dim,
            extra_latent_projection=True,
            multiview_loss_weight=0.1 # weight multiview contrastive loss by 0.1
        )

        self.image_probe = OnlineLinearClassifier(
            feature_dim=hidden_dim_image,
            num_classes=n_classes,
        )
        self.text_probe = OnlineLinearClassifier(
            feature_dim=hidden_dim_text,
            num_classes=n_classes,
        )

    def forward(self, x_image, x_text, **kwargs):
        x_image = self.image_preprocessing(x_image)
        x_text = self.text_preprocessing(x_text)
        return self.clip(x_text, x_image, **kwargs)

    def training_step(self, batch, batch_idx):
        self.clip.training = True
        x_image, (x_text, y) = batch
        loss = self.forward(x_image, x_text, return_loss=True, freeze_image_encoder=self.freeze_image_encoder)
        self.log("clip_train_loss", loss)

        loss_image_probe, metrics_image_probe = self.image_probe.training_step(
            (self.image_encoder(self.image_preprocessing(x_image)).mean(dim=1), y), batch_idx=batch_idx
        )
        metrics_image_probe = {f"image_{k}": v for k, v in metrics_image_probe.items()}

        
        loss_text_probe, metrics_text_probe = self.text_probe.training_step(
            (self.text_encoder(self.text_preprocessing(x_text)).mean(dim=1), y),
            batch_idx=batch_idx
        )
        metrics_text_probe = {f"text_{k}": v for k, v in metrics_text_probe.items()}

        self.log_dict(metrics_image_probe)
        self.log_dict(metrics_text_probe)

        return loss + loss_image_probe + loss_text_probe

    def validation_step(self, batch, batch_idx):
        self.clip.training = True
        x_image, (x_text, y) = batch
        loss = self.forward(x_image, x_text, return_loss=True)
        self.log("clip_val_loss", loss)

        loss_image_probe, metrics_image_probe = self.image_probe.validation_step(
            (self.image_encoder(self.image_preprocessing(x_image)).mean(dim=1), y), batch_idx=batch_idx
        )
        metrics_image_probe = {f"image_{k}": v for k, v in metrics_image_probe.items()}

        loss_text_probe, metrics_text_probe = self.text_probe.validation_step(
            (self.text_encoder(self.text_preprocessing(x_text)).mean(dim=1), y),
            batch_idx=batch_idx
        )
        metrics_text_probe = {f"text_{k}": v for k, v in metrics_text_probe.items()}

        self.log_dict(metrics_image_probe)
        self.log_dict(metrics_text_probe)

        return loss + loss_image_probe + loss_text_probe

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
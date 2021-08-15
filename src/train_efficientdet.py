# get_ipython().system('pip install git+https://github.com/alexhock/object-detection-metrics')
import logging
import os
from typing import List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ensemble_boxes import ensemble_boxes_wbf
# from fastcore.basics import patch
from fastcore.dispatch import typedispatch
from hydra.core.hydra_config import HydraConfig
# from matplotlib import patches
from objdetecteval.metrics.coco_metrics import get_coco_stats
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.core.decorators import auto_move_data

from src.detection.data import (EfficientDetDataModule, draw_pascal_voc_bboxes,
                                get_train_transforms, get_valid_transforms)
from src.models.detection.efficientdet.efficientdet import create_model
from src.utils.common import (get_callback, load_obj, save_model,
                              seed_everything)


# ## Define the training loop
def run_wbf(predictions, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = [(prediction["boxes"] / image_size).tolist()]
        scores = [prediction["scores"].tolist()]
        labels = [prediction["classes"].tolist()]

        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        boxes = boxes * (image_size - 1)
        bboxes.append(boxes.tolist())
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())

    return bboxes, confidences, class_labels


class EfficientDetModel(LightningModule):
    def __init__(
        self,
        cfg,
        fold,
        num_classes=1,
        img_size=512,
        prediction_confidence_threshold=0.2,
        learning_rate=0.0002,
        wbf_iou_threshold=0.44,
        inference_transforms=get_valid_transforms(target_img_size=512),
        model_architecture='tf_efficientnetv2_l',
    ):
        super().__init__()
        self.cfg = cfg
        self.img_size = img_size
        self.model = create_model(
            num_classes, img_size, architecture=model_architecture
        )
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        self.inference_tfms = inference_transforms
        self.fold = fold
        self.hydra_cwd = HydraConfig.get().run.dir + '/weights'
        self.weight_dir = hydra.utils.to_absolute_path(self.hydra_cwd)
        self.best_loss = 10**5
        self.best_score = 0.0


    @auto_move_data
    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        # return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        optimizer = load_obj(self.cfg.optimizer.class_name)(self.model.parameters(), **self.cfg.optimizer.params)
        scheduler = load_obj(self.cfg.scheduler.class_name)(optimizer, **self.cfg.scheduler.params)

        return [optimizer], [{'scheduler': scheduler,
                              'interval': self.cfg.scheduler.step,
                              'monitor': self.cfg.scheduler.monitor}]


    def training_step(self, batch, batch_idx):
        images, annotations, _, image_ids = batch

        losses = self.model(images, annotations)

        logging_losses = {
            "class_loss": losses["class_loss"].detach(),
            "box_loss": losses["box_loss"].detach(),
        }

        self.log("train_loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log(
            "train_class_loss", losses["class_loss"], on_step=True, on_epoch=True, prog_bar=True,
            logger=True
        )
        self.log("train_box_loss", losses["box_loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)

        return losses['loss']


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        outputs = self.model(images, annotations)

        detections = outputs["detections"]

        batch_predictions = {
            "predictions": detections,
            "targets": targets,
            "image_ids": image_ids,
        }

        logging_losses = {
            "class_loss": outputs["class_loss"].detach(),
            "box_loss": outputs["box_loss"].detach(),
        }

        self.log("valid_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        self.log(
            "valid_class_loss", logging_losses["class_loss"], on_step=True, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log("valid_box_loss", logging_losses["box_loss"], on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        return {'valid_loss': outputs["loss"], 'batch_predictions': batch_predictions}


    def aggregate_prediction_outputs(self, outputs):        
        # detections = torch.cat(
        #     [output["batch_predictions"]["predictions"] for output in outputs]
        # )

        detections = torch.stack(
            [output["batch_predictions"]["predictions"] for output in outputs]
        )
        print(detections)

        image_ids = []
        targets = []
        for output in outputs:
            batch_predictions = output["batch_predictions"]
            image_ids.extend(batch_predictions["image_ids"])
            targets.extend(batch_predictions["targets"])

        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = self.post_process_detections(detections)

        return (
            predicted_class_labels,
            image_ids,
            predicted_bboxes,
            predicted_class_confidences,
            targets,
        )

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""

        # validation_loss_mean = torch.stack(
        #     [output["loss"] for output in outputs]
        # ).mean()

        # (
        #     predicted_class_labels,
        #     image_ids,
        #     predicted_bboxes,
        #     predicted_class_confidences,
        #     targets,
        # ) = self.aggregate_prediction_outputs(outputs)

        # truth_image_ids = [target["image_id"].detach().item() for target in targets]
        # truth_boxes = [
        #     target["bboxes"].detach()[:, [1, 0, 3, 2]].tolist() for target in targets
        # ] # convert to xyxy for evaluation
        # truth_labels = [target["labels"].detach().tolist() for target in targets]

        # stats = get_coco_stats(
        #     prediction_image_ids=image_ids,
        #     predicted_class_confidences=predicted_class_confidences,
        #     predicted_bboxes=predicted_bboxes,
        #     predicted_class_labels=predicted_class_labels,
        #     target_image_ids=truth_image_ids,
        #     target_bboxes=truth_boxes,
        #     target_class_labels=truth_labels,
        # )['All']

        # return {"val_loss": validation_loss_mean, "metrics": stats}

        #TODO: calc COCO score
        val_loss_mean = torch.stack([output["valid_loss"] for output in outputs]).mean()

        if val_loss_mean < self.best_loss:
            self.best_loss = val_loss_mean
            ckpt_model_name = self.weight_dir + f'/best_loss_fold{self.fold}.pth'
            torch.save(self.model.model.state_dict(), ckpt_model_name)
            print(f'Best Loss found: {self.best_loss}')
            print(f'Best Loss weight saved to: {ckpt_model_name}')

        # tensorboard_logs = {'val_loss': val_loss_mean, 'val_custom_loss': val_custom_loss}
        return {'valid_loss': val_loss_mean}
    
    
    @typedispatch
    def predict(self, images: List):
        """
        For making predictions from images
        Args:
            images: a list of PIL images

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

        """
        image_sizes = [(image.size[1], image.size[0]) for image in images]
        images_tensor = torch.stack(
            [
                self.inference_tfms(
                    image=np.array(image, dtype=np.float32),
                    labels=np.ones(1),
                    bboxes=np.array([[0, 0, 1, 1]]),
                )["image"]
                for image in images
            ]
        )

        return self._run_inference(images_tensor, image_sizes)

    @typedispatch
    def predict(self, images_tensor: torch.Tensor):
        """
        For making predictions from tensors returned from the model's dataloader
        Args:
            images_tensor: the images tensor returned from the dataloader

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

        """
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(0)
        if (
            images_tensor.shape[-1] != self.img_size
            or images_tensor.shape[-2] != self.img_size
        ):
            raise ValueError(
                f"Input tensors must be of shape (N, 3, {self.img_size}, {self.img_size})"
            )

        num_images = images_tensor.shape[0]
        image_sizes = [(self.img_size, self.img_size)] * num_images

        return self._run_inference(images_tensor, image_sizes)

    def _run_inference(self, images_tensor, image_sizes):
        dummy_targets = self._create_dummy_inference_targets(
            num_images=images_tensor.shape[0]
        )

        detections = self.model(images_tensor.to(self.device), dummy_targets)[
            "detections"
        ]
        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = self.post_process_detections(detections)

        scaled_bboxes = self.__rescale_bboxes(
            predicted_bboxes=predicted_bboxes, image_sizes=image_sizes
        )

        return scaled_bboxes, predicted_class_labels, predicted_class_confidences
    
    def _create_dummy_inference_targets(self, num_images):
        dummy_targets = {
            "bbox": [
                torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device)
                for i in range(num_images)
            ],
            "cls": [torch.tensor([1.0], device=self.device) for i in range(num_images)],
            "img_size": torch.tensor(
                [(self.img_size, self.img_size)] * num_images, device=self.device
            ).float(),
            "img_scale": torch.ones(num_images, device=self.device).float(),
        }

        return dummy_targets
    
    def post_process_detections(self, detections):
        predictions = []
        for i in range(detections.shape[0]):
            predictions.append(
                self._postprocess_single_prediction_detections(detections[i])
            )

        predicted_bboxes, predicted_class_confidences, predicted_class_labels = run_wbf(
            predictions, image_size=self.img_size, iou_thr=self.wbf_iou_threshold
        )

        return predicted_bboxes, predicted_class_confidences, predicted_class_labels

    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections.detach().cpu().numpy()[:, :4]
        scores = detections.detach().cpu().numpy()[:, 4]
        classes = detections.detach().cpu().numpy()[:, 5]
        indexes = np.where(scores > self.prediction_confidence_threshold)[0]
        boxes = boxes[indexes]

        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}

    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims

            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (
                        np.array(bboxes)
                        * [
                            im_w / self.img_size,
                            im_h / self.img_size,
                            im_w / self.img_size,
                            im_h / self.img_size,
                        ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)

        return scaled_bboxes

# We can visualise these predictions using a convenience function
def compare_bboxes_for_image(
    image,
    predicted_bboxes,
    actual_bboxes,
    draw_bboxes_fn=draw_pascal_voc_bboxes,
    figsize=(20, 20),
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image)
    ax1.set_title("Prediction")
    ax2.imshow(image)
    ax2.set_title("Actual")

    draw_bboxes_fn(ax1, predicted_bboxes)
    draw_bboxes_fn(ax2, actual_bboxes)

    plt.show()

# @patch
# def aggregate_prediction_outputs(self: EfficientDetModel, outputs):

#     detections = torch.cat(
#         [output["batch_predictions"]["predictions"] for output in outputs]
#     )

#     image_ids = []
#     targets = []
#     for output in outputs:
#         batch_predictions = output["batch_predictions"]
#         image_ids.extend(batch_predictions["image_ids"])
#         targets.extend(batch_predictions["targets"])

#     (
#         predicted_bboxes,
#         predicted_class_confidences,
#         predicted_class_labels,
#     ) = self.post_process_detections(detections)

#     return (
#         predicted_class_labels,
#         image_ids,
#         predicted_bboxes,
#         predicted_class_confidences,
#         targets,
#     )


# From the PyTorch-lightning docs (see https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#validation-epoch-level-metrics), we can see that we can add an additional hook `validation_epoch_end` which is called after all batches have been processed; at the end of each epoch, a list of step outputs are passed to this hook.
# 
# Let's use this hook to calculate the overall validation loss, as well as the COCO metrics using the `objdetecteval` package. we can use the output that we just calculated when evaluating a single validation batch, but this approach would also extend to the validation loop evaluation during training with lightning.
# @patch
# def validation_epoch_end(self: EfficientDetModel, outputs):
#     """Compute and log training loss and accuracy at the epoch level."""

#     validation_loss_mean = torch.stack(
#         [output["loss"] for output in outputs]
#     ).mean()

#     (
#         predicted_class_labels,
#         image_ids,
#         predicted_bboxes,
#         predicted_class_confidences,
#         targets,
#     ) = self.aggregate_prediction_outputs(outputs)

#     truth_image_ids = [target["image_id"].detach().item() for target in targets]
#     truth_boxes = [
#         target["bboxes"].detach()[:, [1, 0, 3, 2]].tolist() for target in targets
#     ] # convert to xyxy for evaluation
#     truth_labels = [target["labels"].detach().tolist() for target in targets]

#     stats = get_coco_stats(
#         prediction_image_ids=image_ids,
#         predicted_class_confidences=predicted_class_confidences,
#         predicted_bboxes=predicted_bboxes,
#         predicted_class_labels=predicted_class_labels,
#         target_image_ids=truth_image_ids,
#         target_bboxes=truth_boxes,
#         target_class_labels=truth_labels,
#     )['All']

#     return {"val_loss": validation_loss_mean, "metrics": stats}

@hydra.main(config_path="../configs", config_name="train_detection")
def main(cfg):
    log = logging.getLogger(__name__)

    eval_validation = False

    model_path = f'./weights/'
    os.makedirs(model_path, exist_ok=True)

    seed_everything(cfg.system.seed)

    df_folds = pd.read_csv(hydra.utils.to_absolute_path(cfg.data.csv_path))
    df = pd.read_csv(hydra.utils.to_absolute_path('input/detection_train.csv'))

    for fold in range(cfg.data.n_fold):
        log.info('------------ Training with {} started ----------------'.format(fold))

        dm = EfficientDetDataModule(cfg=cfg, df_folds=df_folds, df=df, fold=fold,
            train_transforms=get_train_transforms(target_img_size=cfg.model.input_size),
            valid_transforms=get_valid_transforms(target_img_size=cfg.model.input_size))

        model = EfficientDetModel(cfg=cfg, fold=fold, num_classes=cfg.data.num_classes, img_size=cfg.model.input_size, model_architecture=cfg.model.model_name)

        # As the EfficientDet model is just a standard PyTorch Lightning model, it can be trained in the usual way, by importing and creating an appropriate trainer.
        loggers, model_checkpoint, early_stopping = get_callback(cfg, model_path, fold)

        #lr_logger
        trainer = Trainer(logger=loggers,
                        callbacks=[early_stopping, model_checkpoint],
                        **cfg.trainer)

        trainer.fit(model, dm)
        # We can save this like a regular PyTorch model
        save_model(cfg, model_path, fold)
        # torch.save(model.state_dict(), 'trained_effdet')

        # We can load our trained model again as follows
        if eval_validation:
            model = EfficientDetModel(cfg=cfg, fold=fold, num_classes=cfg.data.num_classes, img_size=cfg.model.input_size)

            model.load_state_dict(torch.load('trained_effdet'))
            # ## Using the model for inference
            # Now we have finetuned the model on our dataset, we can inspect some of the predictions. First we put the model into eval mode.
            model.eval()
            # We can now use our dataset adaptor to load a selection of images

            # For showing sample preds
            # image1, truth_bboxes1, _, _ = custom_train_ds.get_image_and_labels_by_idx(0)
            # image2, truth_bboxes2, _, _ = custom_train_ds.get_image_and_labels_by_idx(1)
            # images = [image1, image2]
            # predicted_bboxes, predicted_class_confidences, predicted_class_labels = model.predict(images)
            # compare_bboxes_for_image(image1, predicted_bboxes=predicted_bboxes[0], actual_bboxes=truth_bboxes1.tolist())
            # compare_bboxes_for_image(image2, predicted_bboxes=predicted_bboxes[1], actual_bboxes=truth_bboxes2.tolist())

            loader = dm.val_dataloader()

            dl_iter = iter(loader)
            batch = next(dl_iter)
            # ### Validation outputs and Coco metrics

            # We can use this batch to see exactly what the model calculated during validation. As lightning takes care of moving data to the correct device during training, for simplicity, we shall do this on the cpu so that we don't have to manually move all of the tensors in each batch to the device.
            device = model.device; device

            # Using the model's hook, we can see what is calculated for each batch during each validation step
            output = model.validation_step(batch=batch, batch_idx=0)

            model.validation_epoch_end([output])

            # ### For inference
            # We can also use the predict function directly on the processed images returned from our data loader. Let's  now unpack the batch to just get the images, as we don't need the labels for inference.
            images, annotations,targets, image_ids = batch
            # Thanks to the `typedispatch` decorator, we can use the same predict function signature on these tensors.

            predicted_bboxes, predicted_class_labels, predicted_class_confidences = model.predict(images)

            # It is important to note at this point that the images given by the dataloader have already been transformed and scaled to size 512. Therefore, the bounding boxes predicted will be relative for an image of 512. As such, to visualise these predictions on the original image, we must rescale it.

            # image, _, _, _ = custom_train_ds.get_image_and_labels_by_idx(0)

            # show_image(image.resize((512, 512)), predicted_bboxes[0])


if __name__ == "__main__":
    main()

import albumentations as A
import PIL
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
import cv2
import hydra

def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height


def draw_pascal_voc_bboxes(
    plot_ax,
    bboxes,
    get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox,
):
    for bbox in bboxes:
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=4,
            edgecolor="black",
            fill=False,
        )
        rect_2 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=2,
            edgecolor="white",
            fill=False,
        )

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)

def show_image(
    image, bboxes=None, draw_bboxes_fn=draw_pascal_voc_bboxes, figsize=(10, 10)
):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if bboxes is not None:
        draw_bboxes_fn(ax, bboxes)

    plt.show()


# Usually, at this point, we would create a PyTorch Dataset to feed this data into the training loop. However, some of this code, such as normalising the image and transforming the labels into the required format, are not specific to this problem and will need to be applied regardless of which dataset is being used. Therefore, let’s focus for now on creating a CustomDatasetAdaptor class, which will convert the specific raw dataset format into an image and corresponding annotations.  An implementation of this is presented below
class CustomDatasetAdaptor:
    def __init__(self, images_dir_path, annotations_dataframe):
        self.images_dir_path = images_dir_path
        self.annotations_df = annotations_dataframe
        self.images = self.annotations_df.image.unique().tolist()

    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_labels_by_idx(self, index):
        image_name = self.images[index]
        image = PIL.Image.open(self.images_dir_path / image_name)
        pascal_bboxes = self.annotations_df[self.annotations_df.image == image_name][
            ["xmin", "ymin", "xmax", "ymax"]
        ].values
        class_labels = np.ones(len(pascal_bboxes))

        return image, pascal_bboxes, class_labels, index
    
    def show_image(self, index):
        image, bboxes, class_labels, image_id = self.get_image_and_labels_by_idx(index)
        print(f"image_id: {image_id}")
        show_image(image, bboxes.tolist())
        print(class_labels)


# def get_train_transforms(target_img_size=512):
#     return A.Compose(
#         [
#             A.HorizontalFlip(p=0.5),
#             A.Resize(height=target_img_size, width=target_img_size, p=1),
#             A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ToTensorV2(p=1),
#         ],
#         p=1.0,
#         bbox_params=A.BboxParams(
#             format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
#         ),
#     )


# def get_valid_transforms(target_img_size=512):
#     return A.Compose(
#         [
#             A.Resize(height=target_img_size, width=target_img_size, p=1),
#             A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ToTensorV2(p=1),
#         ],
#         p=1.0,
#         bbox_params=A.BboxParams(
#             format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
#         ),
#     )

def get_train_transforms(target_img_size=512):
    return A.Compose([
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                        val_shift_limit=0.2, p=0.9),
                    A.RandomBrightnessContrast(brightness_limit=0.2, 
                                            contrast_limit=0.2, p=0.9),
                ],p=0.9),
                A.ToGray(p=0.01),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                # A.ImageCompression(quality_lower=85, quality_upper=95, p=0.5),
                A.OneOf([
                    A.Blur(blur_limit=3, p=1.0), 
                    A.MedianBlur(blur_limit=3, p=1.0),
                    A.MotionBlur(p=1)], p=0.5),
                A.Resize(height=target_img_size, width=target_img_size, p=1),
                A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.5), 
                ToTensorV2(p=1.0), 
                ],
                p=1.0,
                bbox_params=A.BboxParams(
                                        format="pascal_voc",
                                        min_area=0, 
                                        min_visibility=0,
                                        label_fields=['labels']
                )
)


def get_valid_transforms(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

class EfficientDetDataset(Dataset):
    def __init__(
        self, dataset_adaptor, transforms=get_valid_transforms()
    ):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        (
            image,
            pascal_bboxes,
            class_labels,
            image_id,
        ) = self.ds.get_image_and_labels_by_idx(index)

        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }

        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        pascal_bboxes = sample["bboxes"]
        labels = sample["labels"]

        _, new_h, new_w = image.shape
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][
            :, [1, 0, 3, 2]
        ]  # convert to yxyx

        target = {
            "bboxes": torch.as_tensor(pascal_bboxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }

        return image, target, image_id

    def __len__(self):
        return len(self.ds)


class DatasetRetriever(Dataset):

    def __init__(self, train_root_path, marking, image_ids, transforms=None, test=False, image_ext = 'png'):
        super().__init__()
        self.train_root_path = train_root_path
        self.image_ext = image_ext
        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.test = test

    def __getitem__(self, index: int):
        index = self.image_ids[index]
        image_name = self.marking.loc[index]['image_name']

        # if self.test or random.random() > 0.3:
        image, boxes = self.load_image_and_boxes(index)
        # else:
            # image, boxes = self.load_cutmix_image_and_boxes(index)

        new_h, new_w, _ = image.shape

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]), 
                'img_size': (new_h, new_w),'img_scale':torch.tensor([1.0])}
        
        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:, [0, 1, 2, 3]] = target['boxes'][:, [1, 0, 3, 2]]  # yxyx: be warning
                    target['labels'] = target['labels'][:len(target['boxes'])]
                    break

        return image, target, image_name

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_name = self.marking['image_name'][index]
        image_name = image_name.replace('_image', '')
        # if not os.path.exists(self.train_root_path + '/'+ image_name + '.' + self.image_ext):
        image_path = self.train_root_path + '/'+ image_name + '.' + self.image_ext
        image_path = hydra.utils.to_absolute_path(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        row = self.marking.loc[index]

        bboxes = []
        if row['BoxesString'] != 'no_box':
            for bbox in row['BoxesString'].split(';'):
                bboxes.append(list(map(float, bbox.split(' '))))
        return image, np.array(bboxes)


## Define the DataModule
class EfficientDetDataModule(LightningDataModule):
    
    def __init__(self,
                cfg,
                df_folds,
                df,
                fold,
                # train_dataset_adaptor,
                # validation_dataset_adaptor,
                train_transforms,
                valid_transforms,
                ):
        
        # self.train_ds = train_dataset_adaptor
        # self.valid_ds = validation_dataset_adaptor
        self.cfg = cfg
        self.df_folds = df_folds
        self.df = df
        self.fold = fold
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.fold_number = cfg.data.n_fold
        super().__init__()

    def train_dataset(self) -> DatasetRetriever:
        return DatasetRetriever(train_root_path=self.cfg.data.train_image_dir, image_ids=self.df_folds[self.df_folds['fold'] != self.fold].index.values,
            marking=self.df,
            transforms=self.train_tfms,
            test=False
            )
        # return EfficientDetDataset(
        #     dataset_adaptor=self.train_ds, transforms=self.train_tfms
        # )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.cfg.system.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader

        # create datasets using fold_number
        # train_dataset = DatasetRetriever(
        #     image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
        #     marking=df,
        #     transforms=get_train_transforms(),
        #     test=False,
        # )

        # validation_dataset = DatasetRetriever(
        #     image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
        #     marking=df,
        #     transforms=get_valid_transforms(),
        #     test=True,
        # )

    def val_dataset(self) -> DatasetRetriever:
        return DatasetRetriever(
                train_root_path=self.cfg.data.train_image_dir,
                image_ids=self.df_folds[self.df_folds['fold'] == self.fold].index.values,
                marking=self.df,
                transforms=self.valid_tfms,
                test=True
                )
        # return EfficientDetDataset(
        #     dataset_adaptor=self.valid_ds, transforms=self.valid_tfms
        # )

    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.cfg.system.num_workers,
            collate_fn=self.collate_fn,
        )

        return valid_loader
    
    # @staticmethod
    # def collate_fn(batch):
    #     images, targets, image_ids = tuple(zip(*batch))
    #     images = torch.stack(images)
    #     images = images.float()

    #     boxes = [target["bboxes"].float() for target in targets]
    #     labels = [target["labels"].float() for target in targets]
    #     img_size = torch.tensor([target["img_size"] for target in targets]).float()
    #     img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

    #     annotations = {
    #         "bbox": boxes,
    #         "cls": labels,
    #         "img_size": img_size,
    #         "img_scale": img_scale,
    #     }

    #     return images, annotations, targets, image_ids

    # def collate_fn(batch):
    #     return tuple(zip(*batch))

    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["boxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        return images, annotations, targets, image_ids
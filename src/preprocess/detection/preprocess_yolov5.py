# Necessary/extra dependencies. 
import os
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from shutil import copyfile
from sklearn.model_selection import train_test_split, StratifiedKFold
import shutil

# Get the raw bounding box by parsing the row value of the label column.
# Ref: https://www.kaggle.com/yujiariyasu/plot-3positive-classes
def get_bbox(row):
    bboxes = []
    bbox = []
    for i, l in enumerate(row.label.split(' ')):
        if (i % 6 == 0) | (i % 6 == 1):
            continue
        bbox.append(float(l))
        if i % 6 == 5:
            bboxes.append(bbox)
            bbox = []  
            
    return bboxes

# Scale the bounding boxes according to the size of the resized image. 
def scale_bbox(row, bboxes):
    # Get scaling factor
    scale_x = IMG_SIZE/row.dim1
    scale_y = IMG_SIZE/row.dim0
    
    scaled_bboxes = []
    for bbox in bboxes:
        x = int(np.round(bbox[0]*scale_x, 4))
        y = int(np.round(bbox[1]*scale_y, 4))
        x1 = int(np.round(bbox[2]*(scale_x), 4))
        y1= int(np.round(bbox[3]*scale_y, 4))

        scaled_bboxes.append([x, y, x1, y1]) # xmin, ymin, xmax, ymax
        
    return scaled_bboxes

# Convert the bounding boxes in YOLO format.
def get_yolo_format_bbox(img_w, img_h, bboxes):
    yolo_boxes = []
    for bbox in bboxes:
        w = bbox[2] - bbox[0] # xmax - xmin
        h = bbox[3] - bbox[1] # ymax - ymin
        xc = bbox[0] + int(np.round(w/2)) # xmin + width/2
        yc = bbox[1] + int(np.round(h/2)) # ymin + height/2
        
        yolo_boxes.append([xc/img_w, yc/img_h, w/img_w, h/img_h]) # x_center y_center width height
    
    return yolo_boxes


def write_bbox_files(tmp_df, fold_num, split):
    path = f'./input/dataset_folds_{fold}/labels/{split}'
    for i in tqdm(range(len(tmp_df))):
        row = tmp_df.loc[i]
        # Get image id
        img_id = row.id
        # Get image-level label
        label = row.image_level

        file_name = f'{path}/{img_id}.txt'

        if label==1:
            # Get bboxes
            bboxes = get_bbox(row)
            # Scale bounding boxes
            scale_bboxes = scale_bbox(row, bboxes)
            # Format for YOLOv5
            yolo_bboxes = get_yolo_format_bbox(IMG_SIZE, IMG_SIZE, scale_bboxes)

            with open(file_name, 'w') as f:
                for bbox in yolo_bboxes:
                    bbox = [1]+bbox
                    bbox = [str(i) for i in bbox]
                    bbox = ' '.join(bbox)
                    f.write(bbox)
                    f.write('\n')


def preprocess_image_level_df(image_level_path='./input/train_image_level.csv'):
    df = pd.read_csv(image_level_path)

    # Modify values in the id column
    df['id'] = df.apply(lambda row: row.id.split('_')[0], axis=1)
    # Add absolute path
    # df['path'] = df.apply(lambda row: f'{TRAIN_PATH}/{row.id}.png', axis=1)
    # Get image level labels
    # df['image_level'] = df.apply(lambda row: row.label.split(' ')[0], axis=1)

    def _image_level(row):
        label = row.label.split(' ')[0]
        if label == 'opacity': return 1
        else: return 0

    df['image_level'] = df.apply(lambda row: _image_level(row), axis=1)
    print('Done Preprocess Image Level Data')
    return df


def preprocess_meta_df():
    meta_df = pd.read_csv('./input/meta.csv') #siim-covid19-resized-to-256px-png
    train_meta_df = meta_df.loc[meta_df.split == 'train']
    train_meta_df = train_meta_df.drop('split', axis=1)
    train_meta_df.columns = ['id', 'dim0', 'dim1']
    print('Done Preprocess Meta Data')
    return train_meta_df


def preprocess_study_level(image_level_df):
    label_df = pd.read_csv('./input/train_study_level.csv')

    # Modify values in the id column
    label_df['id'] = label_df.apply(lambda row: row.id.split('_')[0], axis=1)
    # Rename the column id with StudyInstanceUID
    label_df.columns = ['StudyInstanceUID', 'Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']

    # Label encode study-level labels
    labels = label_df[['Negative for Pneumonia','Typical Appearance','Indeterminate Appearance','Atypical Appearance']].values
    labels = np.argmax(labels, axis=1)
    label_df['study_level'] = labels

    # ORIGINAL DIMENSION

    # Load meta.csv file
    train_meta_df = preprocess_meta_df()
    # Merge image-level and study-level
    image_study_total = image_level_df.merge(label_df, on='StudyInstanceUID',how="left")
    # Merge with meta_df
    image_study_total = image_study_total.merge(train_meta_df, on='id',how="left")

    # Write as csv file
    image_study_total.to_csv('./input/_image_study_total.csv', index=False)
    print(f'Image Study Total csv file saved to: ./input/_image_study_total.csv')
    return image_study_total


def make_image_level_fold(img_path='./input/siim-covid19-resized-to-256px-png/train/', n_fold=5, seed=42):
    # /Users/rick/Dropbox/python_projects/data_science/Kaggle/siim_covid19/input/siimcovid19-512-img-png-600-study-png/image
    df = pd.read_csv('./input/_image_study_total.csv')
    df['path'] = df.apply(lambda row: f'{img_path}/{row.id}_image.png', axis=1)
    
    # Group by Study Ids and remove images that are "assumed" to be mislabeled
    for grp_df in df.groupby('StudyInstanceUID'):
        grp_id, grp_df = grp_df[0], grp_df[1]
        if len(grp_df) == 1:
            pass
        else:
            for i in range(len(grp_df)):
                row = grp_df.loc[grp_df.index.values[i]]
                if row.study_level > 0 and row.boxes is np.nan:
                    df = df.drop(grp_df.index.values[i])
                    
    print('total number of images: ', len(df))
    
    # Create train and validation split.
    df = df.drop('boxes', axis=1).reset_index()
    Fold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    for n, (train_index, val_index) in enumerate(Fold.split(df, df['image_level'])):
        df.loc[val_index, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)

    df.to_csv('./input/train_fold.csv', index=False)
    print(f'Image Level Fold csv file saved to: ./input/train_fold.csv')
    return df


def create_yaml_file(n_fold=5):
    for fold in range(n_fold):
        data_yaml = dict(
            train = f'./input/dataset_folds_{fold}/images/train',
            val = f'./input/dataset_folds_{fold}/images/valid',
            nc = 2,
            names = ['none', 'opacity']
        )

        # Note that I am creating the file in the yolov5/data/ directory.
        with open(f'./src/models/yolov5/data/data_fold_{fold}.yaml', 'w') as outfile:
            yaml.dump(data_yaml, outfile, default_flow_style=True)
        print(f'config yaml file saved to: ./src/models/yolov5/data/data_fold_{fold}.yaml')


def prepare_images(df, n_fold=5):
     # Remove existing dirs
    for fold in range(n_fold):
        print(f'Preparing Images for Fold {fold}')
        # Prepare train and valid df
        train_df = df.loc[df.fold != fold].reset_index(drop=True)
        valid_df = df.loc[df.fold == fold].reset_index(drop=True)
        
        try:
            shutil.rmtree(f'./input/dataset_folds_{fold}/images')
            shutil.rmtree(f'./input/dataset_folds_{fold}/labels')
            print(f'Deleted: ./input/dataset_folds_{fold}/images')
            print(f'Deleted: ./input/dataset_folds_{fold}/labels')
        except:
            print('No dirs')

        # Make new dirs
        os.makedirs(f'./input/dataset_folds_{fold}/images/train', exist_ok=True)
        os.makedirs(f'./input/dataset_folds_{fold}/images/valid', exist_ok=True)
        os.makedirs(f'./input/dataset_folds_{fold}/labels/train', exist_ok=True)
        os.makedirs(f'./input/dataset_folds_{fold}/labels/valid', exist_ok=True)
        print(f'Made Directory: ./input/dataset_folds_{fold}/images/train')
        print(f'Made Directory: ./input/dataset_folds_{fold}/images/valid')
        print(f'Made Directory: ./input/dataset_folds_{fold}/labels/train')
        print(f'Made Directory: ./input/dataset_folds_{fold}/labels/valid')

        # Move the images to relevant split folder.
        for i in tqdm(range(len(train_df))):
            row = train_df.loc[i]
            copyfile(row.path, f'./input/dataset_folds_{fold}/images/train/{row.id}.png')
            
        for i in tqdm(range(len(valid_df))):
            row = valid_df.loc[i]
            copyfile(row.path, f'./input/dataset_folds_{fold}/images/valid/{row.id}.png')

    print('Done Prepare Images')


if __name__ == '__main__':
    # TRAIN_PATH = 'input/siim-covid19-resized-to-256px-jpg/train/'
    TRAIN_PATH = 'input/siimcovid19-512-img-png-600-study-png/image' #'./input/train/'

    n_fold = 5
    IMG_SIZE = 512 #256
    seed=42

    if n_fold == 5:
        image_level_df = preprocess_image_level_df()
        image_study_total = preprocess_study_level(image_level_df)
        image_level_fold_df = make_image_level_fold(img_path=TRAIN_PATH, n_fold=n_fold, seed=seed)
        prepare_images(image_level_fold_df, n_fold=n_fold)
        create_yaml_file(n_fold=n_fold)

        # Prepare the txt files for bounding box
        for fold in range(n_fold):
            # Prepare train and valid df
            train_df = image_level_fold_df.loc[image_level_fold_df.fold != fold].reset_index(drop=True)
            valid_df = image_level_fold_df.loc[image_level_fold_df.fold == fold].reset_index(drop=True)
            
            # prepare label for train
            write_bbox_files(train_df, fold, 'train')
            # prepare label for valid
            write_bbox_files(valid_df, fold, 'valid')

    elif n_fold == 1:
        if not os.path.exists('./input/train_v1.csv') and n_fold ==1:
            # Load image level csv file
            df = preprocess_image_level_df()
            # Load meta.csv file
            # Original dimensions are required to scale the bounding box coordinates appropriately.
            train_meta_df = preprocess_meta_df()

            image_level_df = df.merge(train_meta_df, on='id',how="left")
            image_level_df.to_csv('./input/train_v1.csv', index=False)

        df = pd.read_csv('./input/train_v1.csv')
        # Create train and validation split.
        train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.image_level.values)

        train_df.loc[:, 'split'] = 'train'
        valid_df.loc[:, 'split'] = 'valid'

        train_df.loc[:, 'fold'] = 1
        valid_df.loc[:, 'fold'] = 0

        df = pd.concat([train_df, valid_df]).reset_index(drop=True)
        df.to_csv('./input/train_image_v1_fold.csv', index=False)


        print(f'Size of dataset: {len(df)}, training images: {len(train_df)}. validation images: {len(valid_df)}')


        os.makedirs('./input/tmp/covid/images/train', exist_ok=True)
        os.makedirs('./input/tmp/covid/images/valid', exist_ok=True)

        os.makedirs('./input/tmp/covid/labels/train', exist_ok=True)
        os.makedirs('./input/tmp/covid/labels/valid', exist_ok=True)


        # Move the images to relevant split folder.
        for i in tqdm(range(len(df))):
            row = df.loc[i]
            if row.split == 'train':
                copyfile(row.path, f'./input/tmp/covid/images/train/{row.id}.jpg')
            else:
                copyfile(row.path, f'./input/tmp/covid/images/valid/{row.id}.jpg')


        # ## 🍜 Create `.YAML` file
        # 
        # The `data.yaml`, is the dataset configuration file that defines 
        # 
        # 1. an "optional" download command/URL for auto-downloading, 
        # 2. a path to a directory of training images (or path to a *.txt file with a list of training images), 
        # 3. a path to a directory of validation images (or path to a *.txt file with a list of validation images), 
        # 4. the number of classes, 
        # 5. a list of class names.
        # 
        # > 📍 Important: In this competition, each image can either belong to `opacity` or `none` image-level labels. That's why I have  used the number of classes, `nc` to be 2. YOLOv5 automatically handles the images without any bounding box coordinates. 
        # 
        # > 📍 Note: The `data.yaml` is created in the `yolov5/data` directory as required. 
        # Create .yaml file 
        data_yaml = dict(
            train = './input/tmp/covid/images/train',
            val = './input/tmp/covid/images/valid',
            # nc = 2,
            # names = ['none', 'opacity']
            nc = 4,
            names = ['Negative for Pneumonia', 'Typical Appearance',
                    'Indeterminate Appearance', 'Atypical Appearance']
        )

        # Note that I am creating the file in the yolov5/data/ directory.
        with open('../src/yolov5/data/data_v2.yaml', 'w') as outfile:
            yaml.dump(data_yaml, outfile, default_flow_style=True)
            

        # ## 🍮 Prepare Bounding Box Coordinated for YOLOv5
        # 
        # For every image with **bounding box(es)** a `.txt` file with the same name as the image will be created in the format shown below:
        # 
        # * One row per object. <br>
        # * Each row is class `x_center y_center width height format`. <br>
        # * Box coordinates must be in normalized xywh format (from 0 - 1). We can normalize by the boxes in pixels by dividing `x_center` and `width` by image width, and `y_center` and `height` by image height. <br>
        # * Class numbers are zero-indexed (start from 0). <br>
        # 
        # > 📍 Note: We don't have to remove the images without bounding boxes from the training or validation sets. 

        # Prepare the txt files for bounding box
        for i in tqdm(range(len(df))):
            row = df.loc[i]
            # Get image id
            img_id = row.id
            # Get split
            split = row.split
            # Get image-level label
            label = row.image_level
            
            if row.split=='train':
                file_name = f'./input/tmp/covid/labels/train/{row.id}.txt'
            else:
                file_name = f'./input/tmp/covid/labels/valid/{row.id}.txt'
                
            
            if label=='opacity':
                # Get bboxes
                bboxes = get_bbox(row)
                # Scale bounding boxes
                scale_bboxes = scale_bbox(row, bboxes)
                # Format for YOLOv5
                yolo_bboxes = get_yolo_format_bbox(IMG_SIZE, IMG_SIZE, scale_bboxes)
                
                with open(file_name, 'w') as f:
                    for bbox in yolo_bboxes:
                        bbox = [1]+bbox
                        bbox = [str(i) for i in bbox]
                        bbox = ' '.join(bbox)
                        f.write(bbox)
                        f.write('\n')
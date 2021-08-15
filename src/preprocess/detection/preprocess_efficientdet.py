import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from src.utils.common import seed_everything


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed

def prepare_csv_for_efdet(input_filepath, output_filepath, img_path='./input/siimcovid19-512-img-png-600-study-png/image'):
    train_image_df = pd.read_csv(input_filepath)
    train_image_df['id'] = train_image_df['id'].str.split('_', expand=True)[0]

    image_ids = train_image_df['id'].tolist()
    labels_raw = train_image_df['label'].tolist()
    boxes_raw = train_image_df['boxes'].tolist()

    images = get_all_files_in_folder(Path(img_path), ['*.png'])

    result = []
    for image_path in tqdm(images):
        s = image_path.stem + ','

        boxes = []
        for image_id, label in zip(image_ids, labels_raw):
            image_id = image_id + '_image'
            
            if image_id == image_path.stem:

                label_split = label.split('opacity')
                if len(label_split) > 1:

                    for l in label_split:
                        if l != '':
                            box = l.split(' ')
                            x1 = (float(box[2]))
                            if x1 < 0: x1 = 0.0
                            y1 = (float(box[3]))
                            if y1 < 0: y1 = 0.0
                            x2 = (float(box[4]))
                            y2 = (float(box[5]))

                            boxes.append([x1, y1, x2, y2])

        boxes_str = ''
        if len(boxes):
            for box in boxes:
                boxes_str += str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ';'

            s += boxes_str[:-1]
        else:
            boxes_str = 'no_box'
            s += boxes_str

        s += ',0'
        result.append(s)

    with open(output_filepath, 'w') as f:
        f.write('image_name,BoxesString,domain\n')
        for item in result:
            f.write("%s\n" % item)


if __name__ == '__main__':
    # extract all labels and save them to CSV file
    n_fold = 5
    seed = 42
    seed_everything(seed)
    input_filepath = './input/train_image_level.csv'
    output_folds_filepath = './input/detection_train_fold.csv'
    output_filepath = './input/detection_train.csv'
    prepare_csv_for_efdet(input_filepath, output_filepath)

    # read our new CSV
    df = pd.read_csv(output_filepath)

    # remove samples without bboxes
    df.drop(df[df.BoxesString == 'no_box'].index, inplace=True)

    df.to_csv(output_filepath, index=False)
    # divide on folds
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    df_folds = df[['image_name']].copy()
    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df.image_name, y=df['domain'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    df_folds.to_csv(output_folds_filepath, index=False)
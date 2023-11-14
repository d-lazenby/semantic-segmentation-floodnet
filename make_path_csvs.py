import pandas as pd
import os

import argparse


def get_image_path_dfs(data_dir, split):
    for _, __, f in os.walk(os.path.join(data_dir, split, f'{split}-org-img')):
        df = pd.DataFrame(data=f, columns=['images'])

    df['masks'] = df['images'].str.extract("(\d{4})") + "_lab.png"

    df['images'] = f'{data_dir}/{split}/{split}-org-img/' + df['images']
    df['masks'] = f'{data_dir}/{split}/{split}-label-img/' + df['masks']

    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/MLEng/Projects/semantic-segmentation-floodnet/FloodNet-Supervised_v1.0')
    args = parser.parse_args()

    splits = ['train', 'val', 'test']
    for split in splits:
        if not os.path.exists(f'{split}.csv'):
            print(f"Getting paths for images and masks for {split}...")
            df = get_image_path_dfs(args.data_dir, split)
            print("Images and masks successfully added to dataframe...")
            df.to_csv(f"{split}.csv", index=False)
            print(f"Images and masks successfully added to csv for {split} images\n")
        else:
            print(f"File {split}.csv already exists")

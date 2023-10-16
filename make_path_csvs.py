import pandas as pd
import os


def get_image_path_dfs(split):
    for _, __, f in os.walk(os.path.join('./', split, f'{split}-org-img')):
        df = pd.DataFrame(data=f, columns=['images'])

    df['masks'] = df['images'].str.extract("(\d{4})") + "_lab.png"

    df['images'] = f'semantic-segmentation-floodnet/{split}/{split}-org-img/' + df['images']
    df['masks'] = f'semantic-segmentation-floodnet/{split}/{split}-label-img/' + df['masks']

    return df


if __name__ == '__main__':
    splits = ['train', 'val', 'test']
    for split in splits:
        df = get_image_path_dfs(split)
        df.to_csv(f"{split}.csv", index=False)

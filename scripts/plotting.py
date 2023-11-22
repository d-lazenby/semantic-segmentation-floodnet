import pandas as pd
import cv2
import matplotlib.pyplot as plt


def show_image_mask_pair(df: pd.DataFrame, idx: int) -> None:
    row = df.iloc[idx]

    image_path = row['images']
    mask_path = row['masks']

    image = cv2.imread(image_path)
    # Matplotlib expects RGB but OpenCV provides it in BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Explicitly set to grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    ax[0].set_title('Image')
    ax[0].imshow(image)

    ax[1].set_title('Ground Truth')
    ax[1].imshow(mask, cmap=None)

    plt.show()
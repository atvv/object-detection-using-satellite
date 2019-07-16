from PIL import Image
from collections import defaultdict
from os import makedirs
from params import IMG_SIZE
import pandas as pd

SIZE = 2000
DIMS = dict(zip(range(9), [12, 30, 30, 30, 30, 30, 30, 40, 45]))


def extract_crops_sw(df, size, save_images, step):
    "Extract images using a sliding window. Save vehicles coordinates."
    # Encode class
    encoder = dict(zip('ABCDEFGHI', range(50)))
    df['class_id'] = df['class'].apply(lambda x: encoder[x])

    images = defaultdict(list)
    for row in df.itertuples():
        if row.detections == 'None':
            continue
        vehicles = [x.split(':') for x in row.detections.split('|')]
        vehicles = [(row.class_id, (int(x[0]), int(x[1]))) for x in vehicles]
        images[row.image] += vehicles

    rows = []
    for path, vehicles in images.items():
        for y in list(range(0, SIZE - size, step)) + [SIZE - size]:
            for x in list(range(0, SIZE - size, step)) + [SIZE - size]:
                # New image path
                directory = "data/training_sliding/%s/" % path.replace('.jpg', '')
                save_path = directory + path.replace('.', '_x%sy%s.' % (x, y))

                positions = []
                for class_id, (v_x, v_y) in vehicles:
                    if class_id == 8:
                        continue
                    if all([v_x >= x, v_x < x + size, v_y >= y, v_y < y + size]):
                        positions += [class_id, v_x - x, v_y - y,
                                      DIMS[class_id], DIMS[class_id]]

                # Save new image
                if save_images and positions:
                    try:
                        makedirs(directory)
                    except:
                        pass
                    img = Image.open("data/training/" + path)
                    img2 = img.crop((x, y, x + size, y + size))
                    img2.save(save_path)
                if positions:
                    rows += [[save_path] + positions]

    # Dataset to use for training
    df_out = pd.DataFrame(rows)
    df_out = df_out.rename_axis({0: 'path'}, axis="columns")
    return df_out


def main():
    df = pd.read_csv('data/trainingObservations.csv')
    df_out = extract_crops_sw(df, IMG_SIZE, True, 150)
    df_out.to_csv('data/training_cropped.csv')

if __name__ == '__main__':
    main()

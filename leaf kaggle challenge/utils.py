from PIL import Image, ImageChops, ImageOps
import os
import shutil
import pandas as pd
from sklearn import model_selection


def extract(train_data):
    train = pd.read_csv(train_data)
    _labelmap = dict(zip(train['id'], train['species']))
    _class = set(train['species'])
    return _labelmap, _class


def delete_folders(dirs=['test', 'train', 'validation'], dir_path='leaf/images/'):

    for directory in dirs:
        if os.path.exists(dir_path + directory):
            shutil.rmtree(dir_path + directory)


def pic_resize(f_in, size=(96, 96), pad=True):

    image = Image.open(f_in)
    image.thumbnail(size, Image.ANTIALIAS)
    # if image.size[1] % 2 != 0:
    #     image = image.crop((0, 0, image.size[0], image.size[1] - 1))
    # if image.size[0] % 2 != 0:
    #     image = image.crop((0, 0, image.size[0] - 1, image.size[1]))
    image_size = image.size

    if pad:
        thumb = image.crop((0, 0, size[0], size[1]))

        offset_x = max((size[0] - image_size[0]) // 2, 0)
        offset_y = max((size[1] - image_size[1]) // 2, 0)

        thumb = ImageChops.offset(thumb, offset_x, offset_y)

    else:
        thumb = ImageOps.fit(image, size, Image.ANTIALIAS, (0.5, 0.5))

    return thumb




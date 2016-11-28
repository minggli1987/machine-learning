from PIL import Image, ImageChops, ImageOps
from os import scandir, makedirs
import pandas as pd


def _label(train_data):
    train = pd.read_csv(train_data)
    _labelmap = dict(zip(train['id'], train['species']))
    _class = set(train['species'])
    return _labelmap, _class

path = 'leaf/images/'

label_map, classes = _label('leaf/train.csv')


def f_resize(f_in, f_out, size=(96, 96), pad=True):
    image = Image.open(f_in)
    image.thumbnail(size, Image.ANTIALIAS)
    image_size = image.size

    if pad:
        thumb = image.crop((0, 0, size[0], size[1]))

        offset_x = max((size[0] - image_size[0]) / 2, 0)
        offset_y = max((size[1] - image_size[1]) / 2, 0)

        thumb = ImageChops.offset(thumb, offset_x, offset_y)

    else:
        thumb = ImageOps.fit(image, size, Image.ANTIALIAS, (0.5, 0.5))

    thumb.save(f_out)


pic_names = [i.name for i in scandir(path) if i.is_file()]

for _, name in enumerate(pic_names):
    leaf_id = int(name.split('.')[0])
    if leaf_id in label_map.keys():
        directory = path + 'train/' + label_map[leaf_id]
    else:
        directory = path + 'test/' +
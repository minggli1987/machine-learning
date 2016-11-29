from PIL import Image, ImageChops, ImageOps
from os import scandir, makedirs, path
import pandas as pd
from sklearn import model_selection


def _label(train_data):
    train = pd.read_csv(train_data)
    _labelmap = dict(zip(train['id'], train['species']))
    _class = set(train['species'])
    return _labelmap, _class

dir_path = 'leaf/images/'

label_map, classes = _label('leaf/train.csv')


def f_resize(f_in, size=(96, 96), pad=True):

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

pic_names = [i.name for i in scandir(dir_path) if i.is_file()]

kf_gen = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

train_x = list(label_map.keys())
train_y = list(label_map.values())
leaf_images = dict()

for train_index, valid_index in kf_gen.split(train_x, train_y):

    train_id = [train_x[i] for i in train_index]
    valid_id = [train_x[i] for i in valid_index]

    for _, name in enumerate(pic_names):

        leaf_id = int(name.split('.')[0])
        leaf_images[leaf_id] = f_resize(dir_path + name)

        if leaf_id in train_id:
            directory = dir_path + 'train/' + label_map[leaf_id]
        elif leaf_id in valid_id:
            directory = dir_path + 'validation/' + label_map[leaf_id]
        else:
            directory = dir_path + 'test'
        if not path.exists(directory):
            makedirs(directory)

        leaf_images[leaf_id].save(directory+'/' + name)

    break


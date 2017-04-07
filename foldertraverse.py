import os
import collections
from pprint import PrettyPrinter

def folder_traverse(root_dir, ext=('.jpg')):
    """map all image-only files in a folder"""

    if not os.path.exists(root_dir):
        raise RuntimeError('{0} doesn\'t exist.'.format(root_dir))

    file_structure = collections.defaultdict(list)
    for item in os.scandir(root_dir):
        if item.is_dir():
            file_structure.update(folder_traverse(item.path, ext))
        elif item.is_file() and item.name.endswith(ext):
            dirname = os.path.dirname(item.path)
            file_structure[dirname].append(item.name)
    return file_structure


script_path = os.path.dirname(os.path.realpath(__file__))

tree_structure = folder_traverse(
                        root_dir=script_path,
                        ext=('.py'))

pp = PrettyPrinter(indent=4)


def folder_recursion(root_dir, ext=('.jpg')):
    import collections
    file_structure = collections.defaultdict(list)
    for item in os.scandir(root_dir):
        if item.is_dir():
            file_structure.update(folder_recursion(item.path, ext))
        elif item.is_file() and item.name.endswith(ext):
            dirname = os.path.split(item.path)[0]
            file_structure[dirname].append(item.name)
    return file_structure

a = folder_traverse(root_dir=os.path.join(script_path), ext=('.py', '.csv'))

b = folder_recursion(root_dir=script_path, ext=('.py', '.csv'))

assert a==b
print(b)

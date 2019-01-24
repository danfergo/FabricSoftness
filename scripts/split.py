import os
import random
import operator
import math
from shutil import copyfile
import pathlib


def percentages_to_abs(splits, n_elements):
    assert (sum([x[1] for x in splits.items()]) == 1)
    max_split = max(splits.items(), key=operator.itemgetter(1))
    other_splits = [x for x in splits.items() if x[0] != max_split[0]]

    abs_splits = [(x[0], math.floor(x[1] * n_elements)) for x in other_splits]
    total = sum([x[1] for x in abs_splits])

    abs_splits = [(max_split[0], n_elements - total)] + abs_splits
    return dict(abs_splits)


def do_splits(dataset, splits):
    min_category = min(dataset, key=operator.itemgetter(1))
    other_categories = [cat for cat in dataset if cat[0] != min_category[0]]
    n_elems_per_cat = len(min_category[1])

    balanced_dataset = [min_category] + [(x[0], random.sample(x[1], n_elems_per_cat)) for x in other_categories]

    abs_splits = percentages_to_abs(splits, n_elems_per_cat)

    balanced_split_dataset = {name: {cat_name: None for cat_name, _ in dataset} for name, _ in abs_splits.items()}
    for cat_name, cat_elems in balanced_dataset:
        to_sample = cat_elems
        for s_name, s_n_elems in abs_splits.items():
            sample = random.sample(to_sample, s_n_elems)
            to_sample = list(set(to_sample) - set(sample))
            balanced_split_dataset[s_name][cat_name] = sample

    return balanced_split_dataset


def detect_categories(path):
    return [(x[0].split('/')[-1], [ x[0] + '/' + f_name for f_name in x[2]]) for x in os.walk(path) if
            len(x[2]) > 0]


def print_stats(dataset):
    print('DATASET')
    for split_name, split in dataset.items():
        print(', '.join([str(len(elems)) + ' ' + cat_name.capitalize() for cat_name, elems in split.items()]))


def write_images(dataset, output_path):
    pathlib.Path(output_path).mkdir(parents=True)

    for split_name, split in dataset.items():
        for cat_name, elems in split.items():
            pathlib.Path(output_path + '/' + split_name + '/' + cat_name).mkdir(parents=True, exist_ok=True)
            for elem in elems:
                copyfile(elem, output_path + '/' + split_name + '/' + cat_name + '/' + (elem.split('/')[-1]))


if __name__ == '__main__':
    from_path = 'images/resized'
    to_path = 'images/split/'

    print('Generating splits...')
    raw_dataset = detect_categories(from_path)
    splits = {'train': 0.8, 'validation': 0.1, 'test': 0.1}
    balanced_split_dataset = do_splits(raw_dataset, splits)
    print('Writing...')
    write_images(balanced_split_dataset, to_path)
    print_stats(balanced_split_dataset)
    print('Done.')

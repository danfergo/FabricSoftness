import cv2
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt


def cut1(img, background, wh, xy):
    x, y = xy
    w, h = wh

    cropped_img = img[y:y + h, x:x + w, :]
    cropped_bg = background[y:y + h, x:x + w, :]

    diff = np.sum(cv2.absdiff(cropped_img, cropped_bg))
    return ((diff > 1500000) and (diff < 6500000)), cropped_img
    # if(diff < 6000000):
    #     cv2.imshow('frame', cropped_img)
    #     cv2.waitKey(1000)
    # else:
    #     cv2.imwrite(prefix_path, cropped_img)


def cut5(from_path, to_path, category, background, cuts, wh):
    input_dir = from_path + '/' + category +'/'
    good_dir = to_path + '/' + category + '/'
    excluded_dir = to_path + '/' + category + '_excluded/'

    pathlib.Path(good_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(excluded_dir).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        original_file_name = filename.split('.')[0]

        for idx, xy in enumerate(cuts):
            good, cropped_img = cut1(img, background, wh, xy)
            directory = good_dir if good else excluded_dir
            cv2.imwrite(directory + original_file_name +  str(idx + 1) + '.png', cropped_img)


def compute_max_min_diffs(from_path, background, category, cuts, wh):
    input_dir = from_path + '/' + category

    max_min_diffs = np.zeros((len(cuts), 2))

    lst = []
    for filename in os.listdir(input_dir):

        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        for i, xy in enumerate(cuts):
            x, y = xy
            w, h = wh
            cropped_img = img[y:y + h, x:x + w, :]
            cropped_bg = background[y:y + h, x:x + w, :]
            diff = np.sum(cv2.absdiff(cropped_img, cropped_bg))
            lst.append(diff)
            # print(diff)

    return lst


if __name__ == '__main__':
    from_path = 'images/raw/'
    to_path = 'images/resized/'
    categories = ['soft', 'stiff']

    background = cv2.imread('images/raw/background/2.png')

    cuts = [(80, 0), (320, 0), (80, 240), (320, 240), (120, 200)]
    wh = (240, 240)

    pathlib.Path(to_path).mkdir(parents=True)
    # diffs = []
    # for category in categories:
    #     print('Computing differences to the background for category ' + category + ' ...')
    #     cat_diffs = compute_max_min_diffs(from_path, background, category, cuts, wh)
    #     diffs = diffs + cat_diffs
    #
    #     np_lst = np.asarray(cat_diffs)
    #     plt.hist(np_lst, normed=True, bins=200)
    #     plt.ylabel('Probability')
    #     plt.show()
    #     print('max: ', np.max(np_lst))
    #     print('min: ', np.min(np_lst))
    #
    #
    # np_lst = np.asarray(diffs)
    # plt.hist(np_lst, normed=True, bins=200)
    # plt.ylabel('Probability')
    # plt.show()
    # print('max: ', np.max(np_lst))
    # print('min: ', np.min(np_lst))
    for category in categories:
        print('Writing category ' + category + '...')
        cut5(from_path, to_path, category, background, cuts, wh)

import cv2
import numpy as np
from skimage.filters import gaussian
from src.test import evaluate
import argparse

# 1  face
# 11 teeth
# 12 upper lip
# 13 lower lip
# 17 hair
table = {
    'hair': 11,
    'upper_lip': 12,
    'lower_lip': 13
}


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    # alpha = 1.5
    alpha = 1.2
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color  # [10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 11:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    return changed


if __name__ == '__main__':
    image_path = "/home/sindre/Documents/人脸解析/imgs/116.jpg"
    model = '/home/sindre/Documents/人脸解析/models/face_analysis.pth'
    image = cv2.imread(image_path)
    min_size = min(image.shape[:2])
    image = cv2.resize(image, (min_size, min_size))
    ori = image.copy()
    parsing = evaluate(image_path, model)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    parts = [table['hair'], table['upper_lip'], table['lower_lip']]
    colors = [[255, 255, 255], [20, 70, 180], [20, 70, 180]]

    for part, color in zip(parts, colors):
        image = hair(image, parsing, part, color)

    cv2.imshow('image', cv2.resize(ori, (512, 512)))
    cv2.imshow('color', cv2.resize(image, (512, 512)))

    cv2.waitKey(0)
    # cv2.destroyAllWindows()

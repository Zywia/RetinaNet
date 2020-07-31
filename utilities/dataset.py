import numpy as np
import torch
from PIL import Image
from matplotlib import patches
import matplotlib.pyplot as plt

def resize_512(img):
  sizes = img.size
  size = np.argmax(sizes)
  image_size = 512

  scale = image_size / sizes[size]
  if size:
    second_value = int(sizes[0] * scale)
    toI = img.resize( (second_value, image_size))
  else:
    second_value = int(sizes[1] * scale)
    toI = img.resize ((image_size, second_value))

  new_im = Image.new(img.mode, size= (image_size, image_size))
  new_im.paste(toI)
  return new_im

def mapToBoxes(list_of_mask, list_of_masks_numbers):
    list_of_boxes = []
    for mask, x in zip(list_of_mask, list_of_masks_numbers):
      num_objs = len(x)
      boxes = []

      masks = mask == x[:, None, None]

      for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])

      list_of_boxes.append(boxes)

    return list_of_boxes


def showImageWithBoxes(img, boxes):
  fig, ax = plt.subplots(1)
  ax.imshow(img)

  for x_min, y_min, x_max, y_max in boxes:
    width = x_max - x_min
    height = y_max - y_min
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

  plt.show()


def receptiveField(level):
  receptive_field = 1
  jump = 1
  kernel_size = 3
  stride = 2
  for x in range(level):
    receptive_field += (kernel_size - 1) * jump
    jump *= stride

  return receptive_field

def fromPyramidLvl(parameters, lvl):
  center_x, center_y, left_offset, right_offset, up_offset, down_offet = parameters
  center_x, center_y = [center_x * 2 ** lvl, center_y * 2 ** lvl]
  rF = receptiveField(lvl)
  left_offset, right_offset, up_offset, down_offset = [int(rF * x) for x in
                                                       [left_offset, right_offset, up_offset, down_offet]]

  left_offset += center_x - rF // 2
  up_offset += center_y - rF // 2

  right_offset = center_x + rF // 2 - right_offset
  down_offset = center_y + rF // 2 - down_offset

  return [left_offset, up_offset, right_offset, down_offset]


def intersection_over_union(boxA, boxB):
  # determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  # compute the area of intersection rectangle
  interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
  if interArea == 0:
    return 0
  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
  boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = interArea / float(boxAArea + boxBArea - interArea)

  # return the intersection over union value
  return iou


def boundingBoxFromPointAndPyramidLvl(y_on_pyramid_lvl, x_on_pyramid_lvl, lvl):
  scales = [2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]
  receptive_field = receptiveField(lvl)
  x = x_on_pyramid_lvl * 2 ** lvl
  y = y_on_pyramid_lvl * 2 ** lvl

  bounding_boxes = []

  for a in scales:
    x_min = int(round(x - receptive_field // 2 * a))
    x_max = int(round(x + receptive_field // 2 * a))

    y_min = int(round(y - receptive_field * a))
    y_max = int(round(y + receptive_field * a))
    bounding_boxes.append([x_min, y_min, x_max, y_max])

  return bounding_boxes


def calculateIoUOnPyramidLvl(bounding_boxes_of_an_images, lvl):
  image_height = 512
  height_on_pyramid_lvl = image_height // 2 ** lvl
  number_of_batches = len(bounding_boxes_of_an_images)
  pyramid_lvl = torch.zeros(number_of_batches,
                            height_on_pyramid_lvl,
                            height_on_pyramid_lvl, 3)

  cls = torch.zeros(number_of_batches,
                    height_on_pyramid_lvl,
                    height_on_pyramid_lvl, dtype=torch.long)
  for b, img in enumerate(bounding_boxes_of_an_images):
    for x in range(height_on_pyramid_lvl):
      for y in range(height_on_pyramid_lvl):
        for z, val in enumerate(boundingBoxFromPointAndPyramidLvl(y, x, lvl)):
          for w in img:
            pyramid_lvl[b, y, x, z] = max(intersection_over_union(w, val), pyramid_lvl[b, y, x, z])

  fclass = torch.zeros_like(pyramid_lvl)
  tclass = torch.ones_like(pyramid_lvl)

  return torch.where(pyramid_lvl < 0.5, fclass, tclass).view(-1, 3)


def toPyramidForTraining(bounding_boxes_of_an_images):
  return [calculateIoUOnPyramidLvl(bounding_boxes_of_an_images, x) for x in range(3, 8)]


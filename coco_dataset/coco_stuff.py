# -*- coding: utf-8 -*-

""" 从coco数据集中导出指定数据
"""

from pycocotools.coco import COCO
import os
import shutil
from lxml import etree
import cv2


def save_label_file(filename, ann_file, targets, shape):
    """ 保存VOC格式标注文件
    """
    if targets.__len__() == 0:
        os.remove(filename)
        return

    root = etree.Element('annotation')
    element = etree.SubElement(root, 'folder')
    element.text = 'image'
    element = etree.SubElement(root, 'filename')
    element.text = os.path.basename(filename)
    element = etree.SubElement(root, 'path')
    element.text = filename
    element = etree.SubElement(root, 'source')
    element2 = etree.SubElement(element, 'database')
    element2.text = 'Unknown'

    element = etree.SubElement(root, 'size')
    element2 = etree.SubElement(element, 'width')
    element2.text = str(shape[1])
    element2 = etree.SubElement(element, 'height')
    element2.text = str(shape[0])
    element2 = etree.SubElement(element, 'depth')
    element2.text = str(shape[2])

    element = etree.SubElement(root, 'segmented')
    element.text = '0'

    for target in targets:
        element = etree.SubElement(root, 'object')
        element2 = etree.SubElement(element, 'name')
        element2.text = target[5]
        element2 = etree.SubElement(element, 'pose')
        element2.text = 'Unspecified'
        element2 = etree.SubElement(element, 'truncated')
        element2.text = '0'
        element2 = etree.SubElement(element, 'difficult')
        element2.text = '0'

        element2 = etree.SubElement(element, 'bndbox')
        element3 = etree.SubElement(element2, 'xmin')
        element3.text = str(target[0])
        element3 = etree.SubElement(element2, 'ymin')
        element3.text = str(target[1])
        element3 = etree.SubElement(element2, 'xmax')
        element3.text = str(target[0] + target[2])
        element3 = etree.SubElement(element2, 'ymax')
        element3.text = str(target[1] + target[3])

        tree = etree.ElementTree(root)
    tree.write(ann_file, pretty_print=True)


def save_annotation_file(anns, cat_ids, file_name, ann_file):
    """ 从COCO标注文件中解析信息，并保存VOC标注文件
    """
    
    image = cv2.imread(file_name)
    targets = []
    for ann in anns:
        if not ann['category_id'] in cat_ids:
            continue
        target = []
        target.append(int(ann['bbox'][0]))
        target.append(int(ann['bbox'][1]))
        target.append(int(ann['bbox'][2]))
        target.append(int(ann['bbox'][3]))
        target.append(0)
        target.append('vehicle')

        if target[2] > 50 and target[3] > 50:
            targets.append(target)

    save_label_file(file_name, ann_file, targets, image.shape)


def create_dataset(coco, cat_ids, img_id, ann_ids, data_path, coco_imgs_path):
    """ 拷贝图片文件，并保存VOC标注文件
    """
    images_path = os.path.join(data_path, 'images')
    annotations_path = os.path.join(data_path, 'Annotations')
    if not os.path.exists(data_path):
        os.makedirs(images_path)
        os.makedirs(annotations_path)

    images = coco.loadImgs(img_id)
    anns = coco.loadAnns(ann_ids)

    image = images[0]

    img_file_name = image['file_name']
    ann_file_name = os.path.splitext(img_file_name)[0]
    ann_file_name += '.xml'
    img_dest_file = os.path.join(images_path, img_file_name)
    img_file_name = os.path.join(coco_imgs_path, img_file_name)
    ann_file_name = os.path.join(annotations_path, ann_file_name)

    if os.path.exists(img_file_name):
        shutil.copy(img_file_name, images_path)
        save_annotation_file(anns, cat_ids, img_dest_file, ann_file_name)


def coco_captions_data(data_type, cats_name, data_path):
    """ 解析COCO数据
    """
    # 数据路径
    coco_root = '/home/lz/hutpond/deeplearn/dataset/coco2017'
    annotation_file = os.path.join(coco_root, f'annotations/instances_{data_type}.json')
    coco_imgs_path = os.path.join(coco_root, data_type)

    # 为实例注释初始化COCO的API 
    coco = COCO(annotation_file)

    # get cat ids
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    cat_ids.clear()
    for cat in cats:
        if cat['name'] in cats_name:
            cat_ids.append(cat['id'])

    # get image ids
    img_ids_all = []
    for cat_id in cat_ids:
        img_ids = coco.getImgIds(catIds=cat_id)
        print(img_ids.__len__())
        for ids in img_ids:
            if ids not in img_ids_all:
                img_ids_all.append(ids)
    print(img_ids_all.__len__())

    for img_id in img_ids_all:
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        create_dataset(coco, cat_ids, img_id, ann_ids, data_path, coco_imgs_path)

if __name__ == '__main__':
    cats_name = ('car', 'bus', 'truck')
    data_path = '/home/lz/hutpond/deeplearn/project/car/dataset_0521/'
    coco_captions_data('val2017', cats_name, data_path)
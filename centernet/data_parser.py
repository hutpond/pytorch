#-*- coding: utf-8

""" 解析coco-annotation标注文件，适应CenterNet训练 """

import os
import json
import shutil

if __name__ == '__main__':
    """ """
    # read data from label file
    file_name = 'coco-1613789462.5181398.json'
    with open(file_name, 'r') as f:
        data = json.load(f)
        
    # train label file & valid label file
    data_train = {}
    data_valid = {}

    # info
    data_train_info = {}
    data_train_info['description'] = 'customer dataset format convert to COCO format'
    data_train_info['customer'] = 'http://cocodataset.org'
    data_train_info['version'] = '1.0'
    data_train_info['year'] = '2021'
    data_train_info['contributor'] = 'ezra'
    data_train_info['date_created'] = '2021/03/23'
    data_train['info'] = data_train_info
    data_valid['info'] = data_train_info

    # licenses
    data_train_license = {}
    data_train_license['url'] = 'https://www.apache.org/licenses/LICENSE-2.0.html'
    data_train_license['id'] = 1
    data_train_license['name'] = 'Apache License 2.0'
    data_train['licenses'] = [data_train_license]
    data_valid['licenses'] = [data_train_license]

    # categories
    data_train_category = {}
    data_category = data['categories'][0]
    data_train_category['id'] = data_category['id']
    data_train_category['name'] = data_category['name']
    data_train_category['supercategory'] = 'hand'
    data_train_category['keypoints'] = data_category['keypoints']
    data_train_category['skeleton'] = data_category['skeleton']
    data_train['categories'] = [data_train_category]
    data_valid['categories'] = [data_train_category]

    # images & annotaion
    imgs_count = data['images'].__len__()
    train_count = imgs_count * 2 // 3
    valid_count = imgs_count - train_count

    # train
    data_train_images = []
    data_train_annotations = []
    for i in range(train_count):
        data_image = data['images'][i]
        image = {}
        image['file_name'] = data_image['file_name'] 
        image['height'] = data_image['height'] 
        image['width'] = data_image['width'] 
        image['id'] = data_image['id'] 
        data_train_images.append(image)

        annotation = {}
        data_annotation = data['annotations'][i]
        annotation['id'] = data_annotation['id'] 
        if not 'num_keypoints' in data_annotation:
            continue
        annotation['num_keypoints'] = data_annotation['num_keypoints'] 
        annotation['keypoints'] = data_annotation['keypoints'] 
        annotation['area'] = data_annotation['area'] 
        annotation['iscrowd'] = data_annotation['iscrowd'] 
        annotation['image_id'] = data_annotation['image_id'] 
        annotation['bbox'] = data_annotation['bbox'] 
        annotation['category_id'] = 1 #data_annotation['category_id'] 
        annotation['segmentation'] = data_annotation['segmentation'] 
        data_train_annotations.append(annotation)

        shutil.move(image['file_name'], './train2017')
    data_train['images'] = data_train_images
    data_train['annotations'] = data_train_annotations

    data_valid_images = []
    data_valid_annotations = []
    for i in range(valid_count):
        i += train_count
        data_image = data['images'][i]
        image = {}
        image['file_name'] = data_image['file_name'] 
        image['height'] = data_image['height'] 
        image['width'] = data_image['width'] 
        image['id'] = data_image['id'] 
        data_valid_images.append(image)

        annotation = {}
        data_annotation = data['annotations'][i]
        annotation['id'] = data_annotation['id'] 
        if not 'num_keypoints' in data_annotation:
            continue
        annotation['num_keypoints'] = data_annotation['num_keypoints'] 
        annotation['keypoints'] = data_annotation['keypoints'] 
        annotation['area'] = data_annotation['area'] 
        annotation['iscrowd'] = data_annotation['iscrowd'] 
        annotation['image_id'] = data_annotation['image_id'] 
        annotation['bbox'] = data_annotation['bbox'] 
        annotation['category_id'] = 1 #data_annotation['category_id'] 
        annotation['segmentation'] = data_annotation['segmentation'] 
        data_valid_annotations.append(annotation)

        shutil.move(image['file_name'], './test2017')

    data_valid['images'] = data_valid_images
    data_valid['annotations'] = data_valid_annotations
    
    # save file
    file_name = 'person_keypoints_train2017.json'
    with open(file_name, 'w') as f:
        json.dump(data_train, f)

    file_name = 'person_keypoints_val2017.json'
    with open(file_name, 'w') as f:
        json.dump(data_valid, f)

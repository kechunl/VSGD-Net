import os, json, random, glob, math, cv2, argparse, pdb
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog

import numpy as np
from detectron2.structures import BoxMode
from pycocotools.mask import encode
from pycocotools.mask import decode as decode_mask
import matplotlib.pyplot as plt


class Mask:
    def __init__(self, mask_path, resize_factor=1):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if resize_factor != 1:
            mask_img = cv2.resize(mask_img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST)
        if mask_img.dtype == "uint8":
            mask_img = mask_img / 255 # 2-D index mask
        self.mask = mask_img


    def get_annos(self, dilation=False, dilate_iter=1):
        target = {}

        contours, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        masks = []
        boxes = []
        for i in range(0, len(contours)):
            if contours[i].size < 6:
                continue
            if contours[i].size % 2 != 0:
                continue

            if dilation:
                temp_mask = cv2.drawContours(np.zeros_like(self.mask), [contours[i]], -1, color=255, thickness=cv2.FILLED)
                temp_mask = cv2.dilate(temp_mask, np.ones((3,3), np.uint8), iterations=dilate_iter)
                temp_contours, _ = cv2.findContours(temp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # import pdb; pdb.set_trace()
                assert len(temp_contours) == 1
                ann_x, ann_y, ann_w, ann_h = cv2.boundingRect(temp_contours[0])
                box = [float(ann_x), float(ann_y), float(ann_x + ann_w - 1), float(ann_y + ann_h - 1)]
                boxes.append(box)
                masks.append(temp_mask/255)

                # obj = {
                #     "boxes": box,
                #     "bbox_mode": BoxMode.XYXY_ABS,
                #     "masks": [list(temp_contours[0].astype(np.float).reshape(-1))],
                #     "labels": 1,
                # }
            else:
                ann_x, ann_y, ann_w, ann_h = cv2.boundingRect(contours[i])
                box = [float(ann_x), float(ann_y), float(ann_x + ann_w - 1), float(ann_y + ann_h - 1)]
                boxes.append(box)
                temp_mask = cv2.drawContours(np.zeros_like(self.mask), [contours[i]], -1, color=255, thickness=cv2.FILLED)
                masks.append(temp_mask.astype(np.uint8)/255)

                # obj = {
                #     "boxes": box,
                #     "bbox_mode": BoxMode.XYXY_ABS,
                #     "masks": [list(contours[i].astype(np.float).reshape(-1))],
                #     "labels": 1,
                # }
            # objs.append(obj)
        target["boxes"] = boxes
        target["labels"] = np.ones((len(boxes)), dtype=np.int).tolist()
        target["masks"] = []
        for mask in masks:
            rle = encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            target["masks"].append(rle)
        # target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes)!=0 else np.array([])
        # target["iscrowd"] = np.zeros((len(boxes)), dtype=np.int).tolist()

        return target


def make_dicts(args, dataset='train'):
    '''
    Get MSCOCO format of data.

    :param args:
    :return: dataset_dicts
    '''
    dataset_dicts = []

    # Find patches path
    HE_patch_list = glob.glob(os.path.join(args.root_dir, dataset+'_A', '*.png'))
    assert len(HE_patch_list) > 0
    print('Get {} patches in {} set'.format(len(HE_patch_list), dataset))

    # Process patches to annotations and json info
    for img_idx, img_path in enumerate(tqdm(HE_patch_list, total=len(HE_patch_list))):
        record = {"file_name": img_path, "image_id": img_idx, "height": args.patch_size, "width": args.patch_size}

        mask_path = os.path.join(args.root_dir, '{}_mask'.format(dataset), os.path.basename(img_path))
        assert os.path.exists(mask_path)

        mask = Mask(mask_path, resize_factor=1)
        record["target"] = mask.get_annos(dilation=args.dilate, dilate_iter=1)
        dataset_dicts.append(record)

    if args.dilate:
        with open(os.path.join(args.root_dir, dataset + '_dilated.json'), 'w') as f:
            json.dump(dataset_dicts, f)
    else:
        with open(os.path.join(args.root_dir, dataset + '.json'), 'w') as f:
            json.dump(dataset_dicts, f)


def get_dicts(args, dataset, suffix='.json'):
    json_file = os.path.join(args.root_dir, dataset + suffix)
    with open(json_file) as f:
        dataset_dicts = json.load(f)

    return dataset_dicts

def get_box_mask_stats(args, dataset, suffix='.json'):
    data_list = []
    for ds in dataset:
        with open(os.path.join(args.root_dir, ds+suffix), 'r') as f:
            data_list.extend(json.load(f))
    nuclei_area = []
    for data in data_list:
        masks = data['target']['masks']
        if len(masks) == 0:
            continue
        areas = np.sum(np.transpose(decode_mask(masks), (2,0,1)), axis=(1,2)) # N*256*256
        nuclei_area.extend(areas.tolist())
    plt.hist(nuclei_area, bins=np.max(nuclei_area))
    plt.savefig(os.path.join(args.root_dir, 'nuclei_area_hist.png'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare melanocyte dataset in coco format')
    parser.add_argument('--root_dir', default=None, type=str, help="folder of images")
    parser.add_argument('--num_classes', default=2, type=int, help="foreground + background")
    parser.add_argument('--patch_size', default=256, type=int, help="size of patches")
    parser.add_argument('--dilate', action='store_true', help="whether to dilate the melanocyte")

    args = parser.parse_args()

    # get_box_mask_stats(args, ['train', 'val', 'test'])
    
    for d in ['val', 'train', 'test']:
        make_dicts(args, d)

    # for d in ["train", "val", "test"]:
    #     DatasetCatalog.register("melanocyte_" + d, lambda d=d: get_melanocyte_dicts(args, d))
    #     MetadataCatalog.get("melanocyte_" + d).set(thing_classes=["melanocyte"])
    # melanocyte_metadata = MetadataCatalog.get("melanocyte_train")
    # dataset_dicts = get_dicts(args, 'train')

    # example_dir = os.path.join(args.root_dir, 'example_dilated' if args.dilate else 'example')
    # os.makedirs(example_dir, exist_ok=True)
    # for i, d in enumerate(random.sample(dataset_dicts, 50)):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=melanocyte_metadata, scale=1)
    #     out = visualizer.draw_dataset_dict(d, draw_labels=False)
    #     cv2.imwrite(os.path.join(example_dir, '{}.png'.format(str(i))), cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGB2BGR))
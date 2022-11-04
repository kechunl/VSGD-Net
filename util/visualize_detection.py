import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from .util import tensor2im
import pdb

def display_instances(image, boxes, scores=None, masks=None, mask_thresh=0.1):
    """
    image (Tensor): Tensor of shape (C x H x W) and dtype uint8.
    boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
            the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
            `0 <= ymin < ymax < H`.
    labels (Tensor): Tensor of size (N,) containing the labels of bounding boxes.
    scores (Tensor): Tensor of size (N,) containing the scores of instances.
    masks (Tensor): Tensor of shape (N, H, W) or (H, W) and dtype bool.
    """
    # Number of instances
    # N = boxes.shape[0]
    # if not N:
    #     print("\n*** No instances to display *** \n")
    # else:
    #     assert boxes.shape[0] == masks.shape[0] == labels.shape[0]

    if len(boxes.shape) > 3:
        boxes = torch.squeeze(boxes)

    if len(image.shape) > 3:
        image = torch.squeeze(image)

    # check image dtype
    if torch.max(image) <= 1:
        image = torch.Tensor(tensor2im(image)).to(torch.uint8).permute(2,0,1)

    # generate texts and draw bounding boxes
    # if scores is not None:
    #     texts = ['%.2f' % (score.detach().numpy()) for score in scores]
    #     image_drawn = draw_bounding_boxes(image, boxes, texts, width=1, font='arial.ttf', font_size=1)
    # else:
    #     image_drawn = draw_bounding_boxes(image, boxes, width=1)
    image_drawn = draw_bounding_boxes(image, boxes, width=1)

    # draw masks
    if masks is not None:
        masks = torch.squeeze(masks)
        if masks.shape[0] != 0 and len(masks.shape) == 3:
            image_drawn = draw_segmentation_masks(image_drawn, masks>mask_thresh, alpha=0.7)

    return image_drawn
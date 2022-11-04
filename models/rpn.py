import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import functools
from torch.autograd import Variable
from typing import List, Optional, Dict, Tuple, cast

from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.ops.focal_loss import sigmoid_focal_loss

class RegionProposalNetwork_FocalLoss(RegionProposalNetwork):
	def __init__(self,
		anchor_generator: AnchorGenerator,
		head: nn.Module,
		# Faster-RCNN Training
		fg_iou_thresh: float,
		bg_iou_thresh: float,
		batch_size_per_image: int,
		positive_fraction: float,
		# Faster-RCNN Inference
		pre_nms_top_n: Dict[str, int],
		post_nms_top_n: Dict[str, int],
		nms_thresh: float,
		score_thresh: float,
		focal_loss_gamma: float):
			super().__init__(anchor_generator, 
				head, 
				fg_iou_thresh, 
				bg_iou_thresh,
				batch_size_per_image,
				positive_fraction,
				pre_nms_top_n,
				post_nms_top_n,
				nms_thresh,
				score_thresh)
			self.focal_loss_gamma = focal_loss_gamma

	def compute_loss(self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor]):
			"""
			Args:
				objectness (Tensor)
				pred_bbox_deltas (Tensor)
				labels (List[Tensor])
				regression_targets (List[Tensor])
			Returns:
				objectness_loss (Tensor)
				box_loss (Tensor)
			"""

			sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
			sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
			sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

			sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

			objectness = objectness.flatten()

			labels = torch.cat(labels, dim=0)
			regression_targets = torch.cat(regression_targets, dim=0)

			box_loss = (
				F.smooth_l1_loss(
					pred_bbox_deltas[sampled_pos_inds],
					regression_targets[sampled_pos_inds],
					beta=1 / 9,
					reduction="sum",
				)
				/ (sampled_inds.numel())
			)

			objectness_loss = sigmoid_focal_loss(objectness[sampled_inds], labels[sampled_inds], gamma=self.focal_loss_gamma, reduction='mean')

			return objectness_loss, box_loss
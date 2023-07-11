import math
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchnet.meter import APMeter

class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return (
            self.__class__.__name__
            + " (size={size}, interpolation={interpolation})".format(
                size=self.size, interpolation=self.interpolation
            )
        )

class APMeter2(APMeter):

    def __init__(self, difficult_examples=False):
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.UntypedStorage())
        self.targets = torch.LongTensor(torch.UntypedStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert (
                output.dim() == 2
            ), "wrong output size (should be 1D or 2D with one column \
                per class)"
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert (
                target.dim() == 2
            ), "wrong target size (should be 1D or 2D with one column \
                per class)"
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(
                1
            ), "dimensions for output should match previously added examples."

        # make sure storage is of sufficient size
        # if self.scores.storage().size() < self.scores.numel() + output.numel():
        #     new_size = math.ceil(self.scores.storage().size() * 1.5)
        #     self.scores.storage().resize_(int(new_size + output.numel()))
        #     self.targets.storage().resize_(int(new_size + output.numel()))
        if self.scores.untyped_storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.untyped_storage().size() * 1.5)
            self.scores.untyped_storage().resize_(int(new_size + output.numel()))
            self.targets.untyped_storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0

        # print(offset + output.size(0), output.size(1))
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]

            # compute average precision
            ap[k] = self.average_precision(
                scores, targets, self.difficult_examples
            )
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):
        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.0
        total_count = 0.0
        precision_at_i = 0.0
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        return precision_at_i

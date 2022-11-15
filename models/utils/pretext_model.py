from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from datasets import get_dataset
from models.utils.continual_model import ContinualModel
from utils.no_bn import bn_track_stats
from utils.pretext import get_pretext_task
from utils.pretext.jigsaw import JigsawPretext
from utils.pretext.rotation import RotationPretext

def features_hook(module, input, output):
    """
    Hook to save the output of the features
    """
    module.partial_features = output

class PretextModel(ContinualModel):
    
    def get_ptx_transform(self):
        if self.args.pretext == "rotation":
            return RotationPretext(self.args)
        elif self.args.pretext == "jigsaw":
            return JigsawPretext(self.args)
        else:
            raise NotImplementedError

    def __init__(self, backbone, loss, args, transform):
        super(PretextModel, self).__init__(backbone, loss, args, transform)
        assert args.pretext is not None, "Pretext model needs pretext task"

        self.pretexter = get_pretext_task(args)
        self._partial_train_transform = transforms.Compose(self.train_transform.transforms[-1:])

        self.current_task = 0
        self.eye = torch.eye(self.pretexter.get_num_pretext_classes())
        
    def init_ptx_heads(self, args, backbone) -> nn.Module:
        bname = type(backbone).__name__.lower()
        assert "resnet" in bname or "dualnetnet" in bname, "Ptx heads only implemented for resnet, got {}".format(type(backbone).__name__)

        dset = get_dataset(args)
        x = dset.get_data_loaders()[0].dataset[0][0]
        with torch.no_grad():
            features_shape = backbone(x.unsqueeze(0), returnt="full")[-1][-2].shape

        self.child_index_start, self.child_index_end = 6, 7
        
        backbone.ptx_net = nn.Sequential(
            *deepcopy(list(backbone.children())[self.child_index_start:self.child_index_end]), 
            nn.AvgPool2d(features_shape[-1]),
            nn.Flatten(),
            nn.Linear(backbone.nf * 8 * backbone.block.expansion, get_pretext_task(args).get_num_pretext_classes())
            )

        self.hook = list(backbone.children())[self.child_index_start-1].register_forward_hook(features_hook)

        return backbone

    def get_ptx_outputs(self, ptx_inputs, net=None):
        net = self.net if net is None else net
        with bn_track_stats(net, False):
            _ = net(ptx_inputs)
        c = list(net.children())
        stream_partial_features = c[self.child_index_start-1].partial_features
        return net.ptx_net(stream_partial_features)

    def get_ce_pret_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def get_distill_pret_loss(self, logits, buf_logits, buf_labels):
        return F.mse_loss(logits, buf_logits)
from mindspore.nn import LossBase
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops


class KLDivLoss(LossBase):

    def construct(self, base, target,reduction = 'mean'):
        #_check_is_tensor('logits', base, self.cls_name)
        #_check_is_tensor('labels', target, self.cls_name)
        
        Log = ops.Log()
        log_tar = Log(target)
        
        x = target*(log_tar-base)
        
        if reduction == "mean":
            x = x.mean()
        if reduction == "sum":
            x = x.sum()
        return self.get_loss(x)

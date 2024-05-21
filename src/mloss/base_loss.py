from abc import ABC, abstractmethod


class BaseLoss(ABC):
    """Base class for losses
    """

    def __init__(self, device=None):
        self.device = device

    @abstractmethod
    def __call__(self, data, model, groups=None):
        """Calculate loss for a batch of data

        Args:
            data: list of tuple of tensors [(x1,y1),(x2,y2),...(xk,yk)]. 
                  If target x is needed, the last (xk,yk) is from target domain (yk is not used)
            model (torch nn): a pytorch model
            groups: (1) if None, (xi,yi) in data is a batch from ith domain
                    (2) otherwise groups should be a tensor. 
                        groups[i] identifies which domain (x1[i],y1[i]) is from. 
                        k needs to be 1 (target x not used) or 2 (target x used).
        """
        pass
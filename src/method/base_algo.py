"""
Any DA algorithm is a subclass of BaseAlgo
which implements an domain adaptation (DA) method
by calling other modules
a DA algorithm consists of 3 components
- intialize with model, loss + penalty, trainer etc
- fit model (training)
- prediction (testing)
"""

from abc import ABC, abstractmethod

class BaseAlgo(ABC):
    """Base class for a DA algorithm"""

    def __init__(self, device, model):
        pass

    @abstractmethod
    def fit(self, dataloaders, source, target):
        pass
  
  
    def predict(self, x):
        """Use the learned model to predict labels on the fresh data (1 point)
          requires self.model

        Args:
            x (tensor): one data point
        Returns:
            probability vector for x
        """
        output = self.model(x)
        return output
  
  
    def predict_dataloader(self, dataloader, prop=1.):
        """Use the learned model to predict labels on a dataloader
          requires self.model, self.trainer

        Args:
            dataloader (): pytorch dataloader containing data points
        Returns:
            ypred, accuracy:
        """
        return self.trainer.predict(dataloader, prop=prop)
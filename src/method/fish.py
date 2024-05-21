import copy
from .generic_algo import GenericAlgo
from .. import mloss


class Fish(GenericAlgo):
    """
    Fish method
    """
    def __init__(self, device, model, meta_lr=1e-1, meta_steps=None, 
                 loss_type='CrossEntropyLoss', optimizer='Adam', **kwargs):
        """default initialization

        Args:
            device (device): cuda or cpu
            model (mmodel): model
            meta_lr (float): meta learning rate.
            meta_steps (int): number of meta steps.
            optimizer (torch.optim, optional): optimization. Defaults to None.
            lr (float, optional): learning rate. Defaults to 1e-3.
        """
        
        self.meta_steps = meta_steps
        self.meta_lr = meta_lr
        self.step = 0
        self.loss_type = loss_type

        super().__init__(device=device,
                         model=model,
                         loss=mloss.LossNLL(device, loss_type),
                         optimizer=optimizer,
                         **kwargs
                         )
    
    def process_batch(self, data, groups=None, **kwargs):
        
        if self.meta_steps is None:
            self.meta_steps = len(data)
        
        if groups is None:
            nb_envs = len(data)
            if self.meta_steps is None:
                self.meta_steps = nb_envs
                
            for i in range(nb_envs):
                
                if self.step % self.meta_steps == 0:
                    self.inner_init()
                
                data_i = [data[i]]
                loss_i = self.loss(data, self.model, None, **kwargs)
                self.optimizer_inner.zero_grad()
                loss_i.backward()
                self.optimizer_inner.step()
                self.step += 1
                
                if self.step % self.meta_steps == 0:
                    self.meta_update()

        else:
            
            group_ids = torch.unique(groups)
            if self.meta_steps is None:
                self.meta_steps = len(group_ids)  
            
            x, y = data[0].to(self.device), data[1].to(self.device)
            outputs = self.model(x)
            loss = self.criterion(reduction='none', **kwargs)(outputs, y)
            
            for idx in group_ids:
                if self.step % self.meta_steps == 0:
                    self.inner_init()

                loss_i = loss[groups==idx].mean()
                self.optimizer_inner.zero_grad()
                loss_i.backward()
                self.optimizer_inner.step()
                self.step += 1
                
                if self.step % self.meta_steps == 0:
                    self.meta_update()
    
        return loss_i
    
    def inner_init(self):
        self.model_inner = copy.deepcopy(self.model)
        self.optimizer_inner = self.optimizer(self.model_inner.parameters(), **self.optimizer_args_dict)

    def meta_update(self):
        model_meta_dict = self.model.state_dict()
        model_inner_dict = self.model_inner.state_dict()
        for key in model_meta_dict:
            model_meta_dict[key] += self.meta_lr * (model_inner_dict[key] - model_meta_dict[key])
        self.model.load_state_dict(model_meta_dict)
    
    def __str__(self):
        return self.__class__.__name__
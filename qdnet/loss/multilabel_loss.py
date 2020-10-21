
import torch  
from torch import nn  
 
class WeightedMultilabel(nn.Module):  
    def __init__(self, weights: torch.Tensor):  
        super(WeightedMultilabel, self).__init__()  
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')  
        self.weights = weights  
  
    def forward(self, outputs, targets):  
        loss = self.cerition(outputs, targets)  
        return (loss * self.weights).mean() 


class Loss(object):
    
    def __init__(self, loss_type="bce", w=None):
        # w=torch.tensor([10,2,15,20],dtype=torch.float)  
        if loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss(w)
        elif loss_type == "wml":
            self.criterion = WeightedMultilabel(w)  
        elif loss_type == "mlsl":
            self.criterion = nn.MultiLabelSoftMarginLoss(w)  
        else:
            raise NotImplementedError()
    
    def __call__(self, model, images, targets):
        outputs = model(images)
        loss = self.criterion(outputs, targets)



# 验证三个函数是等价的
if __name__ == '__main__':

    x=torch.randn(3,4)  
    y=torch.randn(3,4)  
    #损失函数对应类别的权重
    w=torch.tensor([10,2,15,20],dtype=torch.float)  
    #测试不同的损失函数
    criterion_BCE=nn.BCEWithLogitsLoss(w)  
    criterion_mult=WeightedMultilabel(w)  
    criterion_mult2=nn.MultiLabelSoftMarginLoss(w)  
  
    loss1=criterion_BCE(x,y)  
    loss2=criterion_mult(x,y)  
    loss3=criterion_mult2(x,y)  
  
    print(loss1)  
    print(loss2)  
    print(loss3)  
  
    # tensor(7.8804)  
    # tensor(7.8804)  
    # tensor(7.8804)
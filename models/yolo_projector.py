from turtle import forward
from torch import nn

from models.common import Conv

NUM_CHANNELS = 8


class ProjectorWrapper(nn.Module):
    def __init__(self, projector) -> None:
        super().__init__()
        self.projector = projector
        self.pointwise = nn.Conv2d(projector.out_ch, 3, kernel_size=1, stride=1)
        
    def forward(self, x):
        return self.pointwise(self.projector(x))

class ProjectorDefault(nn.Sequential):
    def __init__(self) -> None:
        self.out_ch = NUM_CHANNELS
        # TODO: replace CBR with Conv from common.py
        module_list = [
            Conv(3, NUM_CHANNELS, k=5, s=1),
            Conv(NUM_CHANNELS, NUM_CHANNELS, k=5, s=2),
            Conv(NUM_CHANNELS, NUM_CHANNELS, k=5, s=1),
            Conv(NUM_CHANNELS, NUM_CHANNELS, k=5, s=2),
        ]  
        
        super().__init__(*module_list)
        

class Projector2(nn.Sequential):
    def __init__(self) -> None:
        self.out_ch = NUM_CHANNELS
        
        module_list = [
            Conv(3, NUM_CHANNELS, k=5, s=2),
            Conv(NUM_CHANNELS, NUM_CHANNELS, k=5, s=2),
        ] 
        
        super().__init__(*module_list)
        
        
class Projector1(nn.Sequential):
    def __init__(self) -> None:
        self.out_ch = NUM_CHANNELS
        
        module_list = [
            Conv(3, NUM_CHANNELS, k=5, s=4),
        ] 
        
        super().__init__(*module_list)
        
        
class ProjectorResNet(nn.Module):
    def __init__(self, pretrained=False) -> None:
        super().__init__()
        
        self.out_ch = 64
        from torchvision.models import resnet50
        resnet = resnet50(pretrained=pretrained)
        self.resnet_stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
    
    def forward(self, x):
        return self.resnet_stem(x)
    
class ProjectorResNetPretrained(ProjectorResNet):
    def __init__(self) -> None:
        super().__init__(True)
        
class Projector1_7(nn.Sequential):
    def __init__(self) -> None:
        self.out_ch = NUM_CHANNELS
        
        module_list = [
            Conv(3, NUM_CHANNELS, k=7, s=4),
        ] 
        
        super().__init__(*module_list)
        
class Projector1_9(nn.Sequential):
    def __init__(self) -> None:
        self.out_ch = NUM_CHANNELS
        
        module_list = [
            Conv(3, NUM_CHANNELS, k=9, s=4),
        ] 
        
        super().__init__(*module_list)
        
class Projector1_11(nn.Sequential):
    def __init__(self) -> None:
        self.out_ch = NUM_CHANNELS
        
        module_list = [
            Conv(3, NUM_CHANNELS, k=11, s=4),
        ] 
        
        super().__init__(*module_list)
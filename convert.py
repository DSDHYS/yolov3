
import torch
import torchvision
from nets.yolov3 import YoloBody
from utils.config import Config
 
model=YoloBody(Config)
state_dict = torch.load("Epoch-3.pth")
#print(state_dict)
model.load_state_dict(state_dict,False)
model.eval()
 
x = torch.rand(1,3,128,128)
ts = torch.jit.trace(model, x)
ts.save('yolo.pt')

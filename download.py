import torch
import torchvision

# Download the SAM model weights
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
torch.save(model.state_dict(), 'sam_model.pth')
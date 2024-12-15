import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch
from PIL import Image
import PIL
import numpy as np

from config import get_config
import os
        
                
class StyleRepresentation(nn.Module):
    
    def __init__(self, num_filters):
        super().__init__()
        self.n = num_filters
        
    # x = F: (B, N, H, W) -> (B, N, N)
    def forward(self, x):
        # Input shape: (N, H, W)
        B, N, H, W = x.shape  # Extract dimensions
        
        # Reshape to (B, N, M), where M = H * W
        x = x.reshape(B, N, H * W)
        
        # Matrix multiplication to compute (B, N, N)
        G = torch.matmul(x, x.transpose(1, 2))  # Transpose last two dims
        
        return G
    
class StyleLearner(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # retrieve and modify base pretrained model to align with paper specifications (VGG 19 layers)
        self.sections = self._edit_base_model(models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1))
        
        # maps # of sections completed -> num_filters
        self.NUM_FILTERS_MAP = {
            0 : 64,
            1 : 128,
            2 : 256,
            3 : 512,
            4 : 512,
            5 : 512
        }
    
    def forward(self, x, end_at=6):
        
        # hold the output feature map and correlation matrix (style rep) of each block (conv1, conv2, conv3, etc)
        contents = []
        styles = []
        
        feature_map = x
        
        for conv_block in self.sections:
            # end_at to 
            if end_at == len(contents):
                break
            
            # initialize style representation layer with correct number of filters
            style_reprentation = StyleRepresentation(num_filters=self.NUM_FILTERS_MAP[len(contents)])
            
            # run the input through the pipeline
            feature_map = conv_block(feature_map)
            
            # create style gram using feature map outputted from section pipeline
            style_gram = style_reprentation(feature_map)
            
            # append this layer's calculated style and content to respective lists
            contents.append(feature_map)
            styles.append(style_gram)
        
        
        return contents, styles
    
    
    def _edit_base_model(self, model):
        
        # remove feed-forward layers
        model.classifier = nn.Identity()
        
        # initialize variables
        section_start = 0
        conv_list = []
        
        # convert nn.Sequential to nn.ModuleList to modify base model
        layers = nn.ModuleList(model.features.children())
        
        for name, module in layers.named_children():
            if isinstance(module, nn.MaxPool2d):
                
                # collecting 'sections' of VGG (5 conv blocks + 1 avgpool layer)
                # Convert layer index (name) to integer for slicing
                i = int(name)
                
                # collect modules from last pool block to now (excluding this pool block)
                conv_section = layers[section_start:i]
                section_start = i
                
                # add section to list
                conv_list.append(nn.Sequential(*conv_section))
                
                # ensuring avg-pool and max-pool initializations are the same
                kernel_size = module.kernel_size
                stride = module.stride
                padding = module.padding
                ceil_mode = module.ceil_mode
                
                # replace max-pool with average pool
                setattr(layers, name, nn.AvgPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    ceil_mode=ceil_mode
                ))
            
        # add final avgpool to list
        conv_list.append(nn.Sequential(model.avgpool))
                
        # check to make sure list length is correct
        assert len(conv_list) == 6
        
        # returns modulelist of convolution "sections"
        return nn.ModuleList(conv_list)



# the only preprocessing the original VGG paper uses is subtracting the 
# training mean, which is all we will do here as well
class Img():
    
    # preprocess image before forward pass into model
    @staticmethod
    def process(image, crop=True, scale=True):
        # Crop to a 1:1 aspect ratio
        if crop:
            width, height = image.size
            if width > height:
                margin = (width - height) // 2
                left, upper, right, lower = margin, 0, width - margin, height
            else:
                margin = (height - width) // 2
                left, upper, right, lower = 0, margin, width, height - margin
            image = image.crop((left, upper, right, lower))
            # ensure that image is square
            assert abs(image.size[0] - image.size[1]) <= 1

        # Resize the image to 224x224
        if scale:
            image = image.resize((224, 224), Image.LANCZOS)

        # convert to tensor (ToTensor scales to [0,1])
        to_tensor = transforms.ToTensor()
        tensor = to_tensor(image)  # shape: (3, 224, 224), values in [0,1]

        # Scale back to [0,255]
        tensor = tensor * 255.0

        # subtract the VGG mean for each channel for normalization
        VGG_MEAN = torch.tensor([123.68, 116.779, 103.939]).reshape(3,1,1)
        tensor = tensor - VGG_MEAN

        # add a batch dimension: (1, 3, 224, 224)
        tensor = tensor.unsqueeze(0)

        return tensor
    
    # saves tensor as image in given path
    def save_tensor_as_image(img_tensor, path):
        # Ensure the directory exists
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

        # Make sure we are not tracking gradients and work on a CPU copy
        img_tensor = img_tensor.detach().cpu().clone()

        # If there's a batch dimension (N, C, H, W), remove it
        if img_tensor.dim() == 4 and img_tensor.size(0) == 1:
            img_tensor = img_tensor.squeeze(0)

        # VGG mean values
        VGG_MEAN = torch.tensor([123.68, 116.779, 103.939]).reshape(3,1,1)

        # Add the mean back
        img_tensor = img_tensor + VGG_MEAN

        # Clamp values to [0,255]
        img_tensor = torch.clamp(img_tensor, 0, 255)

        # Convert to uint8
        img_tensor = img_tensor.byte()

        # Convert from (C,H,W) to (H,W,C)
        img_array = img_tensor.permute(1, 2, 0).numpy()

        # Convert to PIL Image
        image = Image.fromarray(img_array)
        image.save(path)
        
        
    
    @staticmethod
    def fromtensor(img_tensor):
        # Ensure the tensor is on CPU and detached from the computation graph
        img_tensor = img_tensor.cpu().detach()
        array = img_tensor.permute(1, 2, 0).numpy()

        # returns a an image given a torch tensor
        return Image.fromarray(array.astype(np.uint8))
        
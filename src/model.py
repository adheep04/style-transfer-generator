import torch
import torchvision.models as models
import torch.nn as nn

        
                
class StyleRepresentation(nn.Module):
    
    def __init__(self, num_filters):
        super().__init__()
        self.n = num_filters
        
    # x = F: (h, w, N) -> (N, N)
    def forward(self, x):
        
        # (h, w, N) -> (N, M)
        x = x.reshape(-1, x.shape[2]).transpose(1, 0)
        
        # F^l (N, M) * F^l^T (M, N) = G (N, N)
        return x.dot(x.T)
    
    
    
class StyleLearner(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # retrieve and modify base pretrained model to align with paper specifications (VGG 19 layers)
        self.sections = self.edit_base_model(models.vgg19(pretrained=True))
        
        # maps # of sections completed -> num_filters
        self.n_filters_map = {
            0 : 64,
            1 : 128,
            2 : 256,
            3 : 512,
            4 : 512,
        }
    
    def forward(self, x, end_at=5):
        
        # hold the output feature map and correlation matrix (style rep) of each block (conv1, conv2, conv3, etc)
        contents = []
        styles = []
        
        feature_map = None
        
        for conv_block in self.sections:
            
            # end_at to 
            if end_at == len(contents):
                break
            
            # initialize style representation layer with correct number of filters
            style_reprentation = StyleRepresentation(num_filters=self.n_filters_map[len(contents)])
            
            # run the input through the pipeline
            feature_map = conv_block(x)
            
            # create style gram using feature map outputted from section pipeline
            style_gram = style_reprentation(feature_map)
            
            # append next version of style and content to respective lists
            contents.append(feature_map)
            styles.append(style_gram)
            
        return contents, styles
    
    
    def edit_base_model(self, model):
        
        # remove feed-forward layers
        model.classifier = nn.Identity()
        
        # initialize variables
        section_start = 0
        conv_list = []
        
        # convert nn.Sequential to nn.ModuleList to change it
        layers = nn.ModuleList(model.features.children())
        
        for name, module in layers.named_children():
            if isinstance(module, nn.MaxPool2d):
                
                # collecting 'sections' of VGG (5 conv blocks)
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
                
        # check to make sure list length is correct
        assert len(conv_list) == 5
        
        return conv_list


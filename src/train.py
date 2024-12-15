import torch.nn as nn
import torch
from model import Img, StyleLearner
from PIL import Image
from config import get_config
from torch.optim import lr_scheduler


# global constants
device = torch.device('cuda')
config = get_config()


# given learnable image parameters and the content repr of the original
# returns the prediction F image content
def pred(model, img_params):
    
    # shallow copy of learnable image ensures gradients aren't affected
    img_tensor = img_params.clone().float().to(device)
    assert (1, 3, 224, 224) == img_tensor.shape, f"Shape mismatch: {img_tensor.shape} should be (1, 3, 224, 224)"
    
    return model(img_tensor)

# the following loss class is not vectorized and may seem inefficient, however
# designing it in the following way helped me understand it
class StyleContentLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mse = nn.MSELoss()
        
        # n as layers progress
        self.n = {
            0 : 64,
            1 : 128,
            2 : 256,
            3 : 512,
            4 : 512,
            5 : 512,
        }
        
        # m as layers progress
        self.m = {
            0 : 224,
            1 : 112,
            2 : 56,
            3 : 28,
            4 : 14,
            4 : 7,
        }
        
    # computes total loss
    def forward(self, pred_content, pred_styles, true_content, true_styles, alpha=config['alpha'], beta=config['beta'], style_weights=config['wl']):
        
        # alpha * content_loss + beta * style_loss
        return alpha * self.mse(pred_content, true_content) + beta * self.total_style_loss(pred_styles, true_styles, style_weights)
        
    # returns the style layer loss given 1) pred x 2) label a, 3) current layer number
    def layer_style_loss(self, x, a, i):
        
        # general error calculation
        style_error = torch.sum((x - a) ** 2)
        
        # normalize error using n and m
        loss_normalization = (4*(self.n[i]**2)*(self.m[i]**2))
        return style_error/loss_normalization
    
    def total_style_loss(self, x_list, a_list, w):
        w = torch.tensor(w)
        style_loss = 0
        for i in range(len(x_list)):
            style_loss += self.layer_style_loss(x_list[i], a_list[i], i)
        
        return style_loss

# content_image, style_image -> stylized content 
def replicate_style(
    content_img,
    style_imag, 
    steps=config['steps'], 
    loss_fn=StyleContentLoss(), 
    content_layer=config['content_layer'], 
    style_layer=config['style_layer'], 
    trial=config['trial'],
    lr=config['lr'], 
    factor=config['scheduler'],
    patience=config['patience'],
    ):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # generate blank brown image as vectors
    pred_params = torch.rand(1, 3, 224, 224, device=device) 
    
    # Make it learnable
    pred_params = torch.nn.Parameter(pred_params)  
    
    # process and move img_content and img_style to GPU
    content_img_tensor = Img.process(content_img).to(device)
    style_img_tensor = Img.process(style_imag).to(device)
    
    # initialize models
    model_content = StyleLearner().to(device) # to extract photograph content
    model_style = StyleLearner().to(device) # to extract painting style
    
    # getting content representation from content image from given layer in model
    output_c = model_content(content_img_tensor)
    true_content = output_c[0][content_layer].detach()
    
    # getting style representations from style image UP TO AND INCLUDING GIVEN LAYER (for total style loss calculation)
    output_s = model_style(style_img_tensor)
    
    # detach each layer's style representation in the list individually
    true_styles = [style.detach() for style in output_s[1][0:style_layer+1]]
    
    # initialize optimizer
    optimizer = torch.optim.Adam([pred_params], lr=lr)
    
    # learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    
    # closure function performs per-step grad descent tasks
    def closure():
        # reset gradients
        optimizer.zero_grad()
        
        # content prediction
        pred_content = pred(model_content, pred_params)[0][content_layer]
        
        # style prediction
        pred_styles = pred(model_style, pred_params)[1][0:style_layer+1]
        
        # calculate loss
        loss = loss_fn(pred_content, pred_styles, true_content, true_styles)
            
        # backpropagate and calculate gradients
        loss.backward()
        
        # update scheduler
        scheduler.step(loss.item())
        return loss
    
    for step in range(steps):
        
        # perform backpropogation and get loss
        loss = optimizer.step(closure)
        
        if step%20 == 0:
            print()
            print(f'{step} / {steps}: ')
            print(f'LOSS: {loss.item():.3f}')
            print(f'NORM: {torch.norm(pred_params.grad):.3f}')
        
        if step%20 == 0:
            Img.save_tensor_as_image(
                img_tensor=pred_params.clone().float(),
                path=f'trials/trial_{trial}/im_{step//20}.png'
                )
        
    return pred_params.float()

# from blank_image -> content
def content_recreation(
    img_content,
    steps=config['steps'], 
    loss_fn=nn.MSELoss(), 
    content_layer=config['content_layer'], 
    trial=config['trial'],
    lr=config['lr'], 
    factor=config['scheduler'],
    patience=config['patience'],
    ):
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # generate blank brown image as vectors
    pred_params = torch.rand(1, 3, 224, 224, device=device) 
    
    # Make it learnable
    pred_params = torch.nn.Parameter(pred_params)  
    
    # process and move img_content to GPU
    img_content = Img.process(img_content).to(device)
    
    # initialize models
    model_content = StyleLearner().to(device) # to extract photograph content
    
    # getting content representation from content image from given layer in model
    output_c = model_content(img_content)
    true_content = output_c[0][content_layer].detach()
    
    # initialize optimizer
    optimizer = torch.optim.Adam([pred_params], lr=lr)
    
    # learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    
    def closure():
        # reset gradients
        optimizer.zero_grad()
        
        # content prediction
        pred_content = pred(model_content, pred_params)[0][content_layer]
        
        loss = loss_fn(pred_content, true_content)
        loss.backward()
        scheduler.step(loss.item())
        return loss
    
    for step in range(steps):
        
        # perform backpropogation and get loss
        loss = optimizer.step(closure)
        
        # moniter learning stats
        if step%20 == 0:
            print()
            print(f'{step} / {steps}: ')
            print(f'LOSS: {loss.item():.3f}')
            print(f'NORM: {torch.norm(pred_params.grad):.3f}')
        
        if step%20 == 0:
            Img.save_tensor_as_image(
                img_tensor=pred_params.clone().float(),
                path=f'trials/trial_{trial}/im_{step//20}.png'
                )
        
    return pred_params.float()

replicate_style(Image.open(config['content_path']), Image.open(config['style_path']))

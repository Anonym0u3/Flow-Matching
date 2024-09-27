import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        # Check if input and output channels are the same for the residual connection
        self.same_channels = in_channels == out_channels

        # Flag for whether or not to use residual connection
        self.is_res = is_res

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # GELU activation function
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # GELU activation function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # If using residual connection
        if self.is_res:
            # Apply first convolutional layer
            x1 = self.conv1(x)

            # Apply second convolutional layer
            x2 = self.conv2(x1)

            # If input and output channels are the same, add residual connection directly
            if self.same_channels:
                out = x + x2
            else:
                # If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection
                shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out = shortcut(x) + x2
            #print(f"resconv forward: x {x.shape}, x1 {x1.shape}, x2 {x2.shape}, out {out.shape}")

            # Normalize output tensor
            return out / 1.414

        # If not using residual connection, return output of second convolutional layer
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

    # Method to get the number of output channels for this block
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # Method to set the number of output channels for this block
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels

        

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        
        # Create a list of layers for the upsampling block
        # The block consists of a ConvTranspose2d layer for upsampling, followed by two ResidualConvBlock layers
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        
        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Concatenate the input tensor x with the skip connection tensor along the channel dimension
        x = torch.cat((x, skip), 1)
        
        # Pass the concatenated tensor through the sequential model and return the output
        x = self.model(x)
        return x

    
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        
        # Create a list of layers for the downsampling block
        # Each block consists of two ResidualConvBlock layers, followed by a MaxPool2d layer for downsampling
        layers = [ResidualConvBlock(in_channels, out_channels), ResidualConvBlock(out_channels, out_channels), nn.MaxPool2d(2)]
        
        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential model and return the output
        return self.model(x)
    
class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        This class defines a generic one layer feed-forward neural network for embedding input data of
        dimensionality input_dim to an embedding space of dimensionality emb_dim.
        '''
        self.input_dim = input_dim
        
        # define the layers for the network
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        
        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, self.input_dim)
        # apply the model layers to the flattened tensor
        return self.model(x)
    
class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=64, n_cfeat=5, height=28):  # cfeat - context features
        super(ContextUnet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height  #assume h == w. must be divisible by 4, so 28,24,20,16...

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True) #两个conv conv:3x3,1,1conv->BN->Gelu

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)        # down1 #[100, 64, 8, 8] #两个ResidualConvBlock+Maxpool(2)
        self.down2 = UnetDown(n_feat, 2 * n_feat)    # down2 #[100, 128, 4, 4]
        
        # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 2*n_feat) #Linear(1,2*n_feat)->GELU()->Linear(2*n_feat,2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4), # up-sample 
            nn.GroupNorm(8, 2 * n_feat), # normalize                        
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1), # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=1
            nn.GroupNorm(8, n_feat), # normalize
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1), # map to same number of channels as input
        )
    
    def time_emb(self, t, dim):
        """对时间进行正弦函数的编码，单一维度
       目标：让模型感知到输入x_t的时刻t
       实现方式：多种多样
       输入x：[B, C, H, W] x += temb 与空间无关的，也即每个空间位置（H, W）,都需要加上一个相同的时间编码向量[B, C]
       假设B=1 t=0.1
       1. 简单粗暴法
       temb = [0.1] * C -> [0.1, 0.1, 0.1, ……]
       x += temb.reshape(1, C, 1, 1)
       2. 类似绝对位置编码方式
       本代码实现方式
       3. 通过学习的方式（保证T是离散的0， 1， 2， 3，……，T）
       temb_learn = nn.Parameter(T+1, dim)
       x += temb_learn[t, :].reshape(1, C, 1, 1)
       
       
        Args:
            t (float): 时间，维度为[B]
            dim (int): 编码的维度

        Returns:
            torch.Tensor: 编码后的时间，维度为[B, dim]  输入是[B, C, H, W]
        """
        # 生成正弦编码
        # 把t映射到[0, 1000]
        t = t * 1000
        # 10000^k k=torch.linspace……
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)

        return torch.cat([sin_emb, cos_emb], dim=-1)

    def label_emb(self, y, dim):
        """对类别标签进行编码，同样采用正弦编码

        Args:
            y (torch.Tensor): 图像标签，维度为[B] label:0-9
            dim (int): 编码的维度

        Returns:
            torch.Tensor: 编码后的标签，维度为[B, dim]
        """
        y = y * 1000

        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(y.device)
        sin_emb = torch.sin(y[:, None] / freqs)
        cos_emb = torch.cos(y[:, None] / freqs)

        return torch.cat([sin_emb, cos_emb], dim=-1)

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch,)      : time step
        c : (batch,)    : context label
        """
        # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

        # pass the input image through the initial convolutional layer
        x = self.init_conv(x) #[100, 1, 28, 28]->[100, 64, 28, 28]
        # pass the result through the down-sampling path
        down1 = self.down1(x)       #[100, 64, 28, 28]->[100, 64, 14, 14]
        down2 = self.down2(down1)   #[100, 64, 28, 28]->[100, 128, 7, 7]
        
        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2) #[100, 128, 7, 7]->[100, 128, 1, 1]
        
        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], 1).to(x)
            
        # embed context and timestep
        cemb1 = self.label_emb(c, self.n_feat * 2).view(-1, self.n_feat * 2, 1, 1)     # (batch, 128, 1,1)
        temb1 = self.time_emb(t, self.n_feat * 2).view(-1, self.n_feat * 2, 1, 1)        # (batch, 128, 1,1)
        cemb2 = self.label_emb(c, self.n_feat).view(-1, self.n_feat, 1, 1)       # (batch, 64, 1,1)
        temb2 = self.time_emb(t, self.n_feat).view(-1, self.n_feat, 1, 1)          # (batch, 64, 1,1)
        """ 
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)     # (batch, 128, 1,1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)        # (batch, 128, 1,1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)       # (batch, 64, 1,1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)          # (batch, 64, 1,1) """
        #print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")


        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 + up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2 + up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out
if __name__ == '__main__':
    # Test the ContextUnet model
    model = ContextUnet(1, n_feat=64, n_cfeat=5, height=28)
    x = torch.randn(100, 1, 28, 28)
    t = torch.randn(100, 1)
    #c = torch.randint(0, 10, (100, 5))
    c = None
    out = model(x, t, c)
    print(out.shape)
    # Expected output: torch.Size([100, 1, 28, 28])
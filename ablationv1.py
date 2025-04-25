
import torch
import torch.nn as nn

class Img_inpainting(nn.Module):
    def __init__(self,dilation):
        """
        Ablationv1 model: Two models made up the encoder: 
            1) self.encoder,
            2) self.attention

        """
        
        super (Img_inpainting,self).__init__()
  
        self.encoder = nn.Sequential(
            nn.Conv2d(3,3, kernel_size = 3, stride = 2,padding = 1, groups = 3),
            nn.Conv2d(3,64,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size = 3, stride = 2,padding = 1, groups = 64),
            nn.Conv2d(64,128,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128,128, kernel_size = 3, stride = 2,padding = 1, groups = 128),
            nn.Conv2d(128,256,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 2,padding = 1, groups = 256),
            nn.Conv2d(256,512,kernel_size=1),
            nn.ReLU(),
            )
        
        self.dilation_rates = dilation_rates(512, 512, dilation)
        self.attention = attention(512)
        self.conv_dilation = nn.Sequential(
            nn.Conv2d(512*len(dilation),512*len(dilation),kernel_size=3,stride=1,padding=1,groups=512*len(dilation)),
            nn.Conv2d(512*len(dilation),512*len(dilation),kernel_size=1),
            nn.ReLU())
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512*(len(dilation)), 256, kernel_size=4, stride=2, padding=1),                 
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        
        x_2 = self.encoder(x)
        x_conc = self.attention(x_2) 
        outputs = self.dilation_rates(x_conc) 
        outputs = self.conv_dilation(outputs) 
        outputs = self.decoder(outputs)
        return outputs

class attention(nn.Module):
    def __init__(self, in_channels):
        super(attention,self).__init__()
        
        self.query = nn.Conv2d(in_channels,in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size = 1)
        self.value = nn.Conv2d(in_channels, in_channels,kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self,x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        query = query.view(query.size(0),-1,query.size(2)*query.size(3))
        key = key.view(key.size(0),-1,key.size(2)*key.size(3))
        value = value.view(value.size(0),-1,value.size(2)*value.size(3))
        
        attention = torch.matmul(query.transpose(1,2),key)
        attention = torch.softmax(attention,dim=-1)
        
        out = torch.matmul(value, attention)
        out = out.view(out.size(0),-1,x.size(2),x.size(3))
        
        out = self.gamma*out + x
        
        return out
        
    
class dilation_rates(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):         
        super(dilation_rates, self).__init__()                
            # Lista con diferentes dilation       
        self.depthwise = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=d, dilation=d, groups = in_channels),
                nn.Conv2d(in_channels,out_channels,kernel_size=1)
                ])            
            for d in dilations]) 
            
    def forward(self, x):          
        depth = [depthwise_conv[1](depthwise_conv[0](x)) for depthwise_conv in self.depthwise]            
        concatenated = torch.cat(depth, dim=1)
        return concatenated 

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels = 3, num_filters = 64):
        super(PatchGANDiscriminator,self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,stride=2,padding = 1,groups=in_channels),
            nn.Conv2d(in_channels,num_filters,1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters,num_filters,3,stride=2,padding = 1,groups=num_filters),
            nn.Conv2d(num_filters,num_filters*2,1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters*2,num_filters*2,3,stride=2,padding = 1,groups=num_filters*2),
            nn.Conv2d(num_filters*2,num_filters*4,1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters*4,num_filters*4,3,stride=2,padding = 1,groups=num_filters*4),
            nn.Conv2d(num_filters*4,num_filters*8,1),
            nn.LeakyReLU(0.2, inplace=True),
            )
        
        self.conv = nn.Sequential(
            nn.Conv2d(num_filters*8,num_filters*8,3,stride=1,padding = 1,groups=num_filters*8),
            nn.Conv2d(num_filters*8,1,1))
        

        
    def forward(self,x):
        return (self.conv(self.discriminator(x)))
    
def prepare_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    dilation = [1,2,4]
    modelo = Img_inpainting(dilation)
    modelo.to(device)
    gan = PatchGANDiscriminator()
    gan.to(device)
    return modelo, gan

if __name__ == "__main__":
    
    dilation = [1,2,4]
    modelo = Img_inpainting(dilation)
    input_ = torch.randn((1,3,256,256))
    output = modelo(input_)
    print(output.shape)
    
    

   
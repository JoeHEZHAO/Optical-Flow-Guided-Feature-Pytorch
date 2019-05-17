import torch
import torch.nn as nn
import numpy as np 
import time

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    torch.save(state, str(time.time()) + filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class StableBCELoss(nn.modules.Module):
       def __init__(self):
             super(StableBCELoss, self).__init__()
       def forward(self, input, target):
             neg_abs = - input.abs()
             loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
             return loss.mean()

class SobelFilter(nn.Module):

      '''
      Implement Sobel Filter that not allow training
      sobel edges for both (x and y) directions;
      '''
      
      def __init__(self, input_dim, output_dim):
            super(SobelFilter, self).__init__()
            self.x_param = np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
            self.y_param = np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
            
            # extend from [3, 3] to [input, output, 3, 3]
            self.x_param = np.expand_dims(np.repeat(np.expand_dims(self.x_param, 0), input_dim, axis=0), 1)
            self.y_param = np.expand_dims(np.repeat(np.expand_dims(self.y_param, 0), output_dim, axis=0), 1)

            self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=output_dim) #groups for implement conv operation on each input channel, each filter weights 
            self.conv2 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=output_dim)

            self._init_weights()

      def _init_weights(self):

            self.conv1.weight = nn.Parameter(torch.from_numpy(self.x_param).float(), requires_grad=False)
            self.conv2.weight = nn.Parameter(torch.from_numpy(self.y_param).float(), requires_grad=False)

      def forward(self, input):

            x_grad = self.conv1(input)
            y_grad = self.conv2(input)
            return x_grad, y_grad

class SobelFilter_Diagonal(nn.Module):

      '''
      Implement Sobel Filter that not allow training
      sobel edges combined together for (x + y)
      '''
      
      def __init__(self, input_dim, output_dim):
            super(SobelFilter_Diagonal, self).__init__()
            self.param = np.array([[0, 1, 0],[-1, 0, 1],[0, -1, 0]])
            
            # extend from [3, 3] to [input, output, 3, 3]
            self.param = np.expand_dims(np.repeat(np.expand_dims(self.param, 0), input_dim, axis=0), 1)
            
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=output_dim) #groups for implement conv operation on each input channel, each filter weights 
            
            self._init_weights()

      def _init_weights(self):

            self.conv.weight = nn.Parameter(torch.from_numpy(self.param).float(), requires_grad=False)

      def forward(self, input):

            spatial_grad = self.conv(input)
            return spatial_grad


class SobelFilter_3D(nn.Module):

      '''Implement 3D Sobel Filter to check Temporal Consistence
            @param: input_dim: Spatial-Temporal Tensor Input Channel 
            @param: output_dim: S
      '''

      def __init__(self, input_dim, output_dim):
            super(SobelFilter_3D, self).__init__()
            # self.param = 
            # self.3dconv = nn.Conv3D()
            # self._init_weights()
            pass

      def _init_weights(self):
            pass

      def forward(self, input):
            pass


if __name__ == '__main__':

      ''' test sobelFilter net '''
      from torch.autograd import Variable
      sobel = SobelFilter(192, 192)
      test = Variable(torch.rand(64, 192, 56, 56))
      out_x, out_y = sobel(test)
      print(out_x.shape)

      # from PIL import Image
      # import torch.nn as nn
      # import torch
      # import numpy as np
      # from torchvision import transforms
      # from torch.autograd import Variable
      # img = Image.open('tf_model_zoo/lena_origin.png')
      # shape = img.size

      # T=transforms.Compose([transforms.ToTensor()])
      # P=transforms.Compose([transforms.ToPILImage()])

      # ten=torch.unbind(T(img))
      # x=ten[0].unsqueeze(0).unsqueeze(0)q

      # a=np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
      # conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
      # conv1.weight=nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
      # G_x=conv1(Variable(x)).data.view(1,shape[0],shape[1])

      # b=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
      # conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
      # conv2.weight=nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
      # G_y=conv2(Variable(x)).data.view(1,shape[0], shape[1])

      # # G=torch.sqrt(torch.pow(G_x,2) + torch.pow(G_y,2))
      # X = P(torch.sqrt(torch.pow(G_x, 2)))
      # Y = P(torch.sqrt(torch.pow(G_y, 2)))
      # X.save('x_grad.png')
      # Y.save('y_grad.png')
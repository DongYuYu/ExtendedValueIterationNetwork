import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class VIN(nn.Module):
  
    def __init__(self, config):
        super(VIN, self).__init__()
        self.config = config
    



    
        self.h = nn.Conv2d(in_channels=config.l_i, 
                           out_channels=config.l_h, 
                           kernel_size=(3, 3), 
                           stride=1, padding=1,
                           bias=True)
        self.r = nn.Conv2d(in_channels=config.l_h, 
                           out_channels=1, 
                           kernel_size=(1, 1), 
                           stride=1, padding=0,
                           bias=False)
        self.q = nn.Conv2d(in_channels=1, 
                           out_channels=config.l_q, 
                           kernel_size=(3, 3), 
                           stride=1, padding=1,
                           bias=False)
        self.fc = nn.Linear(in_features=config.l_q, 
                            out_features=8,
                            bias=False)
        self.w = Parameter(torch.zeros(config.l_q,1,3,3), requires_grad=True)
        self.sm = nn.Softmax()
        self.grid_image = None
        self.reward_image = None
        self.value_images = []

    def forward(self, X, S1, S2, O1, O2, config, record_images=False):
        h = self.h(X)
        r = self.r(h)


        if record_images:
            self.grid_image = X.data[0].cpu().numpy()
            self.reward_image = r.data[0].cpu().numpy()
            
        q = self.q(r)
        v, _ = torch.max(q, dim=1, keepdim=True)
        for i in range(0, config.k - 1):
            q = F.conv2d(torch.cat([r, v], 1), 
                         torch.cat([self.q.weight, self.w], 1),
                         stride=1, 
                         padding=1)
            v, _ = torch.max(q, dim=1, keepdim=True)
            if record_images:
                 self.value_images.append(v.data[0].cpu().numpy())
                 

                
        q = F.conv2d(torch.cat([r, v], 1), 
                     torch.cat([self.q.weight, self.w], 1),
                     stride=1, 
                     padding=1)

#        q.unsqueeze_(-1).unsqueeze_(-1)
#        q = q.repeat(1,1,1,1,config.imsize,config.imsize)        
#        t1.unsqueeze(-1)
#        t1 = t1.expand(1,10,8,8,8)
        
        
 #       print("=========================================================================================================")
#        print("t1")
#        print(t1.shape)
 #       print("pq")
 #       print(q.shape)
 #       print("q.size(0)")
 #       print(q.size(1))
 #       t0 = torch.cat([r,v],1)
  #      print("t0")
  #      print(t0.shape)
#        t = torch.cat([self.q.weight, self.w], 1)
   #     print("t")
    #    print(t.shape)
        slice_s1 = S1.long().expand(config.imsize, 1, config.l_q, q.size(0))
        slice_s1 = slice_s1.permute(3, 2, 1, 0)
        q_out = q.gather(2, slice_s1).squeeze(2)

        slice_s2 = S2.long().expand(1, config.l_q, q.size(0))
        slice_s2 = slice_s2.permute(2, 1, 0)
        q_out = q_out.gather(2, slice_s2).squeeze(2)
        
    #    slice_o1 = O1.long().expand(config.imsize, 1, config.l_q, q.size(0))
       # print("s1")
       # print(S1)
        #print(S1)
     #   print("inde0")
     #   slice_o1 = slice_o1.permute(3, 2, 1, 0)
    
      #  print(slice_s1.shape)
    #    print("qbeforesqueeze")
   #     print(q.gather(2, slice_s1))
      #  q_out = q_out.gather(2, slice_o1).squeeze(2)
       # print("q")
       # print(q_out.shape)
        #print("s2")
        #print(S2)
       # slice_o2 = O2.long().expand(1, config.l_q, q.size(0))
        #print(slice_s2)
        #slice_o2 = slice_o2.permute(2, 1, 0)
        #print(slice_s2)
        #q_out = q_out.gather(2, slice_o2).squeeze(2)
 #       print("inde")
 #       print(slice_s2.shape)
 #       print("q_out")
#        print(q_out.shape)

#try

        


        logits = self.fc(q_out)
#        print("logits")
#        print(logits)
        return logits, self.sm(logits)

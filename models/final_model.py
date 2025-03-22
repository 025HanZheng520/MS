import torch
from torch import nn
from .crf import crf
from .vit_decoder_two import decoder_fuser
from .modules import backbone
import torch.nn.functional as F

class GenerateModel(nn.Module):
    def __init__(self,cls_num=7):
        super().__init__()
        self.backone = backbone()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, cls_num)
        self.fuser = decoder_fuser(dim=128, num_heads=8, num_layers=8, drop_rate=0.)
        self.fuser1 = decoder_fuser(dim=256, num_heads=8, num_layers=8, drop_rate=0.)
        self.fuser2 = decoder_fuser(dim=512, num_heads=8, num_layers=8, drop_rate=0.)
        self.ecrf1 = crf()
        self.ecrf2 = crf()
        self.ecrf3 = crf()


    def forward(self, x):
       

        # 3 torch.Size([160, 128, 28, 28])torch.Size([160, 256, 14, 14])torch.Size([160, 512, 7, 7])
        #out_s1, out_s2, out_s3 = self.s_former(x)
        """franch1, franch2 = [], []

        out_t = []

        out_s, out_c = self.s_former(x)
        #print([out_s[i].size() for i in range(len(out_s))])
        #print([out_c[i].size() for i in range(len(out_c))])

        out_r=[]
        for index, data_s in enumerate(out_s):  # 256
            #print(index, data_s.size()) #1 torch.Size([160, 256])
            f1 = self.t_former(data_s)
            f2 = self.t_former(out_c[index])
            franch1.append(f1)
            franch2.append(f2)"""
        franch1, franch2 = self.backone(x)

        out_r = []
        for id, data in enumerate(franch1):
            b,f,c = data.shape   
            x1 = data.view(b*f,c,1,1)
            
            x2 = franch2[id].view(b*f,c,1,1)
            
            if c == 128:  
                
                x = self.ecrf1(x1,x2)
                
            elif c == 256:     
                
                x = self.ecrf2(x1,x2)
                
            else:               
               
                x = self.ecrf3(x1,x2)
                
            out_r.append(x)
        #print([out_r[i].size() for i in range(len(out_r))])

        #print(out_r[0].size(),out_r[1].size(),out_r[2].size())
        out_l = self.fuser1(out_r[1], out_r[0])
        #print(out_f.size())
        out_l = self.fuser2(out_r[2], out_l)#torch.Size([32, 17, 512])
        #print(out_l.size())
  

        x_FER = out_l[:, 0]  # torch.Size([32])


        #print(x_FER.size())

        x = self.fc(x_FER)  # [32,7]
        #print(x.size())

        return x,x_FER


if __name__ == '__main__':
    img = torch.randn((2, 32, 3, 224, 224))
    model = GenerateModel(cls_num=7)
    model(img)

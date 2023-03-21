## Goal

* The Project repo consists of two Python Notebooks (IPYNB files), Segnet and SegNext implementing the Vanilla Segnet and an improved version of Segnet (SegNext) making use of the sailent features of UNet  (Inspired from). We make use of the Spacecraft dataset for segmentation of spacecraft parts.

## SegNext
***Model:*** I create an updated and advanced version of SegNet inspired by UNet. We have seen that segnet is composed of an encoder-decoder architecture. Here we apply successive enoder layers, with filters ranging from *32,64,128 and 256* for each block. The model architecures has double convolutions followed by **BN-ReLU** (inspired by VGG16). For the decoder we applied normal convolution filters and upsample the images by using Max Unpooling. To apply this operation we make use the pooling indices after pooling and initial image sizes before pooling.

***Features:*** For **SegNext**, first we replace the unpooling and normal convolution operations with transpose convolutions(ConvTranspose2d). The encoder block is left as it is as its goal is to downsample and encode the image into a latent space whilst increasing its receptive field beyond the input size. We could have/possibly used a pretrained backbone (ResNet-50) for better performance. We also take inspiration from UNet to add residual/skip connections through concetanation, from the encoder to its corresponding decoder stage. We may also make use of 1x1 to reduce the number of channels. Currently we make use of 3x3 convolutions for the same.

***Dataset:*** The dataset is a collection of satellite images for object detection and segmentation, using both synthesized and real images. It contains 3116 images, each with a mask of size 1280x720 and bounding boxes. Each satellite is segmented into at most 3 parts, namely body, solar panel, and antenna, represented respectively by the colors green, red, and blue. Images from index 0 to 1002 have fine masks, while those from index 1003 to 3116 have coarse masks. The dataset is divided into two parts: a train set that includes 403 fine masks and 2114 coarse masks, and a validation set with 600 fine masks. The bounding boxes for all satellites are available in the file "all_bbox.txt," which is in the form of a dictionary with image index as the key and the bounding boxes represented as [max_x, max_y, min_x, min_y].

* *Link:* https://drive.google.com/drive/u/0/folders/1Q1wR9aBFCyeFEYa3wwyXNu9wk_fZdzUm

I only make use of the fine masks for training the segmentation model i.e 403 datapoints/images. In the future I will make use of all the 3k masks and images for training. The validation, and training datapoint counts have been mentioned above. 

**SegNext Model Structure:**

```
class SegNext(nn.Module):
  def __init__(self):
    super().__init__()

    self.prep = nn.Sequential(
        nn.Conv2d(3,32,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(32)
    )

    self.prep_con = nn.Sequential(
        nn.Conv2d(32,32,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(32)
    )

    self.enc1 = nn.Sequential(
        nn.Conv2d(32,64,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(64)
    ) 

    self.enc1_con = nn.Sequential(
        nn.Conv2d(64,64,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(64)
    )

    self.enc2 = nn.Sequential(
        nn.Conv2d(64,128,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(128)
    )

    self.enc2_con = nn.Sequential(
        nn.Conv2d(128,128,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(128)
    )

    self.enc3 = nn.Sequential(
        nn.Conv2d(128,256,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    self.enc3_con = nn.Sequential(
        nn.Conv2d(256,256,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    self.dec0 = nn.Sequential(
        nn.Conv2d(512,256,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(256,128,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(128)
    )  

    self.dec1 = nn.Sequential(
        nn.Conv2d(256,128,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(128,64,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(64)
    )  


    self.dec2 = nn.Sequential(
        nn.Conv2d(128,64,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64,32,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(32)
    )  

    self.final = nn.Sequential(
        nn.Conv2d(64,32,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32,4,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(4)
    )

  def forward(self, x):
    in_size = x.size()
    x = self.prep(x)
    e1 = self.prep_con(x)
    x = self.enc1(e1)
    e2 = self.enc1_con(x)
    x = self.enc2(e2)
    e3 = self.enc2_con(x)
    x = self.enc3(e3)
    e4 = self.enc3_con(x)
    x = torch.cat((e4,x), 1)
    x = self.dec0(x)
    x = torch.cat((e3,x),1)
    x = self.dec1(x)
    x = torch.concat((e2,x),1)
    x = self.dec2(x)
    x = torch.concat((e1,x),1)
    x = self.final(x)

    return x
```

**Vannila Segnet Structure:**

```
class SegNext(nn.Module):
  def __init__(self):
    super().__init__()

    self.prep = nn.Sequential(
        nn.Conv2d(3,64,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64,64,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(64)
    )

    #self.pool = nn.MaxPool2d(2,2, return_indices=True)

    self.enc1 = nn.Sequential(
        nn.Conv2d(64,128,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128,128,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(128)
    ) 

    self.enc2 = nn.Sequential(
        nn.Conv2d(128,256,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(256,256,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    self.dec1 = nn.Sequential(
        nn.Conv2d(256,256,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(256,256,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )  

    self.conv_dec1 = nn.Conv2d(256,128,1,1)

    self.unpool = nn.MaxUnpool2d(2,2)
    self.unpool1 = nn.MaxUnpool2d(2,2)

    self.dec2 = nn.Sequential(
        nn.Conv2d(128,128,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128,128,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(128)
    )  

    self.conv_dec2 = nn.Conv2d(128,64,1,1)

    self.final = nn.Sequential(
        nn.Conv2d(64,64,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64,64,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(64)
    ) 

    self.conv_dec3 = nn.Sequential(
        nn.Conv2d(64,4,1,1),
        nn.BatchNorm2d(4)
    )

  def forward(self, x):
    in_size = x.size()

    x = self.prep(x)
    x, ind_prep = F.max_pool2d(x, 2, 2, return_indices=True)
    prep_size = x.size()
    #print(prep_size)

    x = self.enc1(x)
    x , ind_enc1 = F.max_pool2d(x, 2, 2, return_indices=True)
    enc1_size = x.size()
    #print(enc1_size)

    x = self.enc2(x)
    x, ind_enc2 = F.max_pool2d(x, 2, 2 ,return_indices=True)

    x = self.unpool(x, ind_enc2, output_size=enc1_size)
    x = self.dec1(x)
    #print(x.size())
    x = self.conv_dec1(x)
    #print(x.size())

    x = self.unpool(x, ind_enc1, output_size=prep_size) 
    x = self.dec2(x)
    x = self.conv_dec2(x)
    
    x = self.unpool(x, ind_prep, output_size=in_size) 
    x = self.final(x)
    x = self.conv_dec3(x)
    
    # x = F.softmax(x,dim=1)
    return x

```


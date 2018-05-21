-- set up
require 'torch'
require 'paths'
require 'nn'
require 'cunn'
require 'cuspn'
require 'image'
require 'cudnn'
dofile './DevidetoTable.lua'
dofile './CancattoThree.lua'

cutorch.setDevice(1)
--load SPN model
model = torch.load('/root/Cloud/2-1-Googlenet/HAP/checkpoints/54-0.08016-20180327185938.t7')

numClass=5
numImg=9193
imagepath = '/root/Dataset/X-ray-dataset/JPEGImages'
xxx =torch.DiskFile( '/root/Cloud/2-1-Googlenet/CAM/data/datasets/trainval.txt','r')


--need first know the size of map
outmap1=torch.zeros(numImg,numClass,56,56):float()
outmap2=torch.zeros(numImg,numClass,28,28):float()
outmap3=torch.zeros(numImg,numClass,14,14):float()
for idx =1,numImg do
    if math.fmod(idx,100)==0 then
        print('process image： '.. idx..'/'..numImg)
    end
    img= image.load(paths.concat(imagepath,xxx:readString('*l')..'.jpg'),3,'float'):float()
    input = image.scale(img, 224, 224, 'bilinear')
    --input[1]  = input[1]/256 - 0.485; input[2]  = input[2]/256 - 0.456; input[3]  = input[3]/256 - 0.406;
    local pc = torch.LongTensor{3,2,1}
    input = input:index(1,pc):mul(256.0)
    input[1]  = input[1]-103.939; input[2]  = input[2]-116.779; input[3]  = input[3]-123.68;
    input = input:view(1,3,224,224):cuda()
    model = model:cuda()
    output = model:forward(input:cuda())

    

    activations1 = model.modules[21].modules[1].modules[2].output
    weights1 =model.modules[21].modules[1].modules[6].weight

    weights2 =model.modules[21].modules[2].modules[4].weight
    activations2 = model.modules[20].output[2]

    weights3 =model.modules[21].modules[3].modules[4].weight
    activations3 = model.modules[20].output[3]

    for j=1,numClass do
        weight = weights1[{j,{}}]
        weight = weight:view(1,(#weights1)[2],1,1):expandAs(activations1):clone()
        cam = activations1:clone():cmul(weight)
        cam = cam:sum(2)
        
        outmap1[{{idx},{j},{},{}}]:add(cam:float())
    end

    for j=1,numClass do
        weight = weights2[{j,{}}]
        weight = weight:view(1,(#weights2)[2],1,1):expandAs(activations2):clone()
        cam = activations2:clone():cmul(weight)
        cam = cam:sum(2)
        
        outmap2[{{idx},{j},{},{}}]:add(cam:float())
    end

     for j=1,numClass do
        weight = weights3[{j,{}}]
        weight = weight:view(1,(#weights3)[2],1,1):expandAs(activations3):clone()
        cam = activations3:clone():cmul(weight)
        cam = cam:sum(2)
        
        outmap3[{{idx},{j},{},{}}]:add(cam:float())
    end

 end
 matio =require 'matio'
 matio.save('SP-GoogLeNet-imsize500-xray-trainvalmap-1-2.mat',outmap1)
 matio.save('SP-GoogLeNet-imsize500-xray-trainvalmap-2-2.mat',outmap2)
 matio.save('SP-GoogLeNet-imsize500-xray-trainvalmap-3-2.mat',outmap3)
 



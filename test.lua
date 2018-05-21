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
numImg=6922
imagepath = '/root/Dataset/X-ray-dataset/JPEGImages'
xxx =torch.DiskFile( '/root/Cloud/2-1-Googlenet/CAM/data/datasets/test.txt','r')
label = torch.load('/root/Cloud/2-1-Googlenet/CAM/data/datasets/classmap-test-label-2.t7')
tgt1 = torch.zeros(numImg,numClass):float()
tgt1:fill(0)
tgt2 = torch.zeros(numImg,numClass):float()
tgt2:fill(0)
tgt3 = torch.zeros(numImg,numClass):float()
tgt3:fill(0)

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

    tgt1[{{idx},{}}]:add(output[1]:float())
    tgt2[{{idx},{}}]:add(output[2]:float())
    tgt3[{{idx},{}}]:add(output[3]:float())

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
 matio.save('SP-GoogLeNet-imsize500-xray-testmap-1-2.mat',outmap1)
 matio.save('SP-GoogLeNet-imsize500-xray-scores-1-2.mat',tgt1)
 matio.save('SP-GoogLeNet-imsize500-xray-testmap-2-2.mat',outmap2)
 matio.save('SP-GoogLeNet-imsize500-xray-scores-2-2.mat',tgt2)
  matio.save('SP-GoogLeNet-imsize500-xray-testmap-3-2.mat',outmap3)
 matio.save('SP-GoogLeNet-imsize500-xray-scores-3-2.mat',tgt3)
 function computeAveragePrecision(scores,gtlabels)
    clsResult = scores
    so,si = torch.sort(-clsResult)
    tp = torch.zeros(numImg)
    fp = torch.zeros(numImg) 
    --compute tp,fp
    tp = torch.zeros(numImg)
    fp = torch.zeros(numImg  )
    numPos = 0
    for i=1,numImg do
        if gtlabels[si[i]]>0 then tp[i] = 1 end
        if gtlabels[si[i]]<0 then fp[i] = 1 end
        if gtlabels[i]>0  then numPos = numPos+1 end
    end
    fp = fp:cumsum()
    tp = tp:cumsum()
    --compute rec,prec
    rec = torch.zeros(numImg)
    prec = torch.zeros(numImg)
     
    for i=1,numImg do
        rec[i] = tp[i]/numPos
        prec[i] =tp[i]/(tp[i]+fp[i])
       
    end
    --compute Ap
    ap = 0 
    for t=0,1,0.1 do
        b=torch.ge(rec,t)
        p=-10000
        for i=1,numImg do
            if b[i]==1  then p=math.max(p,prec[i])  end
        end
        ap=ap+p/11
    end
    return ap
end

score = tgt1:transpose(1,2)
gtlabel = label.labels
class_names = {'Gun','Knife','Wrench','Pliers','Scissors'}
for i = 1,#class_names do
    
    gtlabels = gtlabel[{{i},{}}]:squeeze()
    scores = score[{{i},{}}]:squeeze()
    ap = computeAveragePrecision(scores,gtlabels)*100
    object = {string.format('%s. %.4f',class_names[i],ap)}
    print(object[1])
end

score = tgt2:transpose(1,2)
gtlabel = label.labels
class_names = {'Gun','Knife','Wrench','Pliers','Scissors'}
for i = 1,#class_names do
    
    gtlabels = gtlabel[{{i},{}}]:squeeze()
    scores = score[{{i},{}}]:squeeze()
    ap = computeAveragePrecision(scores,gtlabels)*100
    object = {string.format('%s. %.4f',class_names[i],ap)}
    print(object[1])
end

score = tgt3:transpose(1,2)
gtlabel = label.labels
class_names = {'Gun','Knife','Wrench','Pliers','Scissors'}
for i = 1,#class_names do
    
    gtlabels = gtlabel[{{i},{}}]:squeeze()
    scores = score[{{i},{}}]:squeeze()
    ap = computeAveragePrecision(scores,gtlabels)*100
    object = {string.format('%s. %.4f',class_names[i],ap)}
    print(object[1])
end


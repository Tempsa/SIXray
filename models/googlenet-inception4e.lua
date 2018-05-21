
local nn = require 'nn'
require  'cuspn'
require 'spn'
require 'inn'
--dofile './DevidetoTable.lua'
local paths = require 'paths'

local function createModel(opt)

    if opt.backend == 'cudnn' then
        require 'cunn'
        require 'cudnn'
    end

    -- load model
    if opt.netInit == '' then
        local filename = 'data/pretrained_models/googlenet-inception4e.t7'
        paths.mkdir('data/pretrained_models')
        opt.netInit = filename
    end
    print('[model] read model: ' .. opt.netInit)
    model = torch.load(opt.netInit)
    
    model2 = nn.Sequential()
    model2:add(model.modules[13])
    model2:add(model.modules[14])
    model2:add(model.modules[15])
    model2:add(model.modules[16])
    model2:add(model.modules[17])
    model2:add(model.modules[18])
    model2:add(cudnn.SpatialConvolution(832,832,1,1))
    model2:add(cudnn.ReLU())

    model00 = nn.Sequential()
    model00:add(cudnn.SpatialConvolution(480,480,1,1))
    model00:add(cudnn.ReLU())

    model0 = nn.ConcatTable()
    model0:add(model00)
    model0:add(model2)

    model01 = nn.Sequential()
    model01:add(model0)

    model1 = nn.Sequential()
    model1:add(model.modules[10])
    model1:add(model.modules[11])
    model1:add(model.modules[12])
    model1:add(model01)
    
  
    model:remove(18)
    model:remove(17)
    model:remove(16)
    model:remove(15)
    model:remove(14)
    model:remove(13)
    model:remove(12)
    model:remove(11)
    model:remove(10)


    model3 = nn.Sequential()
    model3:add(cudnn.SpatialConvolution(192,192,1,1))
    model3:add(cudnn.ReLU()) 
    
    model4 = nn.ConcatTable()
    model4:add(model3)
    model4:add(model1)
     

    

    model6 = nn.ConcatTable()
    model6:add(nn.SpatialUpSamplingNearest(2))
    model6:add(cudnn.SpatialConvolution(832,1024,1,1))

    model5 = nn.ParallelTable()
    model5:add(nn.Identity())
    model5:add(nn.Identity())
    model5:add(model6)


    model8 = nn.Sequential()
    model8:add(model4)
    
    model9 = nn.ParallelTable()
    model9:add(nn.Identity())
    model9:add(cudnn.SpatialConvolution(1312,480,1,1))
    model9:add(nn.Identity())
    
    model09 = nn.ParallelTable()
    model09:add(nn.Identity())
    model09:add(cudnn.ReLU())
    model09:add(nn.Identity())

    --model10 =nn.ParallelTable()
    --model10:add(nn.Identity())
    --model10:add(cudnn.SpatialConvolution(1312,1024,1,1))
    --model10:add(nn.Identity())

    model10 = nn.ConcatTable()
    model10:add(nn.SpatialUpSamplingNearest(2))

    model10:add(cudnn.SpatialConvolution(480,512,1,1))

    model11 = nn.Sequential()
    model11:add(model10)

    model12 = nn.ParallelTable()
    model12:add(nn.Identity())
    model12:add(model11)
    model12:add(nn.Identity())
    
    model013 = nn.ParallelTable()
    model013:add(cudnn.SpatialConvolution(672,192,1,1))
    model013:add(nn.Identity())
    model013:add(nn.Identity())

    model13 = nn.ParallelTable()
    model13:add(cudnn.ReLU())
    model13:add(nn.Identity())
    model13:add(nn.Identity())

    model14 = nn.ParallelTable()
    model14:add(cudnn.SpatialConvolution(192,256,1,1))
    model14:add(nn.Identity())
    model14:add(nn.Identity())

    

    model15= nn.Sequential()
    model15:add(cudnn.ReLU())
    --model15:add(cudnn.SpatialAveragePooling(56,56,56,56))
    model15:add(nn.SoftProposal())
    model15:add(nn.SpatialSumOverMap())
    model15:add(nn.View(-1, 256))
    model15:add(nn.Dropout(0.5))
    --model19:add(nn.Dropout(0.5))
    model15:add(nn.Linear(256, opt.nClasses))

    model16 = nn.Sequential()
    model16:add(cudnn.ReLU())
    --model16:add(nn.SoftProposal())
    --model16:add(nn.SpatialSumOverMap())
    model16:add(cudnn.SpatialAveragePooling(28,28,28,28))
    model16:add(nn.View(-1, 512))
    --model16:add(nn.Dropout(0.5))
    model16:add(nn.Linear(512, opt.nClasses))

    
    model17 = nn.Sequential()
    model17:add(cudnn.ReLU())
    --model17:add(nn.SoftProposal())
   -- model17:add(nn.SpatialSumOverMap())
    model17:add(cudnn.SpatialAveragePooling(14,14,14,14))
    model17:add(nn.View(-1, 1024))
    --model17:add(nn.Dropout(0.5))
    model17:add(nn.Linear(1024, opt.nClasses))

    model18 = nn.ParallelTable()
    model18:add(model15)
    model18:add(model16)
    model18:add(model17)



    model:add(model8)
    model:add(nn.CancattoThree())
    model:add(model5)
    model:add(nn.DevidetoTable())
    model:add(model9)
    model:add(model09)
    model:add(model12)
    model:add(nn.DevidetoTable())
    model:add(model013)
    model:add(model13)
    model:add(model14)
    model:add(model18)

    model = model:cuda()

    collectgarbage()
    local numPretrainedParam = model:getParameters():size(1)
     
    if opt.backend =='cudnn' then
      cudnn.convert(model,cudnn)
    end
    -- find the output size to define the spatial aggregation layer
    local input = torch.rand(1, 3, opt.imageSize, opt.imageSize):float()
           input = input:cuda()
    print('[model] input ' .. input:size(1) .. ' x ' .. input:size(2) .. ' x ' .. input:size(3) .. ' x ' .. input:size(4))

    local output = model:forward(input)
    --print('[model] output'..output)
    --print('[model] output ' .. output:size(1) .. ' x ' .. output:size(2))
    local numParam = model:getParameters():size(1)

    --opt.lastconv = 9

    if opt.LRp ~= 1 and opt.LRp >= 0 then
        print('[model] initialize learningRates', opt.LRp)
        local lrs = torch.Tensor(numParam):zero()
        for i = 1, numParam do
            if i > numPretrainedParam then
                lrs[i] = 1.0
            else
                lrs[i] = opt.LRp
            end
        end
        opt.learningRates = lrs
        if opt.backend == 'cudnn' then
            opt.learningRates = opt.learningRates:cuda()
        end
    end

    opt.pathResults = '/k=' .. opt.k

    return model
end

return createModel

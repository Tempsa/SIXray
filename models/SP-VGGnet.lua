
local nn = require 'nn'
require  'cuspn'
require 'spn'
--dofile './DevidetoTable.lua'
local paths = require 'paths'

local function createModel(opt)

    if opt.backend == 'cudnn' then
        require 'cunn'
        require 'cudnn'
    end

    -- load model
    if opt.netInit == '' then
        local filename = 'data/pretrained_models/VGG-conv5-3Relu.t7'
        paths.mkdir('data/pretrained_models')
        opt.netInit = filename
    end
    print('[model] read model: ' .. opt.netInit)
    model = torch.load(opt.netInit)
    model1 = nn.Sequential()
    model1:add(cudnn.SpatialConvolution(512,512,1,1))
    

    model2 = nn.Sequential()
    model2:add(model.modules[27])
    model2:add(model.modules[28])
    model2:add(cudnn.SpatialConvolution(512,512,1,1))
    

    model3 = nn.Sequential()
    model3:add(model.modules[27])
    model3:add(model.modules[28])
    model3:add(model.modules[29])
    model3:add(model.modules[30])
    model3:add(cudnn.SpatialConvolution(512,512,1,1))
    
    model:remove(30)
    model:remove(29)
    model:remove(28)
    model:remove(27)

    model4 = nn.ConcatTable()
    model4:add(model1)
    model4:add(model2)
    model4:add(model3)

    model5 = nn.Sequential()
    model5:add(model4) 

    model6 = nn.ConcatTable()
    model6:add(nn.Identity())
    model6:add(cudnn.SpatialConvolution(512,1024,1,1))

    model7 = nn.ParallelTable()
    model7:add(nn.Identity())
    model7:add(nn.Identity())
    model7:add(model6)

    model8 = nn.ParallelTable()
    model8:add(nn.Identity())
    model8:add(cudnn.SpatialConvolution(1024,1024,1,1))
    model8:add(cudnn.ReLU(true))

    model9 = nn.ConcatTable()
    model9:add(nn.Identity())
    model9:add(cudnn.ReLU(true))

    model13 = nn.Sequential()
    model13:add(model9)

    model10 = nn.ParallelTable()
    model10:add(nn.Identity())
    model10:add(model13)
    model10:add(nn.Identity())

    model11 = nn.ParallelTable()
    model11:add(cudnn.SpatialConvolution(1536,1024,1,1))
    model11:add(nn.Identity())
    model11:add(nn.Identity())

    model12 = nn.ParallelTable()
    model12:add(cudnn.ReLU(true))
    model12:add(nn.Identity())
    model12:add(nn.Identity())







    

    model15 = nn.Sequential()
    model15:add(nn.SoftProposal())
    model15:add(nn.SpatialSumOverMap())
    model15:add(nn.View(-1, 1024))
    model15:add(nn.Dropout(0.5))
    model15:add(nn.Linear(1024, opt.nClasses))

    model16 = nn.Sequential()
    model16:add(nn.SoftProposal())
    model16:add(nn.SpatialSumOverMap())
    model16:add(nn.View(-1, 1024))
    model16:add(nn.Dropout(0.5))
    model16:add(nn.Linear(1024, opt.nClasses))

    model17 = nn.Sequential()
    model17:add(nn.SoftProposal())
    model17:add(nn.SpatialSumOverMap())
    model17:add(nn.View(-1, 1024))
    model17:add(nn.Dropout(0.5))
    model17:add(nn.Linear(1024, opt.nClasses))

    model18 = nn.ParallelTable()
    model18:add(model15)
    model18:add(model16)
    model18:add(model17)


    model:add(model5)
    model:add(model7)
    model:add(nn.DevidetoTable())
    model:add(model8)
    model:add(model10)
    model:add(nn.DevidetoTable())
    model:add(model11)
    model:add(model12)
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

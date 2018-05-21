 --[[
-- A MultiLabel multiclass criterion based on sigmoid:
--
-- the loss is:
-- l(x,y) = - sum_i y[i] * log(p[i]) + (1 - y[i]) * log (1 - p[i])
-- where p[i] = exp(x[i]) / (1 + exp(x[i]))
--
-- and with weights:
-- l(x,y) = - sum_i weights[i] (y[i] * log(p[i]) + (1 - y[i]) * log (1 - p[i]))
--
-- This uses the stable form of the loss and gradients.
--]]


local MultiLabelSoftMarginweightCriterion, parent = torch.class('nn.MultiLabelSoftMarginweightCriterion', 'nn.Criterion')


function MultiLabelSoftMarginweightCriterion:__init(weights, sizeAverage)
   parent.__init(self)
   if sizeAverage ~= nil then
      self.sizeAverage = sizeAverage
   else
      self.sizeAverage = true
   end
   if weights ~= nil then
      assert(weights:dim() == 1, "weights input should be 1-D Tensor")
      self.weights = weights
   end
   self.sigmoid = nn.Sigmoid()
end

function MultiLabelSoftMarginweightCriterion:updateOutput(input, target)
   local weights = self.weights
   if weights ~= nil and target:dim() ~= 1 then
      weights = self.weights:view(1, target:size(2)):expandAs(target)
   end

   local x = input:view(input:nElement())
   local t = target:view(target:nElement())
   local N = 0
   local P = 0
   --for idx =1,target:nElement() do
    --  if t[idx] >0 then 
     --    P=P+1
     -- else
     --    N = N +1;
     -- end
   --end

   local m = torch.rand(t:size())
   local temp = 1
   if P>0 then
      temp=(P+N)/P
   end
   for  ix = 1,target:nElement() do
      if t[ix] > 0 then
         m[ix] = 1
      else
         m[ix] = 0.5
      end
   end


   self.sigmoid:updateOutput(x)

   self._buffer1 = self._buffer1 or input.new()
   self._buffer2 = self._buffer2 or input.new()

   self._buffer1:ge(x, 0) -- indicator

   -- log(1 + exp(x - cmul(x, indicator):mul(2)))
   self._buffer2:cmul(x, self._buffer1):mul(-2):add(x):exp():add(1):log()
   -- cmul(x, t - indicator)
   self._buffer1:mul(-1):add(t):cmul(x)
   -- log(1 + exp(x - cmul(x, indicator):mul(2))) - cmul(x, t - indicator)
   self._buffer2:add(-1, self._buffer1)

   if weights ~= nil then
      self._buffer2:cmul(weights)
   end

   m = m:cuda()
   self._buffer2:cmul(m)

   self.output = self._buffer2:sum()

   if self.sizeAverage then
      self.output = self.output / input:nElement()
   end

   return self.output
end

function MultiLabelSoftMarginweightCriterion:updateGradInput(input, target)
   local weights = self.weights
   if weights ~= nil and target:dim() ~= 1 then
      weights = self.weights:view(1, target:size(2)):expandAs(target)
   end

   self.gradInput:resizeAs(input):copy(self.sigmoid.output)
   self.gradInput:add(-1, target)

   if weights ~= nil then
      self.gradInput:cmul(weights)
   end
   
   local t = target:view(target:nElement())
   local N = 0
   local P = 0
   --for idx =1,target:nElement() do
    --  if t[idx] >0 then 
    --     P=P+1
    --  else
     --    N = N +1;
     -- end
  -- end
   --print(self.gradInput:size())
   --print(target:size())
   local m = torch.rand(self.gradInput:size())
   local temp =1
   if P > 0 then
      temp = (P+N)/P
   end
   for  ix = 1,target:size(1) do
      for id = 1,target:size(2) do
         if target[ix][id] > 0 then
            m[ix][id] = 1
         else
            m[ix][id] = 0.5
         end
      end
   end
    
    m = m:cuda()
   self.gradInput:cmul(m)



   if self.sizeAverage then
      self.gradInput:div(target:nElement())
   end

   return self.gradInput
end

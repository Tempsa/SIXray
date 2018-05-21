local CancattoThree, parent = torch.class('nn.CancattoThree', 'nn.Module')

function CancattoThree:__init()
    parent.__init(self)
    self.gpucompatible = true
end

 function makeContiguous(input)
   if not input:isContiguous() then
      _input = _input or input.new()
      _input:resizeAs(input):copy(input)
      input = _input
   end
   return input
end

function CancattoThree:updateOutput(input)
  self.output = {}
  self.output[1] = input[1]
  self.output[2] = input[2][1]
  self.output[3] = input[2][2]
   --return makeContiguous(self.output)
    return self.output
end

function CancattoThree:updateGradInput(input, gradOutput)
    
     self.gradInput = {}
   
    self.gradInput[1] = gradOutput[1]
    self.gradInput[2] = {gradOutput[2],gradOutput[3]}
    
   
   
   -- return makeContiguous(self.gradInput)
    return self.gradInput
end

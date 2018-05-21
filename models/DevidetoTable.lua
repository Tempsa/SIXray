local DevidetoTable, parent = torch.class('nn.DevidetoTable', 'nn.Module')

function DevidetoTable:__init()
    parent.__init(self)
    self.gpucompatible = true
end

function DevidetoTable:updateOutput(input)
	self.output = {}
    if #input[2] == 2 then
        self.output[1] = torch.cat(input[1],input[2][1],2)
        self.output[2] = input[2][2]
        self.output[3] = input[3]
    end
    if #input[3] == 2  then
        self.output[1] = input[1]
        self.output[2] = torch.cat(input[2],input[3][1],2)
        self.output[3] = input[3][2]
    end
    return self.output
end

function DevidetoTable:updateGradInput(input, gradOutput)
    local x = {}
     self.gradInput = {}
    if #input[2] == 2  then
      
       x = gradOutput[1]:split(input[1]:size()[2],2)
       
       self.gradInput[1] = x[1]
       
       local y=x[2] 
       for i = 3,#x do
       	   y=torch.cat(y,x[i],2)
       	end
      
       self.gradInput[2] = {y,gradOutput[2]}
       self.gradInput[3] = gradOutput[3]
    
    end
    
    if #input[3] == 2  then
       x = gradOutput[2]:split(input[2]:size()[2],2)
       self.gradInput[1] = gradOutput[1]
       self.gradInput[2] = x[1]
       local y=x[2]
       for i = 3,#x do 
       	  y = torch.cat(y,x[i],2) 
       	end
       self.gradInput[3] = {y,gradOutput[3]}   
    end
   
    return self.gradInput
end

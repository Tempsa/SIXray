

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local VOCDataset = torch.class('VOCDataset', M)

local meanstdResnet = {
  mean = { 103.939, 116.779, 123.68 },
   --std = { 0.229, 0.224, 0.225 },
}

function VOCDataset:__init(imageInfo, opt, split)
  self.imageInfo = imageInfo[split]
  self.split = split
  self.dir = paths.concat(opt.data, 'JPEGImages/')
  self.imageSize = opt.imageSize
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
  print('[dataset] X-ray  -  split ' .. self.split .. '  -  ' .. self.imageInfo.imageClass:size(1) .. ' images with ' .. self.imageInfo.imageClass:size(2) .. ' classes')
  opt.nClasses = imageInfo.nClasses
  self.opt = opt
end

function VOCDataset:preprocess()
  if self.opt.preprocessing == 'warp' then
    return t.Compose{
        t.ColorNormalize(meanstdResnet),
      }
  elseif  self.opt.preprocessing == 'warp+hflip' then
    if self.split == 'test' then
      return t.Compose{
        t.ColorNormalize(meanstdResnet),
      }
    else
      return t.Compose{
        t.ColorNormalize(meanstdResnet),
        t.HorizontalFlip(0.5),
      }
    end
  end
end

function VOCDataset:size()
  return self.imageInfo.imageClass:size(1)
end

function VOCDataset:get(i)
  local path = ffi.string(self.imageInfo.imagePath[i]:data())

  local image = self:loadImage(paths.concat(self.dir, path))
  local class = self.imageInfo.imageClass[i]

  return {
    input = image,
    target = class,
    path = path,
  }
end

function VOCDataset:loadImage(path)
  local input = image.load(path, 3, 'float'):float()
  input = image.scale(input, self.imageSize, self.imageSize,'bilinear')
  local pc = torch.LongTensor{3,2,1}
  input = input:index(1,pc):mul(256.0)
  --if self.opt.preprocessing == 'warp' or self.opt.preprocessing == 'warp+hflip' then
   --  input = image.scale(input, self.imageSize, self.imageSize)
 -- end
  return input
end

return M.VOCDataset

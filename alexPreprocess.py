from PIL import Image
import numpy as np
import os
import json
import pandas as pd
import json_lines

root = 'memesData/data/img'
jsonlPaths = ['memesData/data/train.jsonl', 'memesData/data/dev_seen.jsonl', 'memesData/data/dev_unseen.jsonl']
directory = os.listdir(root)
labelDict = {}
imgDataToDump = []
labelDataToDump = []
windowSize = 227

def padImg(img):
    width, height = img.size
    new_width = max(windowSize, width)
    new_height = max(windowSize, height)
    result = Image.new(img.mode, (new_width, new_height), 0)
    leftStart = 0 if new_width == width else (new_width - width) //2
    topStart = 0 if new_height == height else (new_height - height) //2
    result.paste(img, (leftStart, topStart))
    return result

for p in jsonlPaths:
    with open(p) as f:
        for item in json_lines.reader(f):
            labelDict[int(item['id'])] = [item['id'],item['label'],item['text']]

for imgPath in labelDict:
    fPath = f'{root}/{imgPath:05}.png'
    imgId = imgPath
    imgId = int(imgId)
    img = Image.open(fPath)
    npData = np.asarray(img)
    print(imgId)
    label = labelDict[imgId][1]
    img = padImg(img)
    img.save('postPadImage.png')
    if img.size[0] > windowSize * 2 or img.size[1] > windowSize *  2:#divide into sections and crop

        #move to multiple of windowSize, alexnets size
        widthMult = img.size[0]//windowSize
        heightMult = img.size[1]//windowSize
        widthSpill =  img.size[0] % windowSize
        wPad = 0
        heightSpill =  img.size[1] % windowSize
        hPad = 0
        if (widthSpill / 2) % windowSize!= 0:
            wPad += 1
        if (heightSpill / 2) % windowSize!= 0:
            hPad += 1
        img = img.crop((widthSpill//2, heightSpill//2, img.size[0] - (widthSpill//2) - wPad, img.size[1] - (heightSpill//2) - hPad))
        for w in range(widthMult):
            for h in range(heightMult):
                left = w * windowSize 
                right = (w + 1) * windowSize
                top = h * windowSize
                bot = (h + 1) * windowSize
                croppedImg = img.crop((left, top, right, bot)).convert('L')
                croppedImg.save('temp.png')
                croppedImage = Image.open('temp.png')
                imgDataToDump.append(np.array(croppedImg))
                labelDataToDump.append(label)

        
    else : #center crop
        #img are width x height
        top = img.size[1]  // 2 - 113
        bot = img.size[1]  // 2 + 114
        left = img.size[0]  // 2 - 113
        right = img.size[0]  // 2 + 114
        croppedImg = img.crop((left, top, right, bot)).convert('L')
        croppedImg.save('temp.png')
        croppedImage = Image.open('temp.png')
        imgDataToDump.append(np.array(croppedImg))
        labelDataToDump.append(label)


imgDataToDump = np.array(imgDataToDump)
labelDataToDump = np.array(labelDataToDump)
with open('imgData.npy', 'wb') as f:
    np.save(f, imgDataToDump) 
with open('labelData.npy', 'wb') as f:
    np.save(f, labelDataToDump) 
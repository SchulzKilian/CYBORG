import sys
import torch

from torchvision.models import (
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    inception_v3, Inception_V3_Weights
)
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import csv
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append("../")
from xception.network.models import model_selection

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    device = torch.device('cuda')
    parser.add_argument('-imageFolder', default='../Data/',type=str)
    parser.add_argument('-modelPath',  default='../local_results/final_model.pth',type=str)
    parser.add_argument('-csv',  default="../csvs/test.csv",type=str)
    parser.add_argument('-output_scores',  default='../local_results/output_scores.csv',type=str)
    parser.add_argument('-network',  default="densenet",type=str)
    args = parser.parse_args()



    os.makedirs(args.output_scores,exist_ok=True)

    # Load weights of single binary DesNet121 model
    weights = torch.load(args.modelPath)
    if args.network == "resnet":
        im_size = 224
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif args.network == "inception":
        im_size = 299
        model = inception_v3(
            weights=Inception_V3_Weights.DEFAULT,  
            aux_logits=False
        )
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif args.network == "xception":
        im_size = 299
        model, *_ = model_selection(modelname='xception', num_out_classes=2)
    else: # else DenseNet
        im_size = 224
        model = densenet121(weights=DenseNet121_Weights.DEFAULT)  # Updated
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)

    model.load_state_dict(weights['state_dict'])
    model = model.to(device)
    model.eval()

    if args.network == "xception":
        # Transformation specified for the pre-processing
        transform = transforms.Compose([
                    transforms.Resize([im_size, im_size]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*3, [0.5]*3)
                ])
    else:
        # Transformation specified for the pre-processing
        transform = transforms.Compose([
                    transforms.Resize([im_size, im_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    imagesScores=[]
    sigmoid = nn.Sigmoid()

    # imageFiles = glob.glob(os.path.join(args.imageFolder,'*.jpg'))
    imageCSV = open(args.csv,"r")
    for entry in tqdm(imageCSV):
        
        tokens = entry.split(",")
        if tokens[0] != 'test':
            continue
        upd_name = tokens[-1].replace("\n","")
        # upd_name = upd_name.replace(upd_name.split(".")[-1],"png")
        imgFile = args.imageFolder + upd_name

        # Read the image
        image = Image.open(imgFile).convert('RGB')
        # Image transformation
        tranformImage = transform(image)
        image.close()


        ## START COMMENTING HERE

        tranformImage = tranformImage[0:3,:,:].unsqueeze(0)
        tranformImage = tranformImage.to(device)

        # Output from single binary CNN model
        with torch.no_grad():
            output = model(tranformImage)
        
        PAScore = sigmoid(output).detach().cpu().numpy()[:, 1]
        imagesScores.append([imgFile, PAScore[0]])

    # Writing the scores in the csv file
    with open('output.csv','w',newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(imagesScores)

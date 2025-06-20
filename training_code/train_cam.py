import os
import sys
import argparse
import json
from Evaluation import evaluation
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_Loader_cam import datasetLoader
from tqdm import tqdm
sys.path.append("../")
# sys.path.append("./")
from xception.network.models import model_selection


# Description of all argument
parser = argparse.ArgumentParser()
parser.add_argument('-batchSize', type=int, default=20)
parser.add_argument('-nEpochs', type=int, default=50)
parser.add_argument('-csvPath', required=False, default= '',type=str)
parser.add_argument('-datasetPath', required=False, default= '/home/kilianschulz/Programming/Machine-Teaching/pets/oxford-iiit-pet/images/',type=str)
parser.add_argument('-heatmaps', required=False, default= '/home/kilianschulz/Programming/Machine-Teaching/pets/oxford-iiit-pet/annotations/saliency/',type=str)
parser.add_argument('-alpha', required=False, default=0.5,type=float)
parser.add_argument('-network', default= 'densenet',type=str)
parser.add_argument('-nClasses', default= 2,type=int)
parser.add_argument('-create_csv', action='store_true', help='Create CSV file for dataset')
parser.add_argument('-image_percent', default=1.0, type=float, help='Percentage of images to select for testing')
parser.add_argument('-image_choosing', default='random', help='How to choose images for testing, options: random, coreset')
parser.add_argument('-outputPath', required=False, default= '',type=str)
parser.add_argument('-cyborg_weighting', default=1.0, help='How to weight the Cyborg loss, default is 1.0 (no weighting)')
parser.add_argument('-nRuns', default=1, type=int, help='Number of runs for the experiment')

args = parser.parse_args()
device = torch.device('cuda')

print(args)

# --- This activation dictionary will be reset implicitly with each model re-initialization ---
activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output
  return hook

# --- Directory setup (do this once) ---
folder_name_logs = 'Logs_' + args.image_choosing + '_' + str(args.image_percent) + '_' + str(args.cyborg_weighting) + '_' + args.network
log_path = os.path.join(args.outputPath, folder_name_logs)
if not os.path.exists(log_path):
    os.mkdir(log_path)

folder_name_results = 'Results' + args.image_choosing + '_' + str(args.image_percent) + '_' + str(args.cyborg_weighting) + '_' + args.network
result_path = os.path.join(args.outputPath, folder_name_results)
if not os.path.exists(result_path):
    os.mkdir(result_path)

# --- Dataloader setup (do this once) ---
class_assgn = {'Real':0,'Synthetic':1}

if args.create_csv:
    # This part remains the same
    import pandas as pd
    from sklearn.model_selection import train_test_split
    files = [f for f in os.listdir(args.datasetPath) if f.endswith(('.jpg', '.png'))]
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    rows = []
    for f in train_files:
        label = 'Dog' if f[0].isupper() else 'Cat'
        rows.append(['train', label, f])
    for f in test_files:
        label = 'Dog' if f[0].isupper() else 'Cat'
        rows.append(['test', label, f])
    csv_filename = 'args.csv'
    csv_path = os.path.join(os.getcwd(), csv_filename)
    pd.DataFrame(rows, columns=['split', 'class', 'filename']).to_csv(csv_path, index=False)
    class_assgn = {'Dog':0,'Cat':1}
    args.csvPath = csv_path
    print("CSV file created at:", args.csvPath)
    print("Dataset path:", args.datasetPath)

# Set image and map sizes based on network
if args.network == "inception" or args.network == "xception":
    im_size = 299
    map_size = 10 if args.network == "xception" else 8
else:
    im_size = 224
    map_size = 7

dataseta = datasetLoader(args.csvPath,args.datasetPath,train_test='train',c2i=class_assgn,map_location=args.heatmaps,map_size=map_size,im_size=im_size,network=args.network)
dl = torch.utils.data.DataLoader(dataseta, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)
dataset = datasetLoader(args.csvPath,args.datasetPath, train_test='test', c2i=dataseta.class_to_id,map_location=args.heatmaps,map_size=map_size,im_size=im_size,network=args.network)
test = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=False, num_workers=0, pin_memory=True) # Shuffle false for consistent validation
dataloader = {'train': dl, 'test':test}


#####################################################################################
#
############### NEW: Storage for Multiple Runs ######################################
#
#####################################################################################
# Store the loss history from each run
all_runs_train_loss = []
all_runs_test_loss = []


#####################################################################################
#
############### NEW: Main Loop for nRuns ############################################
#
#####################################################################################
for run_num in range(args.nRuns):
    print(f"\n{'='*30}")
    print(f"Starting Run: {run_num + 1} / {args.nRuns}")
    print(f"{'='*30}\n")

    #
    # --- Re-initialize Model, Optimizer, and Scheduler for each run ---
    # This is CRITICAL for ensuring that each run is independent.
    #
    print("Initializing model for new run...")
    if args.network == "resnet":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.nClasses)
        model.layer4[-1].conv3.register_forward_hook(getActivation('features'))
    elif args.network == "inception":
        model = models.inception_v3(pretrained=True,aux_logits=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.nClasses)
        model.Mixed_7c.register_forward_hook(getActivation('features'))
    elif args.network == "xception":
        model, *_ = model_selection(modelname='xception', num_out_classes=2)
        model.model.conv4.register_forward_hook(getActivation('features'))
    else: # Densenet
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, args.nClasses)
    
    model = model.to(device)

    # Re-initialize optimizer and scheduler
    lr = 0.005
    solver = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9)
    lr_sched = optim.lr_scheduler.StepLR(solver, step_size=12, gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()
    criterion_hmap = nn.MSELoss()

    # --- Re-initialize Logging and Tracking variables for each run ---
    log = {'iterations':[], 'epoch':[], 'validation':[], 'train_acc':[], 'val_acc':[]}
    train_loss=[]
    test_loss=[]
    bestAccuracy = 0
    bestEpoch=0

    # Hyperparameters (alpha, cyborg_weighting, etc.) are taken from args
    alpha = args.alpha
    cyborg_weighting = args.cyborg_weighting
    image_percent = args.image_percent
    image_dis = float('inf') if image_percent == 0.0 else 1.0/image_percent

    #####################################################################################
    #
    ############### Training of the model and logging (Original Loop) ###################
    #
    #####################################################################################
    for epoch in range(args.nEpochs):
        for phase in ['train', 'test']:
            is_train = (phase=='train')
            model.train(is_train)
            if args.network == "xception":
                model.model.train(is_train)

            tloss = 0.
            logger_loss = 0.
            acc = 0.
            tot = 0
            c = 0
            
            # These are only used in the 'test' phase
            testPredScore = []
            testTrueLabel = []
            imgNames=[]

            with torch.set_grad_enabled(is_train):
                for batch_idx, (data, cls, imageName, hmap) in enumerate(tqdm(dataloader[phase], desc=f"Run {run_num+1}, Epoch {epoch+1} {phase}")):
                    data, cls, hmap = data.to(device), cls.to(device), hmap.to(device)

                    if torch.isnan(hmap).any():
                        print("NaN in hmap! Exiting.")
                        sys.exit()

                    outputs = model(data)
                    pred = torch.max(outputs,dim=1)[1]
                    corr = torch.sum((pred == cls).int())
                    acc += corr.item()
                    tot += data.size(0)
                    class_loss = criterion(outputs, cls)
                    
                    # Determine if heatmap loss should be calculated for this batch
                    use_hmap = (is_train and 
                                alpha != 1.0 and 
                                image_percent > 0.0 and
                                batch_idx % image_dis < 1 and 
                                not torch.all(hmap == 0))

                    if use_hmap:
                        # Logic for calculating CAM and heatmap loss
                        if args.network == "densenet":
                            features = model.features(data)
                            params = list(model.classifier.parameters())[0]
                        else: # ResNet, Inception, Xception
                            features = activation['features']
                            params = list(model.fc.parameters())[0] if args.network != "xception" else list(model.model.last_linear.parameters())[0]

                        bz, nc, h, w = features.shape
                        beforeDot = features.view(bz, nc, h * w)
                        cams = []
                        for ids, bd in enumerate(beforeDot):
                            weight = params[pred[ids]]
                            cam = torch.matmul(weight, bd)
                            cam_img = cam.view(h, w)
                            cam_img = cam_img - torch.min(cam_img)
                            cam_img = cam_img / torch.max(cam_img)
                            cams.append(cam_img)
                        cams = torch.stack(cams)
                        hmap_loss = criterion_hmap(cams, hmap)
                    else:
                        hmap_loss = torch.tensor(0.0, device=device) # Ensure it's a tensor on the correct device

                    # Calculate total loss and optimize
                    if is_train:
                        loss = (alpha * class_loss) + (1 - alpha) * hmap_loss * cyborg_weighting
                        solver.zero_grad()
                        loss.backward()
                        solver.step()
                        log['iterations'].append(class_loss.item())
                    else: # test phase
                        loss = class_loss # Only consider classification loss for validation metric
                        temp = outputs.detach().cpu().numpy()
                        scores = np.stack((temp[:,0], np.amax(temp[:,1:args.nClasses], axis=1)), axis=-1)
                        testPredScore.extend(scores)
                        testTrueLabel.extend((cls.detach().cpu().numpy()>0)*1)
                        imgNames.extend(imageName)
                    
                    tloss += loss.item()
                    logger_loss += class_loss.item()
                    c += 1

            # --- Logging for the current epoch ---
            if is_train:
                epoch_train_loss = logger_loss / c
                train_loss.append(epoch_train_loss)
                log['epoch'].append(epoch_train_loss)
                log['train_acc'].append(acc / tot)
                print(f'Run {run_num+1}, Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Accuracy: {acc/tot:.4f}')

            else: # test phase
                epoch_test_loss = tloss / c
                test_loss.append(epoch_test_loss)
                log['validation'].append(epoch_test_loss)
                log['val_acc'].append(acc / tot)
                print(f'Run {run_num+1}, Epoch {epoch+1}: Test Loss: {epoch_test_loss:.4f}, Accuracy: {acc/tot:.4f}')
                
                lr_sched.step()
                accuracy = acc / tot
                if (accuracy >= bestAccuracy):
                    bestAccuracy = accuracy
                    testTrueLabels = testTrueLabel
                    testPredScores = testPredScore
                    bestEpoch = epoch
                    
                    # Save run-specific best model
                    save_best_model = os.path.join(log_path, f'final_model_run_{run_num+1}.pth')
                    states = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': solver.state_dict()}
                    torch.save(states, save_best_model)
                    testImgNames = imgNames

    # --- After each run, save its logs and append loss history for final plot ---
    with open(os.path.join(log_path, f'model_log_run_{run_num+1}.json'), 'w') as out:
        json.dump(log, out)
    
    all_runs_train_loss.append(train_loss)
    all_runs_test_loss.append(test_loss)

    # --- Evaluation for the current run ---
    print(f"\nEvaluating best model from Run {run_num+1} (Epoch {bestEpoch+1})")
    # obvResult = evaluation()
    # errorIndex, predictScore, threshold = obvResult.get_result(f"CYBORG_Run_{run_num+1}", testImgNames, testTrueLabels, testPredScores, result_path)

print(f"\n{'='*30}")
print("All runs completed.")
print(f"{'='*30}\n")


#####################################################################################
#
############### NEW: Plotting Aggregated Results ####################################
#
#####################################################################################

# Convert lists of lists to numpy arrays for easier computation
all_runs_train_loss = np.array(all_runs_train_loss)
all_runs_test_loss = np.array(all_runs_test_loss)

# Calculate mean, min, and max across all runs for each epoch
mean_train_loss = np.mean(all_runs_train_loss, axis=0)
min_train_loss = np.min(all_runs_train_loss, axis=0)
max_train_loss = np.max(all_runs_train_loss, axis=0)

mean_test_loss = np.mean(all_runs_test_loss, axis=0)
min_test_loss = np.min(all_runs_test_loss, axis=0)
max_test_loss = np.max(all_runs_test_loss, axis=0)

# Create the plot
plt.figure(figsize=(12, 8))
epochs = np.arange(0, args.nEpochs)

# Plot mean lines
plt.plot(epochs, mean_train_loss, 'r-', label='Mean Train Loss')
plt.plot(epochs, mean_test_loss, 'b-', label='Mean Validation Loss')

# Create shaded grey areas for the range (min to max)
plt.fill_between(epochs, min_train_loss, max_train_loss, color='r', alpha=0.2, label='Train Loss Range')
plt.fill_between(epochs, min_test_loss, max_test_loss, color='b', alpha=0.2, label='Validation Loss Range')

# Formatting
plt.title(f'Train and Validation Loss over {args.nRuns} Runs')
plt.xlabel('Epoch Count')
plt.ylabel('Loss')
plt.grid(True)
plt.legend(loc='upper right')

# Save the final aggregated plot
output_filename = os.path.join(result_path, f'aggregated_loss_{args.nRuns}_runs.jpg')
plt.savefig(output_filename)
print(f"Aggregated loss plot saved to: {output_filename}")

plt.show()
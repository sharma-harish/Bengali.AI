import ast
import os
from dataset import BengaliDatasetTrain
from model_dispatcher import MODEL_DISPATCHER
import torch
import torch.nn as nn
from tqdm import tqdm
import pdb

DEVICE = 'cuda'
TRAIN_FOLDS_CSV = os.environ.get('TRAIN_FOLDS')
IMG_HT = int(os.environ.get('IMG_HT'))
IMG_WD = int(os.environ.get('IMG_WD'))
EPOCHS = int(os.environ.get('EPOCHS'))

TRAIN_BAT_SIZE = int(os.environ.get('TRAIN_BAT_SIZE'))
TEST_BAT_SIZE = int(os.environ.get('TEST_BAT_SIZE'))

MODEL_MEAN = ast.literal_eval(os.environ.get('MODEL_MEAN'))
MODEL_STD = ast.literal_eval(os.environ.get('MODEL_STD'))

TRAIN_FOLDS = ast.literal_eval(os.environ.get('TRAIN_FOLDS'))
VALID_FOLDS = ast.literal_eval(os.environ.get('VALID_FOLDS'))
BASE_MODEL = os.environ.get('BASE_MODEL')

def train(dataset, data_loader, model, optimizer):
    model.train()
    # pdb.set_trace()
    for bi, d in tqdm(enumerate(data_loader), total = int(len(dataset) / data_loader.batch_size)):
        # pdb.set_trace()
        image = d['image']
        grapheme_root = d['grapheme_root']
        vowel_diacritic = d['vowel_diacritic']
        consonant_diacritic = d['consonant_diacritic']
        
        image = image.to(DEVICE, dtype = torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        # pdb.set_trace()
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

def eval(dataset, data_loader, model):
    model.eval()
    final_loss = 0
    counter = 0
    for bi, d in tqdm(enumerate(data_loader), total = int(len(dataset) / data_loader.batch_size)):
        counter = counter+1
        image = d['image']
        grapheme_root = d['grapheme_root']
        vowel_diacritic = d['vowel_diacritic']
        consonant_diacritic = d['consonant_diacritic']
        
        image = image.to(DEVICE, dtype = torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_function(outputs, targets)
        final_loss += loss
    return final_loss/counter

def loss_function(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    # pdb.set_trace()
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    return (l1 + l2 + l3) / 3

def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained = True)
    model.to(DEVICE)

    train_dataset = BengaliDatasetTrain(
        folds = TRAIN_FOLDS,
        img_ht = IMG_HT,
        img_wd = IMG_WD,
        mean = MODEL_MEAN,
        std = MODEL_STD
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size = TRAIN_BAT_SIZE,
        shuffle = True,
        num_workers = 4
    )

    valid_dataset = BengaliDatasetTrain(
        folds = VALID_FOLDS,
        img_ht = IMG_HT,
        img_wd = IMG_WD,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size = TEST_BAT_SIZE,
        shuffle = False,
        num_workers = 4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', 
                patience = 5, factor = 0.3, verbose=True)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)


    #early stopping

    for epoch in range(EPOCHS):
        # pdb.set_trace()
        train(train_dataset, train_loader, model, optimizer)
        # pdb.set_trace()
        with torch.no_grad():
            val_score = eval(valid_dataset, valid_loader, model)
        scheduler.step(val_score)
        torch.save(model.state_dict(), f'{BASE_MODEL}_fold{VALID_FOLDS[0]}.bin')

if __name__ == "__main__":
    main()
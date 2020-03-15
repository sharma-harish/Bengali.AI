set CUDA_VISIBLE_DEVICES=0,1
set IMG_HT=137
set IMG_WD=236
set EPOCHS=50
set TRAIN_BAT_SIZE=64
set TEST_BAT_SIZE=32
set MODEL_MEAN=(0.485, 0.456, 0.406)
set MODEL_STD=(0.229, 0.224, 0.225)
set BASE_MODEL=resnet34
set TRAIN_FOLDS_CSV=F:\Workspace\Bengali.AI\input\train_folds.csv

set TRAIN_FOLDS=(0, 1, 2, 3)
set VALID_FOLDS=(4,)
python train.py

REM set TRAIN_FOLDS=(0, 2, 3, 4)
REM set VALID_FOLDS=(1,)
REM python train.py

REM set TRAIN_FOLDS=(0, 1, 4, 3)
REM set VALID_FOLDS=(2,)
REM python train.py

REM set TRAIN_FOLDS=(0, 1, 4, 2)
REM set VALID_FOLDS=(3,)
REM python train.py

REM set TRAIN_FOLDS=(2, 1, 4, 3)
REM set VALID_FOLDS=(0,)
REM python train.py
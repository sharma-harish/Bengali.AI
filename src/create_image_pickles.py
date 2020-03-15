import joblib
import pandas as pd
from tqdm import tqdm
import glob

if __name__ == "__main__":
    files = glob.glob('F:\\Workspace\\Bengali.AI\\input\\train_*.parquet')
    for f in files:
        df = pd.read_parquet(f)
        image_ids = df.image_id.values
        df = df.drop('image_id', axis = 1)
        image_array = df.values
        for j, img_id in tqdm(enumerate(image_ids), total = len(image_ids)):
            joblib.dump(image_array[j,:], f'F:\\Workspace\\Bengali.AI\\input\\image_pickles\\{img_id}.pkl')

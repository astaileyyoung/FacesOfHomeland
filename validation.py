from pathlib import Path 
from argparse import ArgumentParser

import cv2 
import pandas as pd
from PIL import Image 


def parse_dir(d):
    subdirs = [x for x in Path(d).iterdir()]
    data = []
    for subdir in subdirs:
        files = [x for x in subdir.iterdir()]
        for file in files:
            datum = {'fp': str(file),
                     'cluster': subdir.parts[-1]}
            data.append(datum)
    return data 


def main(args):
    data = []
    for d in Path(args.src).iterdir():
        datum = parse_dir(d)
        data.extend(datum)
    
    df = pd.DataFrame(data)
    for idx, row in df.iterrows():
        img = cv2.imread(str(row['fp']))
        cv2.imshow('frame', img)
        # k = cv2.waitKey(0)
        # if k == ord('q'):
        #     cv2.destroyAllWindows()
        #     continue



if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    args = ap.parse_args()
    main(args)

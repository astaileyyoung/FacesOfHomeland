import ast 
import shutil
from pathlib import Path 

import pandas as pd
from tqdm import tqdm 

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm 

import sklearn.cluster as cluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_images(files):
    data = []
    for file in tqdm(files):
        temp, frame_num, face_num = file.stem.split('_')
        s = int(temp[1:3])
        e = int(temp[4:6])
        datum = {'fp': str(file.absolute().resolve()),
                 'season': s,
                 'episode': e,
                 'frame_num': int(frame_num),
                 'face_num': int(face_num)
                }
        data.append(datum)
    fp_df = pd.DataFrame(data)
    return fp_df


def scatter_thumbnails(data, images, scale_factor=16, colors=None, show_images=True):
    assert len(data) == len(images)

    # reduce embedding dimentions to 2
    x = PCA(n_components=2).fit_transform(data) if len(data[0]) > 2 else data

    # create a scatter plot.
    f = plt.figure(figsize=(22, 15))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], s=4)
    _ = ax.axis('off')
    _ = ax.axis('tight')

    if show_images:
        # add thumbnails :)
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        for i in tqdm(range(len(images))):
            image = plt.imread(images[i])
            h, w = image.shape[:2]
            n_h = int(h/scale_factor)
            n_w = int(w/scale_factor)
            image = cv2.resize(image, (n_w, n_h))
            outputImage = cv2.copyMakeBorder(
                 image, 
                 10, 
                 10, 
                 10, 
                 10, 
                 cv2.BORDER_CONSTANT, 
                 value=colors[i] if colors is not None else None)
            im = OffsetImage(image)
            bboxprops = dict(edgecolor=colors[i]) if colors is not None else None
            ab = AnnotationBbox(im, x[i], xycoords='data',
                                frameon=(bboxprops is not None),
                                pad=0.1,
                                bboxprops=bboxprops)
            ax.add_artist(ab)
    return ax


def save_images(df,
                dst):
    
    if not dst.exists():
        Path.mkdir(dst, parents=True)
        
    for idx, row in tqdm(episode_fp.iterrows(), total=episode_fp.shape[0]):
        fp = dst.joinpath(f'{row["label"]}/{Path(row["fp"]).name}')
        if not fp.parent.exists():
            Path.mkdir(fp.parent)
        shutil.copy(row["fp"], fp)


def save_fig(i, 
             j,
             fig_dir):
    episode = f'S{str(i).zfill(2)}E{str(j).zfill(2)}'
    name = f'{episode}.png'
    fig_fp = fig_dir.joinpath(name)
    plt.save_fig(str(fig_fp), dpi=300)


def main():
    df = pd.read_csv('./data/faces.csv', index_col=0)
    episode_df = pd.read_csv('./data/episodes.csv', index_col=0)
    df = df.merge(episode_df[['imdb_id', 'cast']],
                  on='imdb_id',
                  how='left'
                 )
    files = [x for x in Path('./data/images').iterdir()]
    fp_df = get_images(files)
    df_fp = df.merge(fp_df,
                        on=['frame_num', 'face_num', 'season', 'episode'],
                        how='inner')
    
    fig_dir = Path('./data/figures')
    if not fig_dir.exists():
        Path.mkdir(fig_dir)

    clustered_dir = Path('./data/figures_clustered')
    if not clustered_dir.exists():
        Path.mkdir(clustered_dir)

    total = 8 * 12
    pb = tqdm(total=total)
    for i in range(8):
        for j in range(12):
            episode_fp = df_fp[(df_fp['season'] == i) & (df['episode'] == j)]

            data = np.array([np.array(ast.literal_eval(x)) for x in episode_fp['encoding'].tolist()])
            faces = episode_fp['fp'].tolist()
        
            x = PCA(n_components='mle', svd_solver='full').fit_transform(data)
            tsne = TSNE(perplexity=50,
                        n_components=2,
                        learning_rate=50,
                        n_iter=10000,
                        early_exaggeration=300,
                        # method='exact',
                        n_iter_without_progress=300
                     )
            x = tsne.fit_transform(data)
            ax = scatter_thumbnails(x, faces, scale_factor=16, show_images=True)
            save_fig(i, j, fig_dir)
            
            labels, colors = get_clusters(x, cluster.DBSCAN, n_jobs=-1, eps=4.5, min_samples=15)
            
            ax = scatter_thumbnails(x, faces, scale_factor=16, show_images=True)
            save_fig(i, j, clustered_dir)

            dst = Path(f'./data/clustering/{name}')
            save_images(episode_fp, dst)
            pb.update()


main()

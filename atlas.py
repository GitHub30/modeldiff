
def run(model, ops):

    import numpy as np
    import tensorflow as tf
    import math
    import urllib.parse
    import sklearn

    from umap import UMAP

    from lucid.misc.io import load, show, save

    from clarity.dask.cluster import get_client

    import lucid.optvis.objectives as objectives
    import lucid.optvis.param as param
    import lucid.optvis.render as render
    import lucid.optvis.transform as transform

    from lucid.modelzoo.vision_models import InceptionV1, AlexNet
    import matplotlib.pyplot as plt
    from lucid.misc.io.writing import write_handle

    from clarity.utils.distribute import DistributeDask, DistributeMPI
    from lucid.modelzoo.nets_factory import models_map, get_model

    # Produced by the "collect_activations" notebook
    def load_activations(model, op_name, num_activations=100, batch_size=4096, num_activations_per_image=1):
        activations_collected_per_image = 16 # This is hardcoded from the collection process
        if num_activations_per_image > activations_collected_per_image:
            raise ValueError("Attempting to use more activations than were collected per image.")
        activations = []
        coordinates = []
        for s in range(0, math.ceil(num_activations / num_activations_per_image), batch_size):
            e = s + batch_size
            # acts_per_image=16&end=1003520&model=AlexNet&sampling_strategy=random&split=train&start=999424
            loaded_activations = load(f"gs://openai-clarity/encyclopedia/collect_activations/acts_per_image=16&end={e}&model={model.name}&sampling_strategy=random&split=train&start={s}/{op_name}-activations.npy")
            loaded_coordinates = load(f"gs://openai-clarity/encyclopedia/collect_activations/acts_per_image=16&end={e}&model={model.name}&sampling_strategy=random&split=train&start={s}/{op_name}-image_crops.npy")
            
            activations.append(loaded_activations[:,0:num_activations_per_image,:]) 
            coordinates.append(loaded_coordinates[:,0:num_activations_per_image,:]) 
        acts = np.concatenate(activations)
        flattened_acts = acts.reshape((acts.shape[0] * acts.shape[1], acts.shape[2]))
        
        coords = np.concatenate(coordinates)
        flattened_coords = coords.reshape((coords.shape[0] * coords.shape[1], coords.shape[2]))
        return flattened_acts[:num_activations, ], flattened_coords[:num_activations, ]
        
    def load_ops(model):

        # Load the metadata info so we can get a list of the ops
        metadata = load(f"gs://openai-clarity/encyclopedia/graph_metadata/model={model.name}/metadata.json")
        # Filter the ops list to only the ones that we are interested in
        ops = [(op_key, op['channels']) for op_key, op in metadata['ops'].items() if op['op_type'] in ('Relu', 'Conv2D') and op['rank'] == 4]
        return ops

    def bin_laid_out_activations(layout, activations, partition, grid_size, threshold=5):

        n = activations.shape[0]

        assert layout.shape[0] == activations.shape[0]
        assert n % 2 == 0

        # calculate which grid cells each activation's layout position falls into
        # first bin stays empty because nothing should be < 0, so we add an extra bin
        bins = np.linspace(0, 1, num=grid_size + 1)
        bins[-1] = np.inf  # last bin should include all higher values
        indices = np.digitize(layout, bins) - 1  # subtract 1 to account for empty first bin

        means_x, means_y, coordinates, counts_x, counts_y = [], [], [], [], []

        grid_coordinates = np.indices((grid_size, grid_size)).transpose().reshape(-1, 2)
        for xy_coordinates in grid_coordinates:
            mask = np.equal(xy_coordinates, indices).all(axis=1)
            count_x = np.count_nonzero(mask[0:n//2])
            count_y = np.count_nonzero(mask[n//2:])
            if (count_x + count_y) > threshold:
                counts_x.append(count_x)
                counts_y.append(count_y)
                coordinates.append(xy_coordinates)
                means_x.append(np.average(activations[0:n//2][mask[0:n//2]], axis=0)[0:partition])
                means_y.append(np.average(activations[n//2:][mask[n//2:]], axis=0)[partition:])

        return coordinates, means_x, means_y, counts_x, counts_y

    def get_optimal_maps(X, Y):

        Σ_XX = X.transpose() @ X
        Σ_XY = X.transpose() @ Y
        Σ_YY = Y.transpose() @ Y
        Σ_YX = Σ_XY.transpose()

        A_XY = Σ_XY @ np.linalg.inv(Σ_YY)
        A_YX = Σ_YX @ np.linalg.inv(Σ_XX)

        Xhat = Y @ A_XY.transpose()
        Yhat = X @ A_YX.transpose()

        errx = np.sqrt(np.mean((Y @ A_XY.transpose() - X)**2))
        erry = np.sqrt(np.mean((X @ A_YX.transpose() - Y)**2))

        err_baseline_x = np.sqrt(np.mean((X - np.mean(X,0))**2))
        err_baseline_y = np.sqrt(np.mean((Y - np.mean(Y,0))**2))

        print(errx, err_baseline_x)
        print(erry, err_baseline_y)

        return A_XY, A_YX, Xhat, Yhat, (errx, err_baseline_x), (erry, err_baseline_y)

    def dim_reduce(Z, method="umap"):

        if method == "svd":
            U,S,V = np.linalg.svd(Z, full_matrices = False)
            layout = U[:,0:2]
            return layout

        if method == "umap":
            umap_defaults = dict(
                n_components=2, n_neighbors=50, min_dist=0.05, verbose=True, metric="cosine"
            )
            layout = UMAP(**umap_defaults).fit_transform(Z)
            return layout

    def get_atlas(model, ops):

        model_x, model_y = get_model(model[0]), get_model(model[1])

        model_x.load_graphdef()
        model_y.load_graphdef()

        X = np.concatenate([load_activations(model_x, op, num_activations=50000)[0] for op in ops[0]],1).astype(np.float32)
        Y = np.concatenate([load_activations(model_y, op, num_activations=50000)[0] for op in ops[1]],1).astype(np.float32)

        A_XY, A_YX, Xhat, Yhat, errx, erry = get_optimal_maps(X,Y)

        Xc = np.concatenate([X, Yhat], axis = -1)
        Yc = np.concatenate([Xhat, Y], axis = -1)

        Z  = np.concatenate([Xc, Yc])

        layout = dim_reduce(Z, method="umap")

        layout_centered = (layout - np.min(layout,0))
        layout_centered = layout_centered/np.max(layout_centered,0)

        coordinates, means_x, means_y, counts_x, counts_y = bin_laid_out_activations(layout_centered, Z, X.shape[1], 20)

        coordinates_x = np.array(coordinates)
        counts_x      = np.array(counts_x)
        counts_y      = np.array(counts_y)
        means_x       = np.array(means_x)
        means_y       = np.array(means_y)

        return coordinates, means_x, means_y, counts_x, counts_y, errx, erry, A_XY, A_YX, layout_centered

    import json
    import hashlib
    identifier = hashlib.md5(json.dumps((model, ops)).encode('utf-8')).hexdigest()

    def pre_relu(name):
        if "mixed" in name:
            return (f"{name}_1x1:0", f"{name}_3x3:0", f"{name}_5x5:0", f"{name}_pool_reduce:0")
        else:
            return [name + ":0"]

    coordinates, means_x, means_y, counts_x, counts_y, errx, erry, A_XY, A_YX, layout = \
        get_atlas(model, [pre_relu(ops[0]), pre_relu(ops[1])])

    plt.figure(figsize=(10, 10))
    plt.scatter(layout[0:layout.shape[0]//2,0], layout[0:layout.shape[0]//2,1], 1, "b")
    plt.scatter(layout[layout.shape[0]//2:,0], layout[layout.shape[0]//2:,1], 1, "r")
    plt.show()
    with write_handle(f"gs://clarity-public/ggoh/Diff/{identifier}/scatter.png") as handle:
        plt.savefig(handle)

    manifest = {"model_x": model[0],
                "model_y": model[1],
                "ops_x": ops[0],
                "ops_y": ops[1],
                "coordinates": coordinates, 
                "counts_x": counts_x,
                "counts_y": counts_y,
                "means_x": means_x,
                "means_y": means_y,
                "err_x": errx,
                "err_y": erry,
                "layout": layout,
                "A_XY": A_XY,
                "A_YX": A_YX,
                "identifier": identifier}

    print("Identifier", identifier)
    print(save(manifest, f"gs://clarity-public/ggoh/Diff/{identifier}/manifest.json"))

    del manifest["means_x"]
    del manifest["means_y"]
    del manifest["A_XY"]
    del manifest["A_YX"]

    manifest["layout"] = np.concatenate([layout[0:5000], layout[layout.shape[0]//2:layout.shape[0]//2+5000]]).astype(np.float16)

    print(save(manifest, f"gs://clarity-public/ggoh/Diff/{identifier}/manifest_slim.json"))

if __name__ == "__main__":
    import datetime
    import os
    from rcall import meta
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('backend')
    args = parser.parse_args()

    # model = ("InceptionV1", "InceptionV1_caffe_Places205")

    # if args.backend == "interactive":
    #     ops   = ("mixed4e", "inception_4e_1x1/inception_4e_1x1")
    #     run(model, ops)
    # else:
    #     for op_x in ["mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed5a", "mixed5b"]:
    #         for op_y in ["inception_4a_1x1/inception_4a_1x1", "inception_4b_1x1/inception_4b_1x1", "inception_4c_1x1/inception_4c_1x1", "inception_4d_1x1/inception_4d_1x1", "inception_4e_1x1/inception_4e_1x1", "inception_5a_1x1/inception_5a_1x1", "inception_5b_1x1/inception_5b_1x1"]:
    #             meta.call(
    #                 backend=args.backend,
    #                 fn=run,
    #                 args = [ model, [op_x, op_y] ],
    #                 log_relpath='atlas_%s' % datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%s'),
    #                 num_cpu='auto',
    #                 mpi_proc_per_machine=1,
    #                 mpi_machines=1,
    #                 num_gpu=1)

    model = ("InceptionV1", "InceptionV1")

    if args.backend == "interactive":
        ops = ("mixed3a", "mixed4c")
        run(model, ops)
    else:
        for op_x in ["mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed5a", "mixed5b", "head0_bottleneck", "head1_bottleneck"]:
            for op_y in ["mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed5a", "mixed5b", "head0_bottleneck", "head1_bottleneck"]:
                meta.call(
                    backend=args.backend,
                    fn=run,
                    args = [ model, [op_x, op_y] ],
                    log_relpath='atlas_%s' % datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%s'),
                    num_cpu='auto',
                    num_gpu=1)


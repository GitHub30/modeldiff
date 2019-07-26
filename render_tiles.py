def run(identifier):

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import numpy as np
    import tensorflow as tf
    import math

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

    def spritesheet(imgs):
        k = int(np.ceil(np.sqrt(imgs.shape[0])))
        Z = np.zeros((k**2, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
        Z[0:imgs.shape[0]] = imgs
        Z = np.reshape(Z, (k,k,imgs.shape[1], imgs.shape[2], imgs.shape[3]))
        Z = np.concatenate(Z, 1)
        Z = np.concatenate(Z, 1)
        return Z

    def render_atlas_tile(model,op_name,directions,icon_size=45,n_steps=127,transforms_amount=1,cossim_pow=0,L2_amount=2):      
        transforms_options = [
            [
                transform.jitter(2)
            ],
            [
                transform.pad(12, mode="constant", constant_value=.5),
                transform.jitter(8),
                transform.random_scale([1 + (i - 5) / 50. for i in range(11)]),
                transform.random_rotate(list(range(-10, 11)) + 5 * [0]),
                transform.jitter(4),
            ],
            [
                transform.pad(2, mode='constant', constant_value=.5),
                transform.jitter(4),
                transform.jitter(4),
                transform.jitter(8),
                transform.jitter(8),
                transform.jitter(8),
                transform.random_scale([0.995**n for n in range(-5,80)] + [0.998**n for n in 2*list(range(20,40))]),
                transform.random_rotate(list(range(-20,20))+list(range(-10,10))+list(range(-5,5))+5*[0]),
                transform.jitter(2),
            ],
        ]
        
        param_f = lambda: param.image(icon_size, batch=directions.shape[0])
        obj = objectives.Objective.sum(
          [objectives.direction_neuron(op_name, v, batch=n, cossim_pow=cossim_pow)
           for n,v in enumerate(directions)
          ]) - L2_amount * objectives.L2("input", 0.5) * objectives.L2("input", 0.5)
        thresholds=(n_steps//2, n_steps)

        vis_imgs = render.render_vis(model, obj, param_f, transforms=transforms_options[transforms_amount], thresholds=thresholds, verbose=False)[-1]

        return vis_imgs

    def draw_atlas(A, coordinates):

        grid_size = np.max(np.array(coordinates)) + 1
        canvas = np.ones((grid_size*A.shape[1], grid_size*A.shape[1],3))

        def slice_xy(x,y,img):
            s = A.shape[1]
            canvas[s*x:s*(x+1), s*y:s*(y+1),:] = img

        for i in range(len(coordinates)):
            slice_xy(coordinates[i][0], coordinates[i][1], A[i])

        return canvas

    def run_xy(model, layername, filename, means, coordinates, identifier):

        cluster = DistributeMPI()

        for i in range(means.shape[0]):
            cluster.submit(render_atlas_tile, model, layername, means[i:i+1])

        results = cluster.run()

        if cluster.is_master():
            r = np.array(results)[:,0,:,:,:]

            # Save spritesheet
            result1 = spritesheet(r)
            path = f"clarity-public/ggoh/Diff/{identifier}/atlas_{filename}.webp"
            save(result1, "gs://{}".format(path))
            print("https://storage.googleapis.com/{}".format(path))

            # Save filename
            print(save(r, f"gs://clarity-public/ggoh/Diff/{identifier}/r{filename}z.npy"))

            # Save coordinates
            canvas = draw_atlas(r, coordinates)
            print(save(canvas, f"gs://clarity-public/ggoh/Diff/{identifier}/rendered_atlas_{filename}z.webp"))

        cluster.comm.Barrier()

    manifest = load(f"https://storage.googleapis.com/clarity-public/ggoh/Diff/{identifier}/manifest.json")
    model_x, model_y, ops_x, ops_y = get_model(manifest['model_x']), get_model(manifest['model_y']), manifest['ops_x'], manifest['ops_y']
    model_x.load_graphdef()
    model_y.load_graphdef()

    means_x = np.nan_to_num(np.array(manifest["means_x"])) # Replace Nan's (no elements in bin with 0's)
    means_y = np.nan_to_num(np.array(manifest["means_y"])) 
    
    run_xy(model_y, ops_y, "y", means_y, manifest["coordinates"], identifier)
    run_xy(model_x, ops_x, "x", means_x, manifest["coordinates"], identifier)

if __name__ == "__main__":

    import datetime
    import os
    from rcall import meta
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('backend')
    args = parser.parse_args()

    model = ("InceptionV1", "InceptionV1_caffe_Places205")
    model = ("InceptionV1", "InceptionV1")

    if args.backend == "interactive":
        run("d54e7a6ca24df2af33af4de0d63482aa")
    else:
        # for op_x in ["mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed5a", "mixed5b"]:
        #     for op_y in ["inception_4a_1x1/inception_4a_1x1", "inception_4b_1x1/inception_4b_1x1", "inception_4c_1x1/inception_4c_1x1", "inception_4d_1x1/inception_4d_1x1", "inception_4e_1x1/inception_4e_1x1", "inception_5a_1x1/inception_5a_1x1", "inception_5b_1x1/inception_5b_1x1"]:
        #         import json
        #         import hashlib
        #         ops = [op_x, op_y]
        #         identifier = hashlib.md5(json.dumps((model, ops)).encode('utf-8')).hexdigest()
        #         meta.call(
        #             backend=args.backend,
        #             fn=run,
        #             args = [ identifier ],
        #             log_relpath='atlas_%s' % datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%s'),
        #             num_cpu='auto',
        #             mpi_proc_per_machine=8,
        #             mpi_machines=1,
        #             num_gpu=8)
        for op_x in ["mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed5a", "mixed5b"]:
            for op_y in ["mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed5a", "mixed5b"]:
                import json
                import hashlib
                ops = [op_x, op_y]
                identifier = hashlib.md5(json.dumps((model, ops)).encode('utf-8')).hexdigest()
                meta.call(
                    backend=args.backend,
                    fn=run,
                    args = [ identifier ],
                    log_relpath='atlas_%s' % datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%s'),
                    num_cpu='auto',
                    mpi_proc_per_machine=8,
                    mpi_machines=1,
                    num_gpu=8)


# D = load("gs://clarity-public/ggoh/diff/temp1/means.json")

# coordinates = D['coordinates']

# canvas = np.ones((40*A.shape[1], 40*A.shape[1],3))
# A = load("gs://clarity-public/ggoh/diff/temp1/ry.npy")

# def slice_xy(x,y,img):
#     s = A.shape[1]
#     canvas[s*x:s*(x+1), s*y:s*(y+1),:] = img

# for i in range(len(coordinates)):
#     slice_xy(coordinates[i][0], coordinates[i][1], A[i])

# save(canvas, "gs://clarity-public/ggoh/temp.webp")



# run()

# if cluster.is_master():
#     result1 = spritesheet(np.array([r[0][0] for r in results]))
#     path = f"clarity-public/ggoh/diff/temp1/atlas_x.webp"
#     save(result1, "gs://{}".format(path))
#     print("https://storage.googleapis.com/{}".format(path))

#     result2 = spritesheet(np.array([r[1][0] for r in results]))
#     path = f"clarity-public/ggoh/diff/temp1/atlas_y.webp"
#     save(result2, "gs://{}".format(path))
#     print("https://storage.googleapis.com/{}".format(path))

#     save(np.array([r[0][0] for r in results]), "gs://clarity-public/ggoh/diff/temp1/rx.npy")
#     save(np.array([r[1][0] for r in results]), "gs://clarity-public/ggoh/diff/temp1/ry.npy")

# D = load("gs://clarity-public/ggoh/diff/temp1/means.json")

# coordinates = D['coordinates']


# save(canvas, "gs://clarity-public/ggoh/temp.webp")


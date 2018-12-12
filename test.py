from tensorflow.python import pywrap_tensorflow
import numpy as np

if __name__ == "__main__":
    # pretrain_model_path = './models/pretrain_model/inception_v4.ckpt'
    # reader = pywrap_tensorflow.NewCheckpointReader(pretrain_model_path)
    # keys = reader.get_variable_to_shape_map().keys()
    # print(len(keys))
    # print(keys)

    a = np.arange(0,20).reshape([4,5])

    print(a[:, 0::5])
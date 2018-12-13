import numpy as np
import tensorflow as tf
from lib.dataset.dataload import Dataload
from lib.network.solver_wrapper import SloverWrapper
from lib.utils.config import cfg
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.75
with tf.Session(config=config) as sess:
    s = SloverWrapper(sess)
    s.train_model()

#
# data_dict = np.load(cfg["TRAIN"]["PRETRAIN_MODEL"], encoding='latin1').item()
# for key in data_dict:
#     with tf.variable_scope(key, reuse=True):
#         print('key', key)
#         for subkey in data_dict[key]:
#             print(subkey)
            # try:
            #     var = tf.get_variable(subkey)
            #     session.run(var.assign(data_dict[key][subkey]))
            #     print("assign pretrain model "+subkey+ " to "+key)
            # except ValueError:
            #     print("ignore "+key)
            #     if not ignore_missing:
            #         raise
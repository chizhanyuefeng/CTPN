from lib.network.solver_wrapper import SloverWrapper
import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.75
with tf.Session(config=config) as sess:
    s = SloverWrapper(sess)
    s.train_model()
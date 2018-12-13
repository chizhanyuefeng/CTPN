import os
import time
import logging
import tensorflow as tf
import numpy as np

from lib.utils.config import cfg
from lib.network.ctpn_network import CTPN
from lib.dataset.dataload import Dataload

class SloverWrapper(object):

    def __init__(self, sess):
        self.sess = sess
        self.network = CTPN(is_train=True)
        self.model_output_dir = cfg["TRAIN"]["MODEL_OUTPUT_DIR"]
        # self.logs_dir = cfg["TRAIN"]["LOGS_DIR"]
        self.pretrain_model = cfg["TRAIN"]["PRETRAIN_MODEL"]

        self.writer = tf.summary.FileWriter(logdir=cfg["TRAIN"]["LOGS_DIR"],
                                            graph=tf.get_default_graph(),
                                            flush_secs=5)

        self.train_logger = self._train_logger_init()

    def train_model(self):

        train_data_load = Dataload(cfg["TRAIN"]["TRAIN_IMG_DIR"], cfg["TRAIN"]["TRAIN_LABEL_DIR"])
        total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = self.network.build_loss()

        # scalar summary
        self.saver = tf.train.Saver(max_to_keep=100, write_version=tf.train.SaverDef.V2)
        tf.summary.scalar('rpn_reg_loss', rpn_loss_box)
        tf.summary.scalar('rpn_cls_loss', rpn_cross_entropy)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)
        summary_op = tf.summary.merge_all()

        # optimizer
        lr = tf.Variable(cfg["TRAIN"]["LEARNING_RATE"], trainable=False)
        if cfg["TRAIN"]["SOLVER"] == 'Adam':
            opt = tf.train.AdamOptimizer(cfg.TRAIN.LEARNING_RATE)
        elif cfg.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(cfg.TRAIN.LEARNING_RATE)
        else:
            momentum = cfg.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(lr, momentum)

        global_step = tf.Variable(0, trainable=False)
        with_clip = True
        if with_clip:
            tvars = tf.trainable_variables()
            grads, norm = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 10.0)
            train_op = opt.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
        else:
            train_op = opt.minimize(total_loss, global_step=global_step)

        # intialize variables
        self.sess.run(tf.global_variables_initializer())
        restore_iter = 0

        self.network.load(cfg["TRAIN"]["PRETRAIN_MODEL"], self.sess, True)

        # resuming a trainer
        if cfg["TRAIN"]["RESTORE"]:
            try:
                ckpt = tf.train.get_checkpoint_state(cfg["TRAIN"]["MODEL_OUTPUT_DIR"])
                self.train_logger.info('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                restore_iter = int(stem.split('_')[-1])
                self.sess.run(global_step.assign(restore_iter))
                self.train_logger.info('done')
            except:
                raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

        start_time = time.time()
        for iter in range(restore_iter, cfg["TRAIN"]["MAX_STEPS"]):
            # learning rate
            if iter != 0 and iter % cfg["TRAIN"]["STEPSIZE"] == 0:
                self.sess.run(tf.assign(lr, lr.eval() * cfg["TRAIN"]["GAMMA"]))

            img_input, labels, img_info = train_data_load.getbatch()
            # print(img_input)
            feed_dict = {self.network.img_input: img_input, self.network.gt_boxes: labels, self.network.im_info: img_info}

            fetch_list = [total_loss, model_loss, rpn_cross_entropy, rpn_loss_box,
                          summary_op,
                          train_op]

            total_loss_val, model_loss_val, rpn_loss_cls_val, rpn_loss_box_val, \
            summary_str, _ = self.sess.run(fetches=fetch_list, feed_dict=feed_dict)

            self.writer.add_summary(summary=summary_str, global_step=global_step.eval())

            if iter % cfg["TRAIN"]["DISPLAY"] == 0:
                end_time = time.time()
                self.train_logger.info('iter: %d / %d, total loss: %.4f, model loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, lr: %f'%\
                        (iter, cfg["TRAIN"]["MAX_STEPS"], total_loss_val, model_loss_val, rpn_loss_cls_val, rpn_loss_box_val, lr.eval()))
                self.train_logger.info('speed: {:.3f}s / iter'.format((end_time-start_time)/cfg["TRAIN"]["DISPLAY"]))
                start_time = time.time()

            if (iter+1) % cfg["TRAIN"]["SNAPSHOT_ITERS"] == 0:
                if not os.path.exists(self.model_output_dir):
                    os.makedirs(self.model_output_dir)
                file_name = "CTPN_{}_iter_{}.ckpt".format(cfg["BACKBONE"], iter)
                self.saver.save(self.sess, os.path.join(self.model_output_dir, file_name))
                print('Wrote snapshot to: {:s}'.format(self.model_output_dir))

    def _train_logger_init(self):
        """
        初始化log日志
        :return:
        """
        train_logger = logging.getLogger('train')
        train_logger.setLevel(logging.DEBUG)

        # 添加文件输出
        log_file = cfg["TRAIN"]["LOGS_DIR"] + time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.logs'
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        train_logger.addHandler(file_handler)

        # 添加控制台输出
        consol_handler = logging.StreamHandler()
        consol_handler.setLevel(logging.DEBUG)
        consol_formatter = logging.Formatter('%(message)s')
        consol_handler.setFormatter(consol_formatter)
        train_logger.addHandler(consol_handler)
        return train_logger


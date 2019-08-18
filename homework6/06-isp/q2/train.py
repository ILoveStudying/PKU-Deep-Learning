#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
from model import Model
from dataset import Dataset
from skimage.measure import compare_psnr
from common import config
from utils import from_sub_pixel_to_img


def get_dataset_batch(ds_name):
    dataset = Dataset(ds_name)
    ds_gnr = dataset.load().instance_generator
    ds = tf.data.Dataset.from_generator(ds_gnr, output_types=(tf.float32, tf.float32),)
    if ds_name == 'train':
        ds = ds.shuffle(dataset.instances_per_epoch)
        ds = ds.repeat(config.nr_epoch)
        minibatch_size = config.minibatch_size
    elif ds_name == 'test':
        ds = ds.repeat(config.nr_epoch // config.test_interval)
        minibatch_size = 1
    ds = ds.batch(minibatch_size)
    ds_iter = ds.make_one_shot_iterator()
    sample_gnr = ds_iter.get_next()
    return sample_gnr, dataset

def squared_error_loss(img, img_restored):
    return tf.reduce_mean((img - img_restored) ** 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue', dest='continue_path', required=False)
    args = parser.parse_args()

    ## load dataset
    train_batch_gnr, train_set = get_dataset_batch(ds_name='train')

    test_gnr, test_set = get_dataset_batch(ds_name = 'test')
    ## build graph
    network = Model()
    placeholders, restored = network.build()
    gt_size = config.patch_size - config.edge
    gt = tf.placeholder(tf.float32, shape=(None, )+ (gt_size, gt_size)+ (config.nr_channel * config.ratio * config.ratio ,), name = 'gt')

    loss_squared = squared_error_loss(gt, restored)
    loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = loss_reg + loss_squared
    ## train config
    global_steps = tf.Variable(0, trainable=False)
    boundaries = [train_set.minibatchs_per_epoch*5, train_set.minibatchs_per_epoch*40]
    values = [0.0001, 0.0001, 0.0001]
    lr = tf.train.piecewise_constant(global_steps, boundaries, values)
    opt = tf.train.AdamOptimizer(lr)
    # in order to update BN in every iter

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train = opt.minimize(loss)

    ## init tensorboard
    tf.summary.scalar('loss_regularization', loss_reg)
    tf.summary.scalar('loss_error', loss - loss_reg)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', lr)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'tf_log', 'train'),
                                         tf.get_default_graph())

    ## create a session
    tf.set_random_seed(12345) # ensure consistent results
    global_cnt = 0
    epoch_start = 0
    g_list = tf.global_variables()
    saver = tf.train.Saver(var_list=g_list)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if args.continue_path:
            ckpt = tf.train.get_checkpoint_state(args.continue_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            epoch_start = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[1])
            global_cnt = epoch_start * train_set.minibatchs_per_epoch

        ## training
        file= open('./psnr_ratio3', 'w+')
        for epoch in range(epoch_start + 1, config.nr_epoch + 1):
            for _ in range(train_set.minibatchs_per_epoch):
                global_cnt += 1
                lr_images, sr_images = sess.run(train_batch_gnr)
                # 128*7*7*3,128*9*9*27
                feed_dict = {
                    placeholders['data']: lr_images[:, :, :, :1],
                    gt: sr_images[:, :, :, :1 * config.ratio * config.ratio],
                    global_steps: global_cnt,
                    placeholders['is_training']: True,
                }
                _, loss_v, loss_reg_v, lr_v, summary = sess.run([train, loss, loss_reg,
                                                                 lr, merged],
                                                                feed_dict=feed_dict)
                if global_cnt % config.show_interval == 0:
                    train_writer.add_summary(summary, global_cnt)
                    print(
                        "e:{},{}/{}".format(epoch, global_cnt % train_set.minibatchs_per_epoch,
                                            train_set.minibatchs_per_epoch),
                        'loss: {:.3f}'.format(loss_v),
                        'loss_reg: {:.3f}'.format(loss_reg_v),
                        'lr: {:.4f}'.format(lr_v),
                    )

            ## save model
            if epoch % config.snapshot_interval == 0:
                saver.save(sess, os.path.join(config.log_model_dir, 'epoch-{}'.format(epoch)),
                           global_step=global_cnt)

            if epoch % config.test_interval == 0:
                psnrs = []
                for _ in range(test_set.testing_minibatchs_per_epoch):
                    lr_image, hr_image = sess.run(test_gnr)
                    feed_dict = {
                        placeholders['data']: lr_image[:, :, :, :1],
                        placeholders['is_training']: False,
                    }
                    restored_v = sess.run([restored], feed_dict=feed_dict)
                    restored_img_y = from_sub_pixel_to_img(restored_v[0][0], config.ratio)
                    img = np.clip(restored_img_y[:, :, 0], 0, 1) * 255
                    img = img.astype('uint8')
                    cv2.imwrite('./output/{}.png'.format(global_cnt), img)
                    edge = int(config.edge / 2 * config.ratio)
                    psnr_y = compare_psnr(hr_image[0, edge:-edge, edge:-edge, :1], restored_img_y)
                    psnrs.append(psnr_y)
                    file.write(str(np.mean(psnrs)) + '\n')
                print('average psnr is {:2.2f} dB'.format(np.mean(psnrs)))
        print('Training is done, exit.')
        file.close()
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os._exit(1)

# vim: ts=4 sw=4 sts=4 expandtab

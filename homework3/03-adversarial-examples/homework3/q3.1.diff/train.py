#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import tensorflow as tf
import numpy as np

from custom_vgg16_bn import Model
from dataset import Dataset
from common import config
from attack import Attack
import cv2
from IPython import embed


def get_dataset_batch(ds_name):
    dataset = Dataset(ds_name)
    ds_gnr = dataset.load().instance_generator
    ds = tf.data.Dataset().from_generator(ds_gnr, output_types=(tf.float32, tf.int32), )
    ds = ds.repeat(config.nr_epoch)
    ds = ds.batch(config.minibatch_size)
    ds_iter = ds.make_one_shot_iterator()
    sample_gnr = ds_iter.get_next()
    return sample_gnr, dataset


def affine_transformation(img):
    img=img.reshape(config.image_shape + (config.nr_channel,))
    rows, cols, dims = img.shape
    choice = np.random.choice(['scale', 'rotate', 'shift', 'affine'])
    if choice == 'scale':
        # 放缩
        scale = np.random.choice([0.8, 0.9, 1.0, 1.1, 1.2])
        img = cv2.resize(img, dsize=(int(rows * scale), int(cols * scale)), interpolation=cv2.INTER_LINEAR)
    elif choice == 'rotate':
        # 旋转
        RotateMatrix = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=90, scale=1.2)
        img = cv2.warpAffine(img, RotateMatrix, (rows * 2, cols * 2))
    elif choice == 'shift':
        # 平移
        TranslationMatrix = np.float32([[1, 0, 5], [0, 1, 2]])
        img = cv2.warpAffine(img, TranslationMatrix, (rows, cols))
    elif choice == 'affine':
        # 仿射变换
        pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
        M_affine = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M_affine, (cols, rows))
    img = cv2.resize(img, config.image_shape)
    img = img.reshape((1,) +config.image_shape + (config.nr_channel,))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue', dest='continue_path', required=False)
    parser.add_argument('-l', '--loss', default='softmax')
    args = parser.parse_args()

    ## load dataset
    train_batch_gnr, train_set = get_dataset_batch(ds_name='train')

    data = tf.placeholder(tf.float32, shape=(None,) + config.image_shape + (config.nr_channel,), name='data')
    label = tf.placeholder(tf.int32, shape=(None,), name='label')  # placeholder for targetted label
    gt = tf.placeholder(tf.int32, shape=(None,), name='gt')

    pre_noise = tf.Variable(
        tf.zeros((config.minibatch_size, config.image_shape[0], config.image_shape[1], config.nr_channel),
                 dtype=tf.float32))
    model = Model()
    attack = Attack(model, config.minibatch_size)
    acc, loss, adv, x, logits = attack.generate_graph(pre_noise, data, gt, label)
    acc_gt = attack.evaluate(data, gt)

    placeholders = {
        'data': data,
        'label': label,
        'gt': gt,

    }

    lr = 1e-2
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(loss, [pre_noise])
    train = opt.apply_gradients(grads)
    ## init tensorboard
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'tf_log', 'train'),
                                         tf.get_default_graph())

    ## create a session
    tf.set_random_seed(12345)  # ensure consistent results
    global_cnt = 0
    epoch_start = 0
    succ = 0
    noise_l2 = 0
    with tf.Session() as sess:

        for idx in range(train_set.minibatches):
            global_cnt = 0

            sess.run(tf.global_variables_initializer())  # init all variables
            images, labels = sess.run(train_batch_gnr)

            for epoch in range(epoch_start + 1, config.nr_epoch + 1):
                global_cnt += 1
                feed_dict = {
                    placeholders['data']: images,
                    placeholders['label']: labels,
                    placeholders['gt']: labels,
                }
                _, accuracy, loss_batch, adv_examples, ori_image, a, summary = sess.run([train, acc, loss, adv,x, logits, merged],
                                                                                        feed_dict=feed_dict)

                if global_cnt % config.show_interval == 0:
                    train_writer.add_summary(summary, global_cnt)
                    print(
                        "e:{}/{}, {}".format(idx, train_set.minibatches, epoch),
                        'loss: {:.3f}'.format(loss_batch),
                        'acc: {:3f}'.format(accuracy),
                        'logits: {:4f}'.format(np.max(a[0])),
                        'index: {}'.format(np.argmax(a[0]))
                    )

            print('Training for batch {} is done'.format(idx))

            cv2.imwrite('../../ori_image/' + '{}.png'.format(idx), ori_image.astype('uint8').reshape(32, 32, 3))
            cv2.imwrite('../../adv_image/' + '{}.png'.format(idx), adv_examples.astype('uint8').reshape(32, 32, 3))

            adv_examples=affine_transformation(adv_examples)
            accuracy_gt = acc_gt.eval(feed_dict={placeholders['data']: adv_examples, placeholders['gt']: labels})
            succ = (idx * succ + 1 - accuracy_gt) / (
                        idx + 1)  # compute success rate of generating adversarial examples that can be misclassified
            noise_l2 = (idx * (noise_l2) + ((adv_examples - images)) ** 2) / (
                        idx + 1)  # compute l2 difference between adversarial examples and origina images

        print('Success rate of this attack is {}'.format(succ))
        print('Noise norm of this attack is {}'.format(np.mean(noise_l2)))
        embed()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os._exit(1)

# vim: ts=4 sw=4 sts=4 expandtab

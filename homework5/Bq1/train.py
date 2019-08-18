#!/usr/bin/env mdl
# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from common import config
import custom_vgg16_bn as vgg16
import cv2
import numpy as np

'''use 13 convolution layers to generate gram matrix'''

GRAM_LAYERS= ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
# GRAM_LAYERS=['conv1_1', 'conv1_2', 'pool1','conv2_1', 'conv2_2','pool2']
image_shape = (1, 224, 224, 3)

'''you need to complete this method'''
def get_l2_gram_loss_for_layer(noise, source, layer):
    def gram(inp,shape):
        inp=tf.reshape(inp,[shape[0],-1,shape[3]])
        inp_t=tf.transpose(inp,perm=[0,2,1])

        g=tf.matmul(inp_t,inp)
        g/= shape[1] *shape[2]
        return g

    noise_layer = getattr(noise, layer)
    source_layer = getattr(source, layer)

    shape=noise_layer.get_shape().as_list()
    g_source=gram(source_layer,shape)
    g_noise=gram(noise_layer,shape)
    g_diff=(g_source-g_noise) **2
    g_diff=tf.reshape(g_diff,[shape[0],-1])

    return tf.reduce_mean(g_diff,axis=1)

    # diff_G=0
    # for n1 in range(n_filter):
    #     G = tf.reduce_sum(tf.expand_dims(noise_layer[:,:,:,n1],-1) * noise_layer,[0,1,2])
    #     G_ = tf.reduce_sum(tf.expand_dims(source_layer[:,:,:,n1],-1) * source_layer,[0,1,2])
    #     diff_G += tf.reduce_sum(tf.square(G-G_))
    # weight_filter = 1.0
    # loss = weight_filter * diff_G / tf.cast(4 * (n_filter ** 2) * (h_filter * w_filter) ** 2, dtype=tf.float32)
    # return loss

def get_gram_loss(noise, source):
    with tf.name_scope('get_gram_loss'):
        gram_loss = [get_l2_gram_loss_for_layer(noise, source, layer) for layer in GRAM_LAYERS ]
    return tf.reduce_mean(tf.convert_to_tensor(gram_loss))

def output_img(session, x, save=False, out_path=None):
    shape = image_shape
    img = np.clip(session.run(x),0, 1) * 255
    img = img.astype('uint8')
    if save:
        cv2.imwrite(out_path, (np.reshape(img, shape[1:])))

def main():
    '''training a image initialized with noise'''
    pre_noise = tf.Variable(tf.random_uniform(image_shape, -3, 3 ))
    noise = tf.Variable(tf.nn.sigmoid(pre_noise))

    '''load texture image, notice that the pixel value has to be normalized to [0,1]'''
    image = cv2.imread('../images/stone_1.jpg')
    image = cv2.resize(image, image_shape[1:3])
    image = image.reshape(image_shape)
    image = (image/255).astype('float32')

    ''' get features of the texture image and the generated image'''
    with tf.name_scope('vgg_src'):
        image_model = vgg16.Vgg16()
        image_model.build(image)

    with tf.name_scope('vgg_noise'):
        noise_model = vgg16.Vgg16()
        noise_model.build(noise)

    ''' compute loss based on gram matrix'''
    with tf.name_scope('loss'):
        loss = get_gram_loss(noise_model, image_model)

    total_loss = loss

    global_steps = tf.Variable(0, trainable = False)
    values = [0.01, 0.005, 0.001]
    lr = tf.train.piecewise_constant(global_steps, [200, 1500], values)

    with tf.name_scope('update_image'):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss, [noise])
        update_image = opt.apply_gradients(grads)

    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'tf_log', 'train'),
                                         tf.get_default_graph())

    ''' create a session '''
    tf.set_random_seed(12345) # ensure consistent results
    global_cnt = 0
    epoch_start = 0
    g_list = tf.global_variables()
    saver = tf.train.Saver(var_list=g_list)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # init all variables

        ## training
        for epoch in range(epoch_start+1, config.nr_epoch+1):
            global_cnt += 1
            _, loss, summary =  sess.run([update_image, total_loss, merged],
                                        feed_dict ={ global_steps: global_cnt} )

            if global_cnt % config.show_interval == 0:
                train_writer.add_summary(summary, global_cnt)
                print(
                    "e:{}".format(epoch),'loss: {:.5f}'.format(loss),
                )

            '''save the trained image every 10 epoch'''
            if global_cnt % config.save_interval == 0 and global_cnt >0 :
                out_dir = os.path.dirname(os.path.realpath(__file__)) + '/output'

                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)
                out_dir = out_dir +'/{}.png'.format(global_cnt)
                output_img(sess, noise, save=True, out_path = out_dir)



        print('Training is done, exit.')




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os._exit(1)

# vim: ts=4 sw=4 sts=4 expandtab

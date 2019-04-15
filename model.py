# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + License: MIT
# [2019-04-15] Modifications for EGAN in tensorflow (https://github.com/zlichen)
#   + License: MIT
from __future__ import division
import os
import time
import math
import itertools
from glob import glob
import tensorflow as tf
from six.moves import xrange
from copy import deepcopy
from ops import *
from utils import *

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]

def dataset_files(root):
    """Returns a list of all image files in the given directory"""
    return list(itertools.chain.from_iterable(
        glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))


class DCGAN(object):
    def __init__(self, sess, image_size=64, is_crop=False,
                 batch_size=32, sample_size=64,lowres=8,
                 z_dim=100, gf_dim=2048, df_dim=128,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 checkpoint_dir=None, lam=0.1):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            lowres: (optional) Low resolution image/mask shrink factor. [8]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        # Currently, image size must be a (power of 2) and (8 or higher).
        assert(image_size & (image_size - 1) == 0 and image_size >= 8)

        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, c_dim]

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.lam = lam

        self.c_dim = c_dim
        #EvolutionGans Part:(Zhili Chen)

        #Training times per Iteration for discriminator
        self.disTrainPIter=3

        #Numbers of Variation of Gans
        self.mutNum=3

        self.loss_type=['heuristic','minimax','ls']

        #batch for discriminator
        self.batch_sizeFD=self.batch_size*self.disTrainPIter

        #hyper-parameters for diversity fitness score
        self.gamma=0.001

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bns = [
            batch_norm(name='d_bn{}'.format(i,)) for i in range(10)]


        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        self.model_name = "DCGAN.model"

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        self.GenImages =tf.placeholder(
            tf.float32, [None] + self.image_shape, name='fake_images')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        # self.z_sum = tf.summary.histogram("z", self.z)

        #Initialize the Discriminator and the Generator--Zhili Chen
        #----------------1st Generator
        self.HeG = self.generator(scopeName="HeG",z=self.z)
        #----------------2ne Generator
        self.MinG=self.generator(scopeName="MinG",z=self.z)
        #----------------3rd Generator
        self.LsG=self.generator(scopeName="LsG",z=self.z)


        self.D, self.D_logits = self.discriminator(self.images)
        #----------------1st reuse Discriminator
        self.D_HeG, self.D_logits_HeG = self.discriminator(self.HeG, reuse=True)
        #----------------2nd reuse Discriminator
        self.D_MinG, self.D_logits_MinG = self.discriminator(self.MinG, reuse=True)
        #----------------3rd reuse Discriminator
        self.D_LsG, self.D_logits_LsG = self.discriminator(self.LsG, reuse=True)

        # #histogram and images
        # self.d_sum = tf.summary.histogram("d", self.D)
        # # self.d__sum = tf.summary.histogram("d_", self.D_)
        # self.HeG_sum = tf.summary.image("HeG", self.HeG)
        # self.MinG_sum = tf.summary.image("MinG", self.MinG)
        # self.LsG_sum = tf.summary.image("LsG", self.LsG)

        #Loss function
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                    labels=tf.ones_like(self.D)))
        self.d_loss_fake_HeG = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_HeG,
                                                    labels=tf.zeros_like(self.D_HeG)))
        self.d_loss_fake_MinG = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_MinG,
                                                    labels=tf.zeros_like(self.D_MinG)))
        self.d_loss_fake_LsG = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_LsG,
                                                    labels=tf.zeros_like(self.D_LsG)))
        #----------------1st son's loss function
        self.d_loss_HeG = self.d_loss_real + self.d_loss_fake_HeG
        #----------------2nd son's loss function
        self.d_loss_MinG=self.d_loss_real+self.d_loss_fake_MinG
        #----------------3rd son's loss function
        self.d_loss_LsG=self.d_loss_real+self.d_loss_fake_LsG

        #Evolution Variation generator loss function--Zhili Chen
        #----------------1st Gen's loss function
        self.g_loss_HeG=tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_HeG,
                                                    labels=tf.ones_like(self.D_HeG)))
        #----------------2nd Gen's loss function
        self.g_loss_MinG=-tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_MinG,
                                                    labels=tf.zeros_like(self.D_MinG)))
        #----------------3rd Gen's loss function
        self.g_loss_LsG=tf.reduce_mean(tf.square(tf.subtract(self.D_LsG,1)))


        #For Visualize
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake_HeG)
        self.g_loss_HeG_sum = tf.summary.scalar("g_loss_HeG", self.g_loss_HeG)
        self.g_loss_MinG_sum= tf.summary.scalar("g_loss_MinG", self.g_loss_MinG)
        self.g_loss_LsG_sum = tf.summary.scalar("g_loss_LsG", self.g_loss_LsG)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss_HeG)



        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.HeG_vars = [var for var in t_vars if 'HeG' in var.name]
        self.MinG_vars= [var for var in t_vars if 'MinG' in var.name]
        self.LsG_vars= [var for var in t_vars if 'LsG' in var.name]

        #Evolutional network Diversity fitness score------------1st son
        self.Fd_HeG=tf.gradients(self.d_loss_HeG,self.d_vars)
        self.FdScore_HeG=tf.multiply(self.gamma,tf.log(sum(tf.reduce_sum(tf.square(x)) for x in self.Fd_HeG)))
        #Evolutional network Quality fitness score
        self.FqScore_HeG=tf.reduce_mean(self.D_HeG)

        #Evolutional network Diversity fitness score------------2nd son
        self.Fd_MinG=tf.gradients(self.d_loss_MinG,self.d_vars)
        self.FdScore_MinG=tf.multiply(self.gamma,tf.log(sum(tf.reduce_sum(tf.square(x)) for x in self.Fd_MinG)))
        #Evolutional network Quality fitness score
        self.FqScore_MinG=tf.reduce_mean(self.D_MinG)

        #Evolutional network Diversity fitness score------------3rd son
        self.Fd_LsG=tf.gradients(self.d_loss_LsG,self.d_vars)
        self.FdScore_LsG=tf.multiply(self.gamma,tf.log(sum(tf.reduce_sum(tf.square(x)) for x in self.Fd_LsG)))
         #Evolutional network Quality fitness score
        self.FqScore_LsG=tf.reduce_mean(self.D_LsG)

        #For Visualize
        self.Fitness_Score_sum=tf.summary.scalar("Fitness_Score",(self.FdScore_HeG-self.FqScore_HeG))
        # self.FqScore_sum=tf.summary.scalar("FqScore",self.FqScore_HeG)



        #plot
        # self.dddLoss=[]
        # self.dddLoss_real=[]
        # self.dddLoss_fake=[]
        # self.FFFdScore=[]
        self.NumLossUse=[0,0,0]

        self.saver = tf.train.Saver(max_to_keep=1)

    def train(self, config):
        data = dataset_files(config.dataset)
        np.random.shuffle(data)
        assert(len(data) > 0)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1,beta2=config.beta2) \
                          .minimize(self.d_loss_HeG, var_list=self.d_vars)


        #Evolution Generator Optimizers--Zhili Chen
        g_optim_HeG=tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1,beta2=config.beta2).minimize(self.g_loss_HeG,var_list=self.HeG_vars)
        g_optim_MinG=tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1,beta2=config.beta2).minimize(self.g_loss_MinG,var_list=self.MinG_vars)
        g_optim_LsG=tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1,beta2=config.beta2).minimize(self.g_loss_LsG,var_list=self.LsG_vars)


        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        merged_g_sum_HeG=tf.summary.merge([self.g_loss_HeG_sum])
        merged_g_sum_MinG=tf.summary.merge([self.g_loss_MinG_sum])
        merged_g_sum_LsG=tf.summary.merge([self.g_loss_LsG_sum])
        merged_d_sum=tf.summary.merge(
            [self.Fitness_Score_sum,self.d_loss_real_sum,self.d_loss_fake_sum,self.d_loss_sum])

        # self.g_sum_he= tf.summary.merge(
        #     [self.z_sum, self.d__sum, self.HeG_sum, self.d_loss_fake_sum_HeG, self.g_loss_heuristic_sum])
        # self.g_sum_mi= tf.summary.merge(
        #     [self.z_sum, self.d__sum, self.MinG_sum, self.d_loss_fake_sum_MinG,self.g_loss_minimax_sum])
        # self.g_sum_ls= tf.summary.merge(
        #     [self.z_sum, self.d__sum, self.LsG_sum, self.d_loss_fake_sum_LsG,self.g_loss_ls_sum])
        # self.d_sum = tf.summary.merge(
        #     [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        #Observe for the changes of the same noise
        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        sample_files = data[0:self.sample_size]

        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("""

======
An existing model was found in the checkpoint directory.
If you just cloned this repository, it's a model for faces
trained on the CelebA dataset for 20 epochs.
If you want to train a new model from scratch,
delete the checkpoint directory or specify a different
--checkpoint_dir argument.
======

""")
        else:
            print("""

======
An existing model was not found in the checkpoint directory.
Initializing a new one.
======

""")
        for epoch in xrange(config.epoch):
            #Caculate the the iteration of every epoch--Zhili Chen
            data = dataset_files(config.dataset)
            batch_idxs = min(len(data), config.train_size) // self.batch_sizeFD

            for idx in xrange(0, batch_idxs):
                #Grap the images in the files--Zhili Chen
                #Numbers of images batch_sizeFD=batch_size*3
                batch_files = data[idx*self.batch_sizeFD:(idx+1)*self.batch_sizeFD]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                         for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                recordForLoss=0


                #Initialize the generator --Zhili Chen
                if epoch==0 and idx==0:
                    #record the used loss function
                    recordForLoss=0
                    #batch for training the generators nums=batch_size
                    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)
                    #samples for training the discriminator nums=batch_size*3
                    sample_forD=np.random.uniform(-1,1,[self.batch_sizeFD,self.z_dim]) \
                            .astype(np.float32)
                    #only one parents. Initialize for the first iteration.
                    _,summary_HeG=self.sess.run([g_optim_HeG,merged_g_sum_HeG],
                            feed_dict={self.z:batch_z,self.is_training:True})
                    #generate nums=batch_size*3 imgs for the discriminator
                    fake_img=self.sess.run(self.HeG,
                            feed_dict={self.z:sample_forD,self.is_training:False})
                    #save for the old imgs and this will go into the discriminator
                    gen_old=fake_img
                    #load the weights
                    for var_idx,_ in enumerate(self.HeG_vars):
                        self.MinG_vars[var_idx].load(self.HeG_vars[var_idx].eval(),self.sess)
                        self.LsG_vars[var_idx].load(self.HeG_vars[var_idx].eval(),self.sess)
                else:
                    for type_i in range(self.mutNum):
                        if self.loss_type[type_i]=='heuristic':
                            batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                                .astype(np.float32)
                            
                            _,summary_HeG=self.sess.run([g_optim_HeG,merged_g_sum_HeG],
                                feed_dict={self.z:batch_z,self.is_training:True})
                            self.writer.add_summary(summary_HeG,counter)
                            #generate fake (number=batch_sizeFD) images for Disc...
                            sample_forD=np.random.uniform(-1,1,[self.batch_sizeFD,self.z_dim]) \
                                    .astype(np.float32)
                            fake_img=self.sess.run(self.HeG,
                                feed_dict={self.z:sample_forD,self.is_training:False})
                            #get the scores, not training....
                            Fd,Fq=self.sess.run([self.FdScore_HeG,self.FqScore_HeG],feed_dict={self.HeG:fake_img,self.images:batch_images,self.is_training:False})
                            fit=Fq-Fd

                        elif self.loss_type[type_i]=='minimax':
                            batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                                .astype(np.float32)
                            _,summary_MinG=self.sess.run([g_optim_MinG,merged_g_sum_MinG],
                                feed_dict={self.z:batch_z,self.is_training:True})
                            self.writer.add_summary(summary_MinG,counter)
                            #generate fake (number=batch_sizeFD) images for Disc...
                            sample_forD=np.random.uniform(-1,1,[self.batch_sizeFD,self.z_dim]) \
                                    .astype(np.float32)
                            fake_img=self.sess.run(self.MinG,
                                feed_dict={self.z:sample_forD,self.is_training:False})
                            #get the scores, not training....
                            Fd,Fq=self.sess.run([self.FdScore_MinG,self.FqScore_MinG],feed_dict={self.MinG:fake_img,self.images:batch_images,self.is_training:False})
                            fit=Fq-Fd

                        elif self.loss_type[type_i]=='ls':
                            batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                                .astype(np.float32)
                            _,summary_LsG=self.sess.run([g_optim_LsG,merged_g_sum_LsG],
                            feed_dict={self.z:batch_z,self.is_training:True})
                            self.writer.add_summary(summary_LsG,counter)
                            #generate fake (number=batch_sizeFD) images for Disc...
                            sample_forD=np.random.uniform(-1,1,[self.batch_sizeFD,self.z_dim]) \
                                    .astype(np.float32)
                            fake_img=self.sess.run(self.LsG,
                                feed_dict={self.z:sample_forD,self.is_training:False})
                            #get the scores, not training....
                            Fd,Fq=self.sess.run([self.FdScore_LsG,self.FqScore_LsG],feed_dict={self.LsG:fake_img,self.images:batch_images,self.is_training:False})
                            fit=Fq-Fd

                        if type_i==0:
                            recordForLoss=0
                            fitCur=fit
                            gen_old=fake_img
                        elif fitCur-fit < 0:
                            recordForLoss=type_i
                            fitCur=fit
                            gen_old=fake_img
                if self.loss_type[recordForLoss]=='heuristic':
                    for var_idx,_ in enumerate(self.HeG_vars):
                        self.MinG_vars[var_idx].load(self.HeG_vars[var_idx].eval(),self.sess)
                        self.LsG_vars[var_idx].load(self.HeG_vars[var_idx].eval(),self.sess)
                        
                elif self.loss_type[recordForLoss]=='minimax':
                    for var_idx,_ in enumerate(self.MinG_vars):
                        self.HeG_vars[var_idx].load(self.MinG_vars[var_idx].eval(),self.sess)
                        self.LsG_vars[var_idx].load(self.MinG_vars[var_idx].eval(),self.sess)
                        
                elif self.loss_type[recordForLoss]=='ls':
                    for var_idx,_ in enumerate(self.LsG_vars):
                        self.HeG_vars[var_idx].load(self.LsG_vars[var_idx].eval(),self.sess)
                        self.MinG_vars[var_idx].load(self.LsG_vars[var_idx].eval(),self.sess)

                self.NumLossUse[recordForLoss]=self.NumLossUse[recordForLoss]+1

                #calculate the idex epoch, (3)  128//32
                #every discriminator train for three times
                D_batch_idxs=min(len(batch_images),self.batch_sizeFD)//config.batch_size
                for iidx in xrange(0,D_batch_idxs):
                    xreal=batch_images[iidx*config.batch_size:(iidx+1)*config.batch_size]
                    xfake=gen_old[iidx*config.batch_size:(iidx+1)*config.batch_size]
                    # Update D network
                    _,summary_str=self.sess.run([d_optim,merged_d_sum],
                        feed_dict={self.images: xreal, self.HeG: xfake, self.is_training: True })
                    self.writer.add_summary(summary_str,counter)

                    Fd=self.FdScore_HeG.eval({self.HeG:xfake,self.images:xreal,self.is_training:False})
                    Fq=self.FqScore_HeG.eval({self.HeG:xfake,self.is_training:False})
                    errD_fake = self.d_loss_fake_HeG.eval({self.HeG: xfake, self.is_training: False})
                    errD_real = self.d_loss_real.eval({self.images: xreal, self.is_training: False})
                    errG = self.g_loss_HeG.eval({self.HeG: xfake, self.is_training: False})

                    counter += 1
                    print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss:{:.8f}, fitnessScore: {:.8f}".format(
                    epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real,errG,Fq-Fd))
                    print("HeG:%5d MinG:%5d LsG:%5d"%(self.NumLossUse[0],self.NumLossUse[1],self.NumLossUse[2]))
                #params are the same in gens
                if np.mod(counter, 100) == 1:
                    samples, d_loss= self.sess.run(
                        [self.HeG, self.d_loss_HeG],
                        feed_dict={self.z: sample_z, self.images: sample_images, self.is_training: False}
                    )
                    save_images(samples, [8, 8],
                                './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: {:.8f}".format(d_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # TODO: Investigate how to parameterise discriminator based off image size.
            h0 = lrelu(self.d_bns[0](conv2d(input_=image, output_dim=128, k_h=4,k_w=4,d_h=2,d_w=2, name='d_h0_conv'),self.is_training),name='d_h0_lrelu')
            h1 = lrelu(self.d_bns[1](conv2d(input_=h0,    output_dim=256, k_h=4,k_w=4,d_h=2,d_w=2, name='d_h1_conv'),self.is_training),name='d_h1_lrelu')
            h2 = lrelu(self.d_bns[2](conv2d(input_=h1,    output_dim=512, k_h=4,k_w=4,d_h=2,d_w=2, name='d_h2_conv'),self.is_training),name='d_h2_lrelu')
            h3 = lrelu(self.d_bns[3](conv2d(input_=h2,    output_dim=1024, k_h=4,k_w=4,d_h=2,d_w=2, name='d_h3_conv'),self.is_training),name='d_h3_lrelu')
            # h4 = lrelu(self.d_bns[4](conv2d(input_=h3,    output_dim=2048, k_h=4,k_w=4,d_h=2,d_w=2, name='d_h4_conv'),self.is_training),name='d_h4_lrelu')
            # h5 = lrelu(self.d_bns[5](conv2d(input_=h4,    output_dim=512, k_h=3,k_w=3,d_h=2,d_w=2, name='d_h5_conv'),self.is_training),name='d_h5_lrelu')
            # h6 = lrelu(self.d_bns[6](conv2d(input_=h5,    output_dim=1024,k_h=3,k_w=3,d_h=1,d_w=1, name='d_h6_conv'),self.is_training),name='d_h6_lrelu')
            # h7 = lrelu(self.d_bns[7](conv2d(input_=h6,    output_dim=1024,k_h=3,k_w=3,d_h=2,d_w=2, name='d_h7_conv'),self.is_training),name='d_h7_lrelu')
            # h8 = lrelu(self.d_bns[8](conv2d(input_=h7,    output_dim=2048,k_h=3,k_w=3,d_h=1,d_w=1, name='d_h8_conv'),self.is_training),name='d_h8_lrelu')
            # h9 = lrelu(self.d_bns[9](conv2d(input_=h8,    output_dim=2048,k_h=3,k_w=3,d_h=2,d_w=2, name='d_h9_conv'),self.is_training),name='d_h9_lrelu')
            h4 = linear(tf.reshape(h3, [-1, 16384]), 1, 'd_h10_lin')
    
            return tf.nn.sigmoid(h4), h4

    def generator(self, scopeName,z):
        with tf.variable_scope(scopeName) as scope:
            self.z_, self.h0_w, self.h0_b = linear(z, 1024*4*4, 'g_h0_lin', with_w=True)
            ssize=tf.shape(z)[0]
			# TODO: Nicer iteration pattern here. #readability
            #hs = [None]
            hs0 = lrelu(tf.reshape(self.z_, [-1, 4, 4, 1024]),name='g_h0_lrelu')
            hs1, _, _ = conv2d_transpose(hs0,[ssize,  8,  8, 1024],k_h=4,k_w=4,d_h=2,d_w=2, name='g_h1_deconv', with_w=True)
            hs1=lrelu(hs1,name='g_h1_lrelu')
            hs2, _, _ = conv2d_transpose(hs1,[ssize, 16, 16, 512],k_h=4,k_w=4,d_h=2,d_w=2, name='g_h2_deconv', with_w=True)
            hs2=lrelu(hs2,name='g_h2_lrelu')
            hs3, _, _ = conv2d_transpose(hs2,[ssize, 32, 32,  512],k_h=4,k_w=4,d_h=2,d_w=2, name='g_h3_deconv', with_w=True)
            hs3=lrelu(hs3,name='g_h3_lrelu')
            hs4, _, _ = conv2d_transpose(hs3,[ssize, 64, 64,  256],k_h=4,k_w=4,d_h=2,d_w=2, name='g_h4_deconv', with_w=True)
            hs4=lrelu(hs4,name='g_h4_lrelu')
            hs5, _, _ = conv2d_transpose(hs4,[ssize, 64, 64,    3],k_h=3,k_w=3,d_h=1,d_w=1, name='g_h5_deconv', with_w=True)
            # hs5=lrelu(hs5,name='g_h5_lrelu')
            # hs6, _, _ = conv2d_transpose(hs5,[ssize, 128, 128,  3],k_h=3,k_w=3,d_h=1,d_w=1, name='g_h6_deconv', with_w=True)
            # hs6=lrelu(hs6,name='g_h6_lrelu')
            # hs7, _, _ = conv2d_transpose(hs6,[ssize, 64, 64,  256],k_h=4,k_w=4,d_h=2,d_w=2, name='g_h7_deconv', with_w=True)
            # hs7=lrelu(hs7,name='g_h7_lrelu')
            # hs8, _, _ = conv2d_transpose(hs7,[ssize, 64, 64,  128],k_h=4,k_w=4,d_h=1,d_w=1, name='g_h8_deconv', with_w=True)
            # hs8=lrelu(hs8,name='g_h8_lrelu')
            # hs9, _, _ = conv2d_transpose(hs8,[ssize,128,128,  128],k_h=4,k_w=4,d_h=2,d_w=2, name='g_h9_deconv', with_w=True)
            # hs9=lrelu(hs9,name='g_h9_lrelu')
            # hs10, _, _=       conv2d_transpose(hs9,[ssize,128,128,    3],k_h=3,k_w=3,d_h=1,d_w=1, name='g_h10_deconv',with_w=True)
            return tf.nn.tanh(hs5)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

    # def _compute_gradients(self,tensor,var_ls):
    #     grads=tf.gradients(tensor,var_ls)
    #     return [grad if grad is not None else tf.zeros_like(var)
    #            for var,grad in zip(var_ls,grads)]
    
    # def generate_pic(self, config):
    #     #will generate 5120=64*80pics
    #     # def make_dir(name):
    #         # Works on python 2.7, where exist_ok arg to makedirs isn't available.
    #         # p = os.path.join(config.outDir, name)
    #     if not os.path.exists('FakeImages'):
    #         os.makedirs('FakeImages')

    #     try:
    #         tf.global_variables_initializer().run()
    #     except:
    #         tf.initialize_all_variables().run()

    #     isLoaded = self.load(self.checkpoint_dir)
    #     assert(isLoaded)

    #     for i in range(80):
    #         sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
    #         samples=self.sess.run(self.HeG,
    #                         feed_dict={self.z: sample_z,self.is_training: False})
    #         for j in range(self.sample_size):
    #             scipy.misc.imsave('./FakeImages/Fake_{:04d}.png'.format(i*80+j),samples[j,:,:,:])
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
from six.moves.urllib.parse import parse_qs
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from datetime import datetime
import cv2
import os
import glob
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("/content/ArtMeUp/runs")
from typing import Tupple, List


class Artist:
    """
    Class in charge of the Style Transfer Learning task, using VGG19 model

    Args:
      img_size (Tupple, defaults to (400,400)): height*size of the images 
      random_seed (int, defaults to 272): seed for reproducibility
      pretrained_model_path(str, defaults to None): model where the pretrained model is stored
    """

    def __init__(
        self, img_size:Tupple=(400, 400), random_seed:int=272, pretrained_model_path:str=None
    ):

        self.img_size = img_size
        if pretrained_model_path is None:
            pretrained_model_path = (
                "model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
            )
        tf.random.set_seed(random_seed)
        self.vgg = tf.keras.applications.VGG19(
            include_top=False,
            input_shape=(img_size[1], img_size[0], 3),
            weights=pretrained_model_path,
        )
        self.vgg.trainable = False
        self.layers = self.vgg.layers

    def load_images(self, content_img_path=None, style_img_path=None):
        """
        Load style and content images

        Args:
          content_img_path(str, defaults to None): path to the content image to be
          used. If not set, the image stored at trial_images/rosedal.jpg is used
          style_img_path(str, defaults to None): path to the style image to be
          used. If not set, the image stored at trial_images/style.jpg is used
        """
        if content_img_path is None:
            content_img_path = "trial_images/rosedal.jpg"
        if style_img_path is None:
            style_img_path = "trial_images/style.jpg"
        self.content_img_path = content_img_path
        content_image = np.array(
            Image.open(content_img_path).resize((self.img_size[0], self.img_size[1]))
        )
        content_image = tf.constant(
            np.reshape(content_image, ((1,) + content_image.shape))
        )
        style_image = np.array(
            Image.open(style_img_path).resize((self.img_size[0], self.img_size[1]))
        )
        style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))
        self.content_image = content_image
        self.style_image = style_image

    def compute_content_cost(self, content_output, generated_output):
        """
        Computes the content cost

        Args:
          a_C(tf.Tensor): tensor of dimension (1, n_H, n_W, n_C), hidden layer 
          activations representing content of the image C
          a_G(tf.Tensor): tensor of dimension (1, n_H, n_W, n_C), hidden layer 
          activations representing content of the image G

        Returns
          J_content(tf.Tensor): content cost
        """
        a_C = content_output[-1]
        a_G = generated_output[-1]
        m, n_H, n_W, n_C = a_G.get_shape().as_list()
        a_C_unrolled = tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
        a_G_unrolled = tf.reshape(a_G, shape=[m, n_H * n_W, n_C])
        # compute the cost (Normalization constant * sum of squared distances)
        J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(
            tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))
        )
        return J_content

    def gram_matrix(self, A):
        """
        Calculate the correlation coeff matrix to get a proxy to the
        image style

        Args
          A(tf.Tensor): matrix of shape (n_C, n_H*n_W)

        Returns
          GA(tf.Tensor): Gram matrix of A, of shape (n_C, n_C)
        """
        GA = tf.matmul(A, tf.transpose(A))
        return GA

    def compute_layer_style_cost(self, a_S, a_G):
        """
        Computes the content cost

        Args:
          a_S(tf.Tensor): tensor of dimension (1, n_H, n_W, n_C), hidden layer 
          activations representing style of the image S
          a_G(tf.Tensor): tensor of dimension (1, n_H, n_W, n_C), hidden layer 
          activations representing style of the image G

        Returns:
          J_style_layer(tf.Tensor): style cost
        """

        m, n_H, n_W, n_C = a_G.get_shape().as_list()
        a_S = tf.reshape(a_S, [n_H * n_W, n_C])
        a_S = tf.transpose(a_S)
        a_G = tf.reshape(a_G, [n_H * n_W, n_C])
        a_G = tf.transpose(a_G)
        GS = self.gram_matrix(a_S)
        GG = self.gram_matrix(a_G)
        # compute the cost (Normalization constant * sum of squared distances)
        J_style_layer = (
            (1 / (2 * n_C * n_H * n_W))\
            * tf.math.reduce_sum(tf.square(tf.subtract(GS, GG)))
        )
        return J_style_layer

    def define_content_style_layers(self, content_layer:List[Tupple], style_layers:List[Tupple]):
        """
        Set the layers from where to catch style features and the weight
        assigned to each layer.
        
        Args:
          content_layer(List[Tupple]): list with a single Tupple containing the name
          of the layer and the weight to be applied. Must always be the final
          layer.
          style_layerscontent_layer(List[Tupple]): list with as many Tupples as 
          existing layes, containing the name of the layers and the weights
          to be applied to each one

        Example:
          define_content_style_layers(
            [("block5_conv4", 1)],
            [
                ("block1_conv1", 0.2),
                ("block2_conv1", 0.2),
                ("block3_conv1", 0.2),
                ("block4_conv1", 0.2),
                ("block5_conv1", 0.2),
            ]
          )
        -----
        Hint: The deeper the layer the more complex the features.
        """
        if style_layers is None:
            self.STYLE_LAYERS = [
                ("block1_conv1", 0.2),
                ("block2_conv1", 0.2),
                ("block3_conv1", 0.2),
                ("block4_conv1", 0.2),
                ("block5_conv1", 0.2),
            ]
        else:
            self.STYLE_LAYERS = style_layers
        self.vgg_model_outputs = self.get_layer_outputs(self.vgg, content_layer)


    def get_layer_outputs(self, vgg:tf.keras.Model, content_layer:str):
        """
        Creates a vgg model that returns a list of intermediate output values

        Args:
          vgg(tf.keras.Model): model to be used
          content_layer('str'): indicates the content layer to be used
        """
        layer_names = self.STYLE_LAYERS
        layer_names.extend(content_layer)
        outputs = [self.vgg.get_layer(layer[0]).output for layer in layer_names]
        model = tf.keras.Model([self.vgg.input], outputs)
        return model

    def compute_style_cost(self, style_image_output:tf.Tensor, generated_image_output:tf.Tensor) -> tf.Tensor:
        """
        Computes the overall style cost from several chosen layers

        Args:
        style_image_output(tf.Tensor): the style image processed as a TF tensor
        generated_image_output(tf.Tensor): the output from the generated image
        processed as a TF tensor

        Returns:
          J_style(tf.Tensor): tensor representing a scalar value with the style
          cost. To calculate it, we sum through all the weighted style cost for 
          all the layers in the network
        """
        J_style = 0
        a_S = style_image_output[:-1]
        a_G = generated_image_output[:-1]
        for i, weight in zip(range(len(a_S)), self.STYLE_LAYERS):
            J_style_layer = self.compute_layer_style_cost(a_S[i], a_G[i])
            J_style += weight[1] * J_style_layer
        return J_style

    @tf.function()
    def total_cost(self, J_content:tf.Tensor, J_style:tf.Tensor, alpha:float, beta:float) -> tf.Tensor:
        """
        Computes the total cost function, summing alpha*J_content + beta*J_style

        Args:
          J_content(tf.Tensor): content cost
          J_style(tf.Tensor): style cost
          alpha(float): hyperparameter weighting the importance of the 
          content cost
          beta(float): hyperparameter weighting the importance of the 
          style cost

        Returns:
          J(tf.Tensor): total epoch cost
        """
        J = alpha * J_content + beta * J_style
        return J

    def clip_0_1(self, image) -> tf.Tensor:
        """
        Truncate all the pixels in the tensor to be between 0 and 1

        Args:
        image(tf.Tensor): the image to be clipped

        Returns:
          tf.Tensor: clipped image
        """
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def tensor_to_image(self, tensor):
        """
        Converts the given tensor into a PIL image

        Args:
        tensor(tf.Tensor): the tensor to be converted

        Returns:
          Image(Image.Image): A PIL image
        """
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)

    def preprocess(self):
        """ 
        Preprocess the images to tf.Tensors
        """
        #Generated image starts being = to content image in the 1st epoch
        generated_image = tf.Variable(
            tf.image.convert_image_dtype(self.content_image, tf.float32)
        )
        self.generated_image = generated_image
        self.a_G = self.vgg_model_outputs(generated_image)

        preprocessed_content = tf.Variable(
            tf.image.convert_image_dtype(self.content_image, tf.float32)
        )
        self.a_C = self.vgg_model_outputs(preprocessed_content)

        preprocessed_style = tf.Variable(
            tf.image.convert_image_dtype(self.style_image, tf.float32)
        )
        self.a_S = self.vgg_model_outputs(preprocessed_style)

    def set_optimizer(self, learning_rate:float):
        """
        Start the Adam optimizer

        Args:
          learning_rate(float): learning rate to be used
        """
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def train_step(self, generated_image:tf.Tensor, alpha:float, beta:float):
        """
        Single training step for an epoch, computes costs and applies gradients

        Args:
          generated_image(tf.Tensor): tensor containing the generated image
          alpha(float): hyperparameter weighting the importance of the 
          content cost
          beta(float): hyperparameter weighting the importance of the 
          style cost
        """
        with tf.GradientTape() as tape:
            self.a_G = self.vgg_model_outputs(generated_image)
            self.J_style = self.compute_style_cost(self.a_S, self.a_G)
            self.J_content = self.compute_content_cost(self.a_C, self.a_G)
            self.J = self.total_cost(self.J_content, self.J_style, alpha=alpha, beta=beta)
        grad = tape.gradient(self.J, generated_image)
        self.optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(self.clip_0_1(generated_image))
        self.generated_image = generated_image

    def initialize(
        self,
        content_img_path:str=None,
        style_img_path:str=None,
        content_layer:List[Tupple]=[("block5_conv4", 1)],
        style_layers:List[Tupple]=None,
    ):
        """
        Initialize the model

        Args:
        content_img_path(str, defaults to None): path to the content image
        style_img_path(str, defaults to None): path to the style image
          content_layer(List[Tupple], defaults to [("block5_conv4", 1)]): list 
          with a single Tupple containing the name of the layer and the weight 
          to be applied. Must always be the final layer.
          style_layers(List[Tupple], defaults to None): list with as many
          Tupples as existing layers, containing the name of the layers and 
          the weights to be applied to each one
        """
        self.load_images(content_img_path, style_img_path)
        self.define_content_style_layers(content_layer, style_layers)

    def run(
        self,
        epochs:int=2000,
        alpha:float=1,
        beta:float=1,
        verbose:bool=False,
        verbose_step:int=250,
        learning_rate:float=0.01,
        save_images:bool=True,
        save_step:int=250,
        save_path:str="output/images/",
        plot:bool=False,
        plot_step:int=250,
        fig_size:Tupple=(7, 7),
        early_stopping_rounds:int=100,
        skip_frames:bool=True
    ):
        """
        Process manager, in charge of running the whole trianing, saving
        images and writing to tensorboard in (ArtMeUp/runs)

        Args:
          epochs(int, defaults to 2000): number of epochs to be runned
          alpha(float, defaults to 1): hyperparameter weighting the importance of the 
          content cost
          beta(float, defaults to 1): hyperparameter weighting the importance of the 
          style cost
          verbose(bool, defaults to False): controls if log prints are made
          verbose_step(int, defaults to 250): controls the log printing assiduity
          learning_rate(float, defaults to 0.01): learning rate for the optimizer
          save_images(bool, defaults to True): controls wheter to save or not the
          generated images
          save_step(int, defaults to 250): controls the images saving assiduity
          save_path(str, defaults to "output/images/"): path where to store the images
          plot(bool, defaults to False): controls wheter to plot the images
          while training
          plot_step(int, defaults to 250): controls the images plotting assiduity
          fig_size(Tupple, defaults to (7, 7)): controls the images plotting
          size
          early_stopping_rounds(int, defaults to 100): if the total cost does
          not decreases for early_stopping_rounds, the process concludes
          skip_frames(bool, defaults to True): set to True if you don't want
          to save very similar images (useful for videos)
        """
        self.preprocess()
        self.set_optimizer(learning_rate)

        if save_images:
            save_path = (
                save_path + f'run_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'
            )
            os.mkdir(save_path)
            self.save_path = save_path

        non_improving_round = 0
        for i in range(epochs):
            self.train_step(self.generated_image, alpha=alpha, beta=beta)
            total_cost = self.J.numpy()
            if i == 0:
              lowest_overal_cost  = total_cost
            if verbose:
                if i % verbose_step == 0:
                    print(f"Epoch {i} ")
            if plot:
                if i % plot_step == 0:
                    plt.figure(figsize=fig_size)
                    if i == 0:
                        image = Image.open(self.content_img_path).resize(
                            (self.img_size[1], self.img_size[0])
                        )
                    else:
                        image = self.tensor_to_image(self.generated_image)
                    imshow(image)
                    plt.show()
            if save_images:
                if i % save_step == 0:
                    if i == 0:
                        image = Image.open(self.content_img_path).resize(
                            (self.img_size[1], self.img_size[0])
                        )
                        image.save(f"{save_path}/{i}.jpg")
                    else:
                        image = self.tensor_to_image(self.generated_image)

                    if skip_frames:
                      decay = (lowest_overal_cost-total_cost)/lowest_overal_cost
                      if decay<=.05:
                        pass   
                      else: 
                        image.save(f"{save_path}/{i}.jpg")
                    else:
                      image.save(f"{save_path}/{i}.jpg")       

            if total_cost< lowest_overal_cost:
              lowest_overal_cost = total_cost
              non_improving_round = 0
            else:
              non_improving_round+=1
              if non_improving_round>=early_stopping_rounds:
                break

            writer.add_scalar(tag='Style loss', scalar_value=total_cost, global_step=i)
            writer.add_scalar(tag='Content loss', scalar_value=self.J_content.numpy(), global_step=i)
            writer.add_scalar(tag='Total loss', scalar_value=self.J.numpy(), global_step=i)


class ImgsToVideo:
    """
    Convert a bunch of images into a video
    
    Args:
      images_path(str): path to the dir where the images are stored. Recall that 
      the images must be in NUMERICAL order for the class to function well.
      frame_size (Tupple, default to (400x400)): manage the output's frames sizes
      output_video_name(str, defaults to 'output_video.avi'): output file name
      fps(int, defaults to 3): output file's fps
    """

    def __init__(self, images_path:str, frame_size:Tupple=(400, 400), fps:int=3):
        self.images_path = images_path
        self.frameSize = frame_size
        self.fps = fps

    def read_images(self):
        """
        Read the images from images_path and store them for processing
        """
        images_path = self.images_path
        images = os.listdir(images_path)
        try:
            imgs_dict = {int(name.split(".")[0]): name for name in images}
        except:
            raise Exception(
                "Images must be ordered in numerical ascending way (1,2,3...,n)"
            )
        sorted_keys = sorted(imgs_dict.keys())
        imgs = [imgs_dict[sorted_key] for sorted_key in sorted_keys]
        self.imgs = imgs

    def images_to_video(self, save_path:str="output/videos/", expand_beggining:int=1):
        """
        Convert images into video using cv2

        Args:
          save_path(str, defaults to output/videos/): path there to save the video
          expand_beggining(bool, defaults to False): controls how many times the
          first frame is multiplied. Useful to give a smoother begining to the
          video.
        """
        save_path = save_path + f'run_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'
        os.mkdir(save_path)
        frameSize = self.frameSize
        fps = self.fps
        images_path = self.images_path
        if images_path[-1] != "/":
            images_path = images_path + "/"
        self.read_images()
        images = self.imgs
        # Multiply the 1st image to create an smooth beginning
        if expand_beggining:
            if "0" in [x.split(".")[0] for x in images]:
                extension = images[0].split(".")[1]
                expansion = [f"0.{extension}"] * expand_beggining
                expansion.extend(images)
                images = expansion
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        out = cv2.VideoWriter(save_path + "/Artwork.mp4", fourcc, fps, frameSize)
        for filename in images:
            img = cv2.imread(images_path + filename)
            # Resize height
            r = frameSize[1] / img.shape[0]
            dim = (int(img.shape[1] * r), frameSize[1])
            img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
            # Resize width
            r = frameSize[0] / img.shape[1]
            dim = (frameSize[0], int(img.shape[0] * r))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
            out.write(img)
        out.release()

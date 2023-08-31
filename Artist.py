import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from datetime import datetime
import cv2
import os
import glob


class Artist:
    """
    Class in charge of the Style Transfer Learning task, using VGG19 model
    -----
    Args:
    > img_size (optional - default=400):
    > random_seed (optional - default=272):
    > pretrained_model_path
    """

    def __init__(
        self, img_size=(400, 400), random_seed=272, pretrained_model_path=None
    ):

        self.img_size = img_size
        if pretrained_model_path is None:
            pretrained_model_path = (
                "model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
            )
        tf.random.set_seed(random_seed)
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            input_shape=(img_size[1], img_size[0], 3),
            weights=pretrained_model_path,
        )
        vgg.trainable = False
        self.vgg = vgg
        self.layers = vgg.layers

    def load_images(self, content_img_path=None, style_img_path=None):
        """
        Load style and content images
        """
        if content_img_path is None:
            content_img_path = "trial_images/rosedal.jpg"
        if style_img_path is None:
            style_img_path = "trial_images/style.jpg"
        self.content_img_path = content_img_path

        img_size = self.img_size
        content_image = np.array(
            Image.open(content_img_path).resize((img_size[0], img_size[1]))
        )
        content_image = tf.constant(
            np.reshape(content_image, ((1,) + content_image.shape))
        )

        style_image = np.array(
            Image.open(style_img_path).resize((img_size[0], img_size[1]))
        )
        style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

        self.content_image = content_image
        self.style_image = style_image

    def compute_content_cost(self, content_output, generated_output):
        """
        Computes the content cost

        Arguments:
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

        Returns:
        J_content -- scalar that you compute using equation 1 above.
        """
        a_C = content_output[-1]
        a_G = generated_output[-1]

        # Retrieve dimensions from a_G
        m, n_H, n_W, n_C = a_G.get_shape().as_list()

        # Reshape a_C and a_G
        a_C_unrolled = tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
        a_G_unrolled = tf.reshape(a_G, shape=[m, n_H * n_W, n_C])

        # compute the cost with tensorflow
        J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(
            tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))
        )

        return J_content

    def gram_matrix(self, A):
        """
        Argument:
        A -- matrix of shape (n_C, n_H*n_W)

        Returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
        """

        GA = tf.matmul(A, tf.transpose(A))
        return GA

    def compute_layer_style_cost(self, a_S, a_G):
        """
        Arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

        Returns:
        J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
        """

        # Retrieve dimensions from a_G
        m, n_H, n_W, n_C = a_G.get_shape().as_list()

        # Reshape the images from (n_H * n_W, n_C) to have them of shape (n_C, n_H * n_W)
        a_S = tf.reshape(a_S, [n_H * n_W, n_C])
        a_S = tf.transpose(a_S)
        a_G = tf.reshape(a_G, [n_H * n_W, n_C])
        a_G = tf.transpose(a_G)

        # Computing gram_matrices for both images S and G
        GS = self.gram_matrix(a_S)
        GG = self.gram_matrix(a_G)

        # Computing the loss
        J_style_layer = (
            1
            / (4 * n_C**2 * (n_H * n_W) ** 2)
            * tf.math.reduce_sum(tf.square(tf.subtract(GS, GG)))
        )

        return J_style_layer

    def define_content_style_layers(self, content_layer, style_layers):
        """
        Set the layers from where to catch style features and the weight assigned to each layer.
        -----
        Args:
        > style_and_weight(optional): list of tupples, last tuple MUST be the content layer.
                                      Example: [('block1_conv1', 0.2),
                                                ('block2_conv1', 0.2),
                                                ('block3_conv1', 0.2),
                                                ('block4_conv1', 0.2),
                                                ('block5_conv1', 0.2),
                                                ('block5_conv4', 1)]
        -----
        Hint: The deeper the layer the more complex the features.
        """
        vgg = self.vgg

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

        self.vgg_model_outputs = self.get_layer_outputs(vgg, content_layer)

    def get_layer_outputs(self, vgg, content_layer):
        """Creates a vgg model that returns a list of intermediate output values."""
        vgg = self.vgg
        layer_names = self.STYLE_LAYERS
        layer_names.extend(content_layer)

        outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
        model = tf.keras.Model([vgg.input], outputs)

        return model

    def compute_style_cost(self, style_image_output, generated_image_output):
        """
        Computes the overall style cost from several chosen layers

        Arguments:
        style_image_output -- our tensorflow model
        generated_image_output --

        Returns:
        J_style -- tensor representing a scalar value, style cost defined above by equation (2)
        """
        STYLE_LAYERS = self.STYLE_LAYERS

        # initialize the overall style cost
        J_style = 0

        # Set a_S to be the hidden layer activation from the layer we have selected.
        a_S = style_image_output[:-1]

        # Set a_G to be the output of the choosen hidden layers.
        a_G = generated_image_output[:-1]
        for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
            # Compute style_cost for the current layer
            J_style_layer = self.compute_layer_style_cost(a_S[i], a_G[i])

            # Add weight * J_style_layer of this layer to overall style cost
            J_style += weight[1] * J_style_layer

        return J_style

    @tf.function()
    def total_cost(self, J_content, J_style, alpha=10, beta=40):
        """
        Computes the total cost function

        Arguments:
        J_content -- content cost coded above
        J_style -- style cost coded above
        alpha -- hyperparameter weighting the importance of the content cost
        beta -- hyperparameter weighting the importance of the style cost

        Returns:
        J -- total cost as defined by the formula above.
        """
        J = alpha * J_content + beta * J_style

        return J

    def clip_0_1(self, image):
        """
        Truncate all the pixels in the tensor to be between 0 and 1

        Arguments:
        image -- Tensor
        J_style -- style cost coded above

        Returns:
        Tensor
        """
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def tensor_to_image(self, tensor):
        """
        Converts the given tensor into a PIL image

        Arguments:
        tensor -- Tensor

        Returns:
        Image: A PIL image
        """
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)

    def preprocess(self):
        """ """
        vgg_model_outputs = self.vgg_model_outputs
        content_image = self.content_image
        style_image = self.style_image
        generated_image = tf.Variable(
            tf.image.convert_image_dtype(content_image, tf.float32)
        )
        self.generated_image = generated_image

        preprocessed_content = tf.Variable(
            tf.image.convert_image_dtype(content_image, tf.float32)
        )
        self.a_C = vgg_model_outputs(preprocessed_content)
        self.a_G = vgg_model_outputs(generated_image)

        preprocessed_style = tf.Variable(
            tf.image.convert_image_dtype(style_image, tf.float32)
        )
        self.a_S = vgg_model_outputs(preprocessed_style)

    def set_optimizer(self, learning_rate):
        """ """
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def train_step(self, generated_image, alpha, beta):
        a_S = self.a_S
        a_G = self.a_G
        a_C = self.a_C

        with tf.GradientTape() as tape:

            a_G = self.vgg_model_outputs(generated_image)

            # Style cost
            J_style = self.compute_style_cost(a_S, a_G)

            # Content cost
            J_content = self.compute_content_cost(a_C, a_G)
            # Total cost
            J = self.total_cost(J_content, J_style, alpha=alpha, beta=beta)

        grad = tape.gradient(J, generated_image)

        self.optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(self.clip_0_1(generated_image))
        self.generated_image = generated_image

    def initialize(
        self,
        content_img_path=None,
        style_img_path=None,
        content_layer=[("block5_conv4", 1)],
        style_layers=None,
    ):
        """
        Initialize the model
        """
        self.load_images(content_img_path, style_img_path)
        self.define_content_style_layers(content_layer, style_layers)

    def run(
        self,
        epochs=1000,
        alpha=10,
        beta=40,
        verbose=False,
        verbose_step=250,
        learning_rate=0.03,
        save_images=True,
        save_step=250,
        save_path=None,
        plot=False,
        plot_step=250,
        fig_size=(7, 7),
    ):
        """ """
        self.preprocess()
        self.set_optimizer(learning_rate)

        if save_images:
            if save_path is None:
                save_path = "output/images/"
            save_path = (
                save_path + f'run_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'
            )
            os.mkdir(save_path)
            self.save_path = save_path

        for i in range(epochs):
            self.train_step(self.generated_image, alpha=alpha, beta=beta)
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
                    else:
                        image = self.tensor_to_image(self.generated_image)
                    image.save(f"{save_path}/{i}.jpg")


class ImgsToVideo:
    """
    Convert a bunch of images into a video
    -----
    Args:
    > images_path: path to the dir where the images are stored.
                   Recall that the images must be in NUMERICAL order
                   for the class to function well.
    > frame_size (optional-default=(400x400)): manage the output's frames sizes
    > output_video_name(optional-default='output_video.avi'): output file name
    > fps(optional-default=3): output file's fps
    """

    def __init__(self, images_path, frame_size=(400, 400), fps=3):

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

    def images_to_video(self, save_path=None, expand_beggining=False):
        """
        Convert images into video using cv2
        """
        if save_path is None:
            save_path = "output/videos/"
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

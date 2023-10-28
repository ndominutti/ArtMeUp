# ArtMeUp

### <b>Overview</b>

ArtMeUp is in first place an implementation of the research conducted by Leon A.Gatys, Alexander S.Ecker and Matthias Bethge in their work [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), inspired by the content seen in deeplearning.ai's Deep Learning Specialization (Module 4). But this module has another functionalities like the hability to generate a transitioning video from the images that the model generates in every epoch.
<br>

So, this package has 2 main objectives:
1) Perform Style Transfer from one image (content image) to another (style image) -using a vgg19 as default base model-,
2) Create a video with the transition of this transformation, like this one:

![Gif of a neural style transfer transition](trial_images/artmeup.gif)

<br>
When runing the code, you can decide to stop after objective one (eg. save your images) or keep working to fulfill objective 2.
<br>

----------

### <b>Repo usage</b>

To use this repo you should:
1) Clone it to your machine
2) Depending in the objective pursued you should instanciate class __Artist__, __ImgsToVideo__ or both (take a glance to this [sample notebook](https://colab.research.google.com/drive/1k-E8njmqnKmWNzwkAvfIbaFzT3XF_VFL?usp=sharing#scrollTo=tDO4kVJP83nB) to see an end to end process)

Have in mind that when performing task 1, the __Artist__ will store the generated images in the ArtMeUp/output/images DIR, under a sub-dir with the time of the run. You can control the amount of saved images and another functionalities through the _run_ method in the __Artist__ class.


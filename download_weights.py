from keras_segmentation.models.all_models import model_from_name
print(model_from_name.keys())

from keras_segmentation.pretrained import download_pretrained_model
download_pretrained_model('pspnet_50')
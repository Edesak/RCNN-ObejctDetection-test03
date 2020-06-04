import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from skimage import data,io
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.visualize
import csv
import argparse as args
from mrcnn import model as modellib
import matplotlib.pyplot as plt
# Path to trained weights file
COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
DATASET_FOLDER ='Dataset'


############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "RS3"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 5 + 1  # Background + baloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.8

    #IMAGE_MAX_DIM = 1920


############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("RS3", 1, "dig_site")
        self.add_class("RS3", 2, "person")
        self.add_class("RS3", 3, "uncovered_dig_site")
        self.add_class("RS3", 4, "deposit")
        self.add_class("RS3", 5, "dig_site_a")


        assert subset in ["test","train"]
        dataset_dir = os.path.join(dataset_dir, subset)

        if(subset=='train'):
            annotations = json.load(open('./Dataset/annots/via_region_data_train.json'))
        else:
            annotations = json.load(open('./Dataset/annots/via_region_data_test.json'))
        annotations = list(annotations.values())  # don't need the dict keys

        annotations = [a for a in annotations if a['regions']]

        for i1 in range(len(annotations)):
            polygons = []
            num_ids = []
            filename =  annotations[i1]['filename']
            i3 = annotations[i1]['regions']
            for i4 in range (len(i3)):
                polygons_row = []
                polygons_row.append(annotations[i1]['regions'][list(annotations[i1]["regions"])[i4]]['shape_attributes'])
                polygons.append(polygons_row)
                n=annotations[i1]['regions'][list(annotations[i1]["regions"])[i4]]['region_attributes']['names']
                try:
                    if n=='dig_site':
                        num_ids.append(1)
                    elif n=='person':
                        num_ids.append(2)
                    elif n=='uncovered_dig_site':
                        num_ids.append(3)
                    elif n =='deposit':
                        num_ids.append(4)
                    elif n =='dig_site_a':
                        num_ids.append(5)
                except:
                    pass
            image_path = os.path.join(dataset_dir, filename)
            image = skimage.io.imread(image_path)
            (height, width) = image.shape[:2]
            self.add_image(
                'object',
                image_id=filename,
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                num_ids=num_ids)
        # also change the return value of def load_mask()
        num_ids = np.array(num_ids, dtype=np.int32)
        #return mask, num_ids

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a bottle dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
       #mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
        #                dtype=np.uint8)
        mask = np.zeros((info["height"], info["width"], len(info["polygons"])),dtype=np.uint8)
        '''
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rrt = p["all_points_y"]
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
            plt.imshow(mask)
            plt.show()
        '''

        for i in range(len(info["polygons"])):
            rr, cc = skimage.draw.polygon(info["polygons"][i][0]['all_points_y'], info["polygons"][i][0]['all_points_x'],mask.shape)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(DATASET_FOLDER, "train")

    dataset_train.prepare()

    dataset_test = BalloonDataset()
    dataset_test.load_balloon(DATASET_FOLDER, "test")

    dataset_test.prepare()
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_test,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads',)




config = CustomConfig()


model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir='models')
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])
train(model)

import os

import shutil

import traceback
from typing import List

from PIL import Image, ImageDraw, ImageFont

from app.base_app import BaseApp
from app.inference_app import InferenceApp
from data.image_info import ImageInfo
from utils import log
import config as cf
from utils.file_handler import open_file
import matplotlib.pyplot as plt


class InferenceVisualizerApp(BaseApp):
    """An object of this class will not only run inference, but also visualize the results.

    This app is not extending, but using an InferenceApp.
    The results will be visualized on duplicates of the input images. Those duplicates are written to disk. Finally,
    the folder containing the duplicates will be opened automatically.
    """

    def __init__(self, inference_app: InferenceApp, images: List[ImageInfo]):
        """Create a new InferenceVisualizerApp.

        :param inference_app: The app which will be used to run the inference. May use single or cascade inferring.
        :param images: The images which should be used for inference and visualizing.
        """
        self._inference_app = inference_app
        self._images = images
        BaseApp.__init__(self)

    def _main(self):

        # run inference
        results = self._inference_app.run_inference_on_images(self._images, merge=cf.get("inference_merge"))

        # delete old visualizations
        shutil.rmtree(cf.get("bbox_visualization_dir"))
        os.mkdir(cf.get("bbox_visualization_dir"))

        # color map to indicate confidence of each bbox. see
        # http://scipy.github.io/old-wiki/pages/Cookbook/Matplotlib/Show_colormaps
        color_map = plt.get_cmap('hot')

        # draw bounding boxes for each image
        for i in range(len(self._images)):
            try:
                img = self._images[i]
                bboxes = results[i]

                # log basic stats
                if len(self._images) < 10:  # don't log this if there are a lot of images
                    log.log("found {} objects in {}".format(
                        len(bboxes),
                        img.basename
                    ))

                # load image
                source_img = Image.open(img.path_original).convert("RGBA")
                draw = ImageDraw.Draw(source_img)

                # draw bboxes
                bbox_max = None  # maximum of bboxes to draw on one image
                bbox_nr = 1
                for bbox in bboxes:

                    # convert confidence into color
                    color = color_map(bbox.confidence)
                    color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

                    # write text containing the confidence score
                    font_size = max(int(bbox.width / 7.5), 5)
                    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", size=font_size)
                    draw.text((bbox.xmin, bbox.ymin), "{}".format(int(100 * bbox.confidence)), color, font=font)

                    draw.rectangle(((bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)), outline=color)
                    bbox_nr += 1
                    if bbox_max is not None and bbox_nr > bbox_max:
                        break

                # save img
                target_file = os.path.join(cf.get("bbox_visualization_dir"), "{}-{}".format(i, img.basename))
                source_img.save(target_file)

            except FileNotFoundError:
                log.log(" .. Skipped {}, because the file could not be found".format(
                    img.path_resized
                ))
            except:
                log.log(" .. Skipped {}, because of an unexpected error:\n{}".format(
                    img.path_resized,
                    traceback.format_exc()
                ))

        # log and open folder containing the visualizations
        log.log(cf.get("bbox_visualization_dir"))
        open_file(cf.get("bbox_visualization_dir"))

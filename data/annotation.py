"""
This module handles additional ground truth data provided for individual images.
"""
import os
import sqlite3
from typing import List

import config as cf


class Annotation:
    """An annotation object bundles meta information about a specific image file.
    
    Currently, this is mainly one BoundingBox. If an image is associated with multiple bounding boxes, it will get
    one annotation object for each.
    """

    def __init__(self, xmin, ymin, xmax, ymax):
        from data.rectangles import BoundingBox
        self._bbox = BoundingBox(xmin, ymin, xmax, ymax)

    @property
    def xmin(self):
        return self._bbox.xmin

    @property
    def ymin(self):
        return self._bbox.ymin

    @property
    def xmax(self):
        return self._bbox.xmax

    @property
    def ymax(self):
        return self._bbox.ymax

    @property
    def bbox(self):
        return self._bbox

    @property
    def bbox_is_valid(self):
        return self._bbox.is_valid


def has_annotations(dataset_key: str) -> bool:
    """Whether the given dataset provides additional annotations."""
    return dataset_key in cf.get("dataset_keys_annotated")


def get_annotations(img_info: 'ImageInfo') -> List[Annotation]:
    """Load all annotation objects that are provided for the given image info.

    :param img_info:
    :return:
    """
    if not has_annotations(img_info.dataset_key):
        raise ValueError("The dataset {} does not provide any annotations.".format(img_info.dataset_key))
    elif img_info.dataset_key == "aflw":
        return _get_annotations_aflw(img_info)
    else:
        raise AttributeError("The dataset {} should provide annotations, but the implementation is missing.".format(
            img_info.dataset_key))


def _get_annotations_aflw(img_info: 'ImageInfo') -> List[Annotation]:
    """Internal method to get all annotations of an image info that belongs to the AFLW dataset.

    :param img_info:
    :return:
    """
    if img_info.dataset_key != "aflw":
        raise ValueError("The image {} is not part of the AFLW dataset.".format(img_info.full_key))

    result = []

    # extract required information from the given params
    _, subfolder_nr = os.path.split(os.path.split(img_info.path_original)[0])
    img_file_name_key = os.path.join(subfolder_nr, img_info.basename)

    # Open the sqlite database
    sql_db_path = os.path.join(cf.get("dataset_path_root"), "aflw/aflw.sqlite")
    conn = sqlite3.connect(sql_db_path)
    c = conn.cursor()

    # Creating the query string for retrieving: roll, pitch, yaw and faces position
    # Change it according to what you want to retrieve
    # TODO this query would be much more efficient when applied for all images at once
    select_string = "facerect.x, facerect.y, facerect.w, facerect.h"
    from_string = "faceimages, faces, facerect"
    where_string = "faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id AND faceimages.filepath = '" + img_file_name_key + "'"
    query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string
    # log.log(query_string)

    # It iterates through the rows returned from the query
    i = 0
    for row in c.execute(query_string):
        xmin = row[0]
        ymin = row[1]
        width = row[2]
        height = row[3]

        # convert attributes
        if xmin is not None and width is not None:
            xmax = xmin + width
        else:
            xmax = None
        if ymin is not None and height is not None:
            ymax = ymin + height
        else:
            ymax = None

        # create and save a new annotation object
        result.append(Annotation(xmin, ymin, xmax, ymax))

        i += 1

    if i == 0:
        raise ValueError("ERROR: Could not find AFLW db entry for " + img_file_name_key)

    return result

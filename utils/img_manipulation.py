import random

import config as cf


class PotentialDeadlockError(Exception):
    """This error type will be raised, if the program seems to have reached a deadlock."""
    pass


def random_img_patch(img, restricted_areas=[], max_iou=0):
    """Get a random image patch from the given image.

    :param img: Must be an image object, not only a numpy array.
    :param restricted_areas: Rectangles describing regions which must not intersect the returned patch by an IoU value
                                of more than max_iou.
    :param max_iou: The maximum allowed IoU between the returned image patch and any restricted area.
                        Must be a float in [0, 1). Set to 0 to disallow any intersection.
    :return:
    """
    from data.rectangles import Rectangle

    old_img_width = img.size[0]
    old_img_height = img.size[1]

    width_min = cf.get("img_width")  # minimum width is the same as used for the network input
    width_max = min(old_img_width, old_img_height)

    max_tries = 100
    tries = 0
    while True:

        # it may be impossible (or at least very unlikely) to find a patch that does not overlap with any rectangle
        # in restricted_areas. so try a few times and stop after some time.
        tries += 1
        if tries > max_tries:
            raise PotentialDeadlockError("Possible deadlock: Could not determine a patch that does not overlap "
                                         "with any rectangle in the restricted areas.")

        width_abs = random.randint(width_min, width_max)

        # use square dimensions
        # TODO try non-squares
        height_abs = width_abs

        # choose random position for the given dimension (without leaving the image borders)
        xmin_max = old_img_width - width_abs  # largest valid value for the xmin coordinate
        ymin_max = old_img_height - height_abs  # largest valid value for the ymin coordinate
        xmin = random.randint(0, xmin_max)
        ymin = random.randint(0, ymin_max)
        xmax = xmin + width_abs
        ymax = ymin + height_abs

        # ensure that the random patch does not overlap the given restricted areas more than allowed.
        # otherwise, try again.
        any_bad_overlap = False
        if len(restricted_areas) > 0:
            request = Rectangle(xmin, ymin, xmax, ymax)
            for restriction in restricted_areas:
                if (max_iou == 0 and request.intersects(restriction)) \
                        or request.intersection_over_union(restriction) > max_iou:
                    any_bad_overlap = True
                    break

        # random image patch intersects restricted area. trying again.
        if any_bad_overlap:
            continue

        # finally crop
        img = img.crop((xmin, ymin, xmax, ymax))

        return img

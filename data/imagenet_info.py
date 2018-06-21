"""This module bundles some additional information about the ImageNet dataset.

See http://image-net.org/download-API
"""
from typing import List

from data.cache import Cache
from utils import log
from utils.file_handler import read_txt_url_lines_to_list


def get_human_wordnet_ids():
    """Get all wordnet ids of the category ’person, individual, someone, somebody, mortal, soul'.
    
    This list might contain ids that are not included in the offline dataset, too.
    Taken from http://image-net.org/synset?wnid=n00007846
    """
    result = _get_sub_ids("n00007846")
    return result


def get_ignored_wordnet_ids():
    result = [
        # human (alike)
        "n03141823",
        "n02669723",  # academic gown, academic robe, judge's robe
        "n04591157",  # Windsor tie
        "n04370456",  # sweatshirt
        "n04228054",  # ski
        "n02879718",  # bow
        "n01440764",  # men holding fish
    ]

    # clothing => usually contains human beings, too
    # Clothing, article of clothing, vesture, wear, wearable, habiliment
    # http://image-net.org/api/text/wordnet.structure.hyponym?wnid=n03051540&full=1
    result += _get_sub_ids("n03051540")

    return result


def _get_sub_ids(id: str, recursive=True) -> List[str]:
    cache_name = "{}{}".format(
        id,
        recursive
    )
    cached_infobox = Cache().load_single("imagenet_ids", cache_name)
    if cached_infobox is not None:
        result = cached_infobox
    else:
        # fetch online content
        url = "http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid={}".format(
            id
        )
        if recursive:
            url += "&full=1"
        try:
            log.log("Trying to fetch information from the ImageNet online API. This may take a while.")
            lines = read_txt_url_lines_to_list(url)
        except:
            raise ConnectionError("Unable to fetch information from the ImageNet online API. "
                                  "The server may be temporary unavailable. You may want to check the "
                                  "URL manually: {}".format(url))

        # parse
        result = []
        for line in lines:
            line = line.replace("-", "").strip()
            if line != "":
                result.append(line)

        Cache().save_single("imagenet_ids", cache_name, result, suffix_extension=".p")

    return result


def get_img_urls_by_id(id: str) -> List[str]:
    """Get all image urls belonging to the synset with the specified id."""
    # fetch and parse online content
    url = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}".format(
        id
    )
    lines = read_txt_url_lines_to_list(url)

    # empty url lists will deliver text lines though, so remove them
    img_urls = []
    for line in lines:
        if "http" in line:
            img_urls.append(line)

    return img_urls

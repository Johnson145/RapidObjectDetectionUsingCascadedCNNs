import os
import shutil
from typing import List

import config as cf
from app.base_app import BaseApp
from app.inference_cascade_app import InferenceCascadeApp
from data.db import label
from data.image_info import ImageInfo
from utils import log, console


class EvaluateFDDBApp(BaseApp):
    """An object of this class will evaluate the default cascade on the FDDB.

    This app is not extending but using an InferenceApp.

    For more information about FDDB see:
    http://vis-www.cs.umass.edu/fddb/
    """

    def __init__(self, session_suffix=""):
        """Create a new EvaluateFDDBApp.

        :param session_suffix: Optionally, you may describe this session by an additional suffix, which will be
                                appended to the result folder's name.
        """
        # ensure that this project is about face detection in the first place
        if not cf.get("foreground_equals_face"):
            raise AttributeError("This app does not make sense, if you're not looking for a face detector")

        # this evaluation isn't about speed at all, so we..
        # .. increase the number of evaluated image scales
        cf.set("window_scale_factor", 1.005)

        # .. may want to disable window merging in favor of less RAM usage
        cf.set("inference_merge", False)

        # we don't want to keep the squared bboxes here, because the vertically-enlarged ones fit better to the
        # ellipses of the FDDB
        cf.set("vertically_enlarge_bboxes", True)

        # save the given session suffix
        self._session_suffix = session_suffix
        if self._session_suffix != "" and not self._session_suffix.startswith("_"):
            self._session_suffix = "_" + self._session_suffix

        # path to the official FDDB evaluation script as well as the additionally provided gnuplot config files
        # you can get this from: http://vis-www.cs.umass.edu/fddb/results.html
        self._path_fddb_evaluation_script = cf.get("fddb_per_evaluation_script_path")
        self._path_fddb_cont_roc_script = os.path.join(cf.get("fddb_gnuplot_compare_dir"), "contROC.p")
        self._path_fddb_disc_roc_script = os.path.join(cf.get("fddb_gnuplot_compare_dir"), "discROC.p")

        # ensure that the external fddb scripts are available
        if not os.path.exists(self._path_fddb_evaluation_script):
            raise FileNotFoundError("Could not find the external FDDB evaluation script.")
        elif not os.path.exists(self._path_fddb_cont_roc_script):
            raise FileNotFoundError("Could not find the external FDDB gnuplot script for the continuous ROC plot.")
        elif not os.path.exists(self._path_fddb_disc_roc_script):
            raise FileNotFoundError("Could not find the external FDDB gnuplot script for the discrete ROC plot.")

        # results will be exported to the following dir/file
        # (group output files by current session)
        self._export_dir = os.path.join(cf.get("fddb_detection_output_dir"), cf.get("session_key") + self._session_suffix)
        if not os.path.exists(self._export_dir):
            os.makedirs(self._export_dir)

        # parent constructor
        BaseApp.__init__(self)

    def _get_img_infos_for_fddb_images(self, fold_nr: int) -> List[ImageInfo]:
        """Create image info objects for all images belonging to the specified fold.

        :param fold_nr: The number of the FDDB fold which should be parsed.
        :return:
        """
        # read fold file to get the relative file paths of images belonging to this fold
        fold_input_file_name = "FDDB-fold-{:02d}.txt".format(fold_nr)
        fold_input_file_path = os.path.join(cf.get("fddb_folds_dir"), fold_input_file_name)
        with open(fold_input_file_path) as f:
            relative_img_paths = f.readlines()
        relative_img_paths = [x.strip() for x in relative_img_paths]

        # use the parsed img paths to create ImageInfo objects
        # (we must keep the original order!)
        img_infos = []
        for relative_img_path in relative_img_paths:
            img_absolute_file_path = os.path.join(cf.get("fddb_img_base_dir"), relative_img_path)
            img_absolute_file_path += ".jpg"
            img_info = ImageInfo(img_absolute_file_path, label.get_by_key(label.KEY_FOREGROUND), "fddb")
            img_infos.append(img_info)

        return img_infos

    def _persist_fold_results(self, fold_nr: int, img_infos: List[ImageInfo], results_per_img):
        """Save all results of the specified fold into one text file.

        :param fold_nr: The number of the FDDB fold which should be parsed.
        :return:
        """
        # results will be exported to the following file
        fold_output_file_name = "fold-{:02d}-out.txt".format(fold_nr)
        export_file_path = os.path.join(self._export_dir, fold_output_file_name)

        # build content
        export_file_content = ""
        for i in range(len(img_infos)):
            img = img_infos[i]
            bboxes = results_per_img[i]

            # begin with basic information
            # ...
            # <image name i>
            # <number of faces in this image =im>
            img_key = img.path_original.replace(cf.get("fddb_img_base_dir"), "").replace(".jpg", "")
            if img_key.startswith("/"):  # remove leading slash
                img_key = img_key[1:]
            n_bboxes = len(bboxes)
            export_file_content += "{}\n".format(img_key)
            export_file_content += "{}\n".format(n_bboxes)

            # add bbox information
            # <face i1>
            # <face i2>
            # ...
            # <face im>
            # ...
            for bbox in bboxes:
                # Each face region is represented as:
                # <left_x top_y width height detection_score>
                export_file_content += "{} {} {} {} {}\n".format(
                    bbox.xmin, bbox.ymin, bbox.width, bbox.height, bbox.confidence
                )

        # write content to file
        with open(export_file_path, "w") as text_file:
            text_file.write(export_file_content)

    def _main(self):

        log.log("Running inference on the FDDB dataset")

        # initialize the cascade
        app_inference = InferenceCascadeApp()

        # process fold by fold
        n_folds = 10  # reducing this will affect the own inference, but not the FDDB evaluation assumptions
        for fold_nr in range(1, n_folds + 1):  # fold_nr starts with 1 not 0
            log.log("*******************  Fold {}/{}  *************************".format(fold_nr, n_folds))

            # read image meta data
            img_infos = self._get_img_infos_for_fddb_images(fold_nr)

            # run inference on all img_infos of the current fold
            results_per_img = app_inference.run_inference_on_images(img_infos, merge=cf.get("inference_merge"))

            # export results
            self._persist_fold_results(fold_nr, img_infos, results_per_img)

        # create a symlink pointing to the new folder containing the latest results
        # (this way we do not need to manipulate the original FDDB evaluation script each time. instead, we can just
        # configure it once to point to the symlink)
        log.log("Creating symlink {}".format(
            cf.get("fddb_latest_detection_output_dir")
        ))
        if os.path.exists(cf.get("fddb_latest_detection_output_dir")) \
                and os.path.islink(cf.get("fddb_latest_detection_output_dir")):  # remove old symlink
            os.remove(cf.get("fddb_latest_detection_output_dir"))
        os.symlink(self._export_dir, cf.get("fddb_latest_detection_output_dir"))

        # start FDDB evaluation script
        log.log("Running the FDDB evaluation script (in Perl)")
        console.run(self._path_fddb_evaluation_script)

        # create additional ROC curves including existing results to compare
        console.run(["gnuplot", self._path_fddb_cont_roc_script])
        console.run(["gnuplot", self._path_fddb_disc_roc_script])
        shutil.copy2(os.path.join(cf.get("fddb_gnuplot_compare_dir"), "contROC-compare.png"), self._export_dir)
        shutil.copy2(os.path.join(cf.get("fddb_gnuplot_compare_dir"), "discROC-compare.png"), self._export_dir)

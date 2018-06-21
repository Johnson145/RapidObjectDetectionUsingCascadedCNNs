import traceback
from imp import reload
from typing import List

from app.base_app import BaseApp
from app.train_app import TrainApp
from data.datasets import SPLIT_KEY_VAL
from network.hyper_tuner_random import HyperTunerRandom
import config as cf
from network.hyper_tuner_successive import HyperTunerSuccessive
from network.net import Net
from utils import log


class TuneSingleApp(BaseApp):
    """This app can be used to automatically tune the hyper parameters in order to get the best single net performance."""

    def __init__(self, param_keys: List[str], random=True):
        """Create a new TuneSingleApp.

        :param param_keys: keys of hyper parameters that should be tuned.
        :param random: Whether tuning should be done randomized (or sequentially).
        """
        # these hyper parameters will be tuned
        self._param_keys = param_keys

        # this app will be used to start a single training session.
        self._trainer = self._create_trainer()

        if random:
            self._tuner = HyperTunerRandom(self._param_keys)
        else:
            self._tuner = HyperTunerSuccessive(self._param_keys)

        self._n_different_train_sessions = self._tuner.required_iterations()
        self._n_total_train_sessions = self._n_different_train_sessions * cf.get("n_repeat_same_session")
        self._more_than_one_run = self._n_total_train_sessions > 1

        # call _main by using the parent constructor
        BaseApp.__init__(self)

    def _main(self):

        # save the best global results (independently from the hyper tuner)
        self._global_best_val_results = None
        self._global_best_session = None

        self._session_i = 0
        while self._session_i < self._n_total_train_sessions:  # don't use a for-loop, as we want to manipulate i inside the loop

            log.log('###############################################################')
            log.log('########################  BEGIN  ##############################')

            # reset configuration (especially the session key etc)
            if self._session_i > 0:
                reload(cf)

            # repetition info
            if cf.get("n_repeat_same_session") > 1:

                rep_i = self._session_i % cf.get("n_repeat_same_session") + 1
                repeat_last_run = 1 < rep_i <= cf.get("n_repeat_same_session")

                log.log("Repetition of current training session: {} of {}".format(
                    rep_i,
                    cf.get("n_repeat_same_session")
                ))
            else:
                repeat_last_run = False

            # use new parameter configuration whenever the current one has been repeated cf.get("n_repeat_same_session") times
            self._tuner.tune(repeat_last_run)

            try:
                # run a single training instance
                # TODO don't use private _main() method
                self._trainer._main()
            except:
                # complete tuning should never fail because of an error in a single session
                log.log("ERROR: cancelling current training, because of an unknown error.")
                log.log(traceback.format_exc())

            try:
                self._finalize_latest_session()
            except:
                # complete tuning should never fail because of an error in a single session
                log.log("ERROR: could not finalize latest session, because of an unknown error.")
                log.log(traceback.format_exc())

            self._session_i += 1

    def _on_cancel(self):
        """This method will be called when the user interrupted the main method."""
        # ask the user whether the latest results should be saved
        finalizeAndSave = cf.get("auto_save_on_abort") or eval(input("Do you want to save the latest data? [y/n]"))
        if finalizeAndSave != "n":
            log.log("Saving latest results.")
            # finalize as usual
            self._trainer._finalize_latest_session()  # training was cancelled, too. so this wasn't called yet
            self._finalize_latest_session()
        else:
            log.log("Results deleted.")

    def _finalize_latest_session(self):
        """Finalize the training session that was started at last."""
        # call super method first to trigger final evaluation
        final_results = self._trainer.final_results

        if final_results is None:
            # this happens e.g. when cancelling a cascade training, because the final results can not be calculated
            # before training all single nets
            log.log("No final results available")
            val_eval = None
        else:
            val_eval = final_results[SPLIT_KEY_VAL]

            # send final values to the hyper tuner
            self._tuner.receive_results(val_eval)

        # delete created graph file
        # (otherwise the tuner will cause way too much memory waste)
        self._trainer.delete_graph_file()

        log.log('########################  END  ################################')
        log.log('###############################################################')

        # if there is more than one run, check global stats
        if self._more_than_one_run:
            # save results
            last_run_is_the_best = False
            if final_results is not None and (self._global_best_val_results is None or \
                    (val_eval[cf.get("tuning_main_criteria")] is not None and
                             val_eval[cf.get("tuning_main_criteria")] > self._global_best_val_results[cf.get("tuning_main_criteria")])):
                self._global_best_val_results = val_eval
                self._global_best_session = cf.get("session_key")
                last_run_is_the_best = True

            # print global results
            if self._global_best_val_results is not None:
                log.log('###############################################################')
                log.log('#################  GLOBAL STATS BEGIN  ########################')

                # evaluation results for the best run in total
                if last_run_is_the_best:
                    log.log(
                        "The best global results have been achieved in the very last run (See evaluation above).")
                else:
                    Net.log_results(self._global_best_val_results,
                                                 "The best global results could not be improved. The highscore is:")
                    log.log("session: {}".format(self._global_best_session))

                # if everything is done: hypertuner results (including the values of the used parameters)
                very_last_run = self._session_i == self._n_total_train_sessions - 1
                if very_last_run and self._param_keys is not None:
                    self._tuner.finalize()
                    self._tuner.log_best_values()  # TODO do not print this only in the very last iteration (at least for the random part)

                log.log('################## GLOBAL STATS END  ##########################')
                log.log('###############################################################')

                # save log
                # (not using the global optimum, but the result of the current run
                log_name = round(val_eval[cf.get("tuning_main_criteria")] * 100, 2) if val_eval is not None else "unknown"
                log.log_set_name('tune-{}p'.format(
                    log_name
                ))
                log.log_save(cf.get("log_dir"))

    def _create_trainer(self) -> TrainApp:
        """Create a new trainer app that will be used for a single training session."""
        return TrainApp(run_now=False)

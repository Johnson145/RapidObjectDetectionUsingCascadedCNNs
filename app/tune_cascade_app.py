from app.tune_single_app import TuneSingleApp
from app.train_cascade_app import TrainCascadeApp


class TuneCascadeApp(TuneSingleApp):
    """Tune a cascade instead of a single net."""

    def _create_trainer(self) -> TrainCascadeApp:
        return TrainCascadeApp(run_now=False)

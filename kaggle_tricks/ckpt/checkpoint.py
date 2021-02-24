import os
from tensorflow import keras
from tensorflow.python.lib.io import file_io


class ModelCheckpointInGcs(keras.callbacks.ModelCheckpoint):
    """Saves h5 model checkpoint in Google Cloud Storage.
    Usage::
        from scml import ModelCheckpointInGcs
        callback = ModelCheckpointInGcs(
            filepath='best_model.h5',
            gcs_dir='gs://mybucket/path/to/dir',
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )
    """

    def __init__(
        self,
        filepath,
        gcs_dir: str,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        options=None,
        **kwargs,
    ):
        super().__init__(
            filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq,
            options=options,
            **kwargs,
        )
        self._gcs_dir = gcs_dir

    def _save_model(self, epoch, logs):
        super()._save_model(epoch, logs)
        filepath = self._get_file_path(epoch, logs)
        if os.path.isfile(filepath):
            with file_io.FileIO(filepath, mode="rb") as inp:
                with file_io.FileIO(
                    os.path.join(self._gcs_dir, filepath), mode="wb+"
                ) as out:
                    out.write(inp.read())
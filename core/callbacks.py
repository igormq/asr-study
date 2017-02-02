import keras.callbacks as callbacks

import h5py
import numpy as np
import yaml


class MetaCheckpoint(callbacks.ModelCheckpoint):
    """
    Checkpoints some training information with the model. This should enable
    resuming training and having training information on every checkpoint.

    Thanks to Roberto Estevao @robertomest - robertomest@poli.ufrj.br
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, training_args=None, meta=None):

        super(MetaCheckpoint, self).__init__(filepath, monitor='val_loss',
                                             verbose=0, save_best_only=False,
                                             save_weights_only=False,
                                             mode='auto', period=1)

        self.filepath = filepath
        self.meta = meta or {'epochs': []}

        if training_args:
            training_args = vars(training_args)

            self.meta['training_args'] = training_args

    def on_train_begin(self, logs={}):
        super(MetaCheckpoint, self).on_train_begin(logs)

    def on_epoch_end(self, epoch, logs={}):
        super(MetaCheckpoint, self).on_epoch_end(epoch, logs)

        # Get statistics
        self.meta['epochs'].append(epoch)
        for k, v in logs.items():
            # Get default gets the value or sets (and gets) the default value
            self.meta.setdefault(k, []).append(v)

        # Save to file
        filepath = self.filepath.format(epoch=epoch, **logs)

        if self.epochs_since_last_save == 0:
            with h5py.File(filepath, 'r+') as f:
                meta_group = f.create_group('meta')
                meta_group.attrs['training_args'] = yaml.dump(
                    self.meta.get('training_args', '{}'))
                meta_group.create_dataset('epochs',
                                          data=np.array(self.meta['epochs']))
                for k in logs:
                    meta_group.create_dataset(k, data=np.array(self.meta[k]))


class ProgbarLogger(callbacks.ProgbarLogger):

    def __init__(self, show_metrics=None):
        super(ProgbarLogger, self).__init__()

        self.show_metrics = show_metrics

    def on_train_begin(self, logs=None):
        super(ProgbarLogger, self).on_train_begin(logs)

        if self.show_metrics:
            self.params['metrics'] = self.show_metrics

from keras.callbacks import Callback
import yaml

class MetaCheckpoint(Callback):
    '''
    Checkpoints some training information on a meta file. Together with the
    Keras model saving, this should enable resuming training and having training
    information on every checkpoint.
    '''

    def __init__(self, filepath, schedule=None, training_args=None):
        self.filepath = filepath
        self.meta = {'epoch': []}
        if schedule:
            self.meta['schedule'] = schedule.get_config()
        if training_args:
            self.meta['training_args'] = training_args

    def on_train_begin(self, logs={}):
        self.epoch_offset = len(self.meta['epoch'])

    def on_epoch_end(self, epoch, logs={}):
        # Get statistics
        self.meta['epoch'].append(epoch + self.epoch_offset)
        for k, v in logs.items():
            # Get default gets the value or sets (and gets) the default value
            self.meta.setdefault(k, []).append(v)

        # Save to file
        filepath = self.filepath.format(epoch=epoch, **logs)

        with open(filepath, 'wb') as f:
            yaml.dump(self.meta, f)

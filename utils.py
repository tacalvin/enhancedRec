import os
import yaml

from pytorch_lightning.callbacks.progress import ProgressBar

class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

    def include(self, node):

        filename = self.construct_scalar(node)

        with open(filename, 'r') as f:
            return yaml.load(f, Loader)


def load_config(path):
    Loader.add_constructor('!include', Loader.include)
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=Loader)
    return cfg

class LitProgressBar(ProgressBar):

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running validation ...')
        return bar
        
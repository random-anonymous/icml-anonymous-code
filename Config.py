import configparser
import traceback
import json


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

    """

    """General"""
    revision = 'basic-cent-tune03'
    datapath = './data/smallset/SST-1/'
    embed_path = './data/smallset/SST-1/embedding.txt'

    optimizer = 'adam'
    attn_mode = 'basic_cent'
    seq_encoder = 'bilstm'

    max_snt_num = 50
    max_wd_num = 50
    max_epochs = 50
    pre_trained = True
    batch_sz = 128
    batch_sz_min = 32
    bucket_sz = 5000
    partial_update_until_epoch = 0

    embed_size = 300
    hidden_size = 200
    dense_hidden = [300, 5]

    lr = 0.0003
    decay_steps = 500
    decay_rate = 0.98

    dropout = 0.6
    early_stopping = 10
    reg = 1e-6
    vbs_config = {'max-iter': 200, 'power-eta': 1e-10, 'grad-iter': 20, 'center-mode': 'center', 'units': 50}

    def __init__(self):
        self.attr_list = [i for i in list(Config.__dict__.keys()) if
                          not callable(getattr(self, i)) and not i.startswith("__")]

    def printall(self):
        for attr in self.attr_list:
            print(attr, getattr(self, attr), type(getattr(self, attr)))

    def saveConfig(self, filePath):

        cfg = configparser.ConfigParser()
        cfg['General'] = {}
        gen_sec = cfg['General']
        for attr in self.attr_list:
            try:
                gen_sec[attr] = json.dumps(getattr(self, attr))
            except Exception as e:
                traceback.print_exc()
                raise ValueError('something wrong in “%s” entry' % attr)

        with open(filePath, 'w') as fd:
            cfg.write(fd)

    def loadConfig(self, filePath):

        cfg = configparser.ConfigParser()
        cfg.read(filePath)
        gen_sec = cfg['General']
        for attr in self.attr_list:
            try:
                val = json.loads(gen_sec[attr])
                assert type(val) == type(getattr(self, attr)), \
                    'type not match, expect %s got %s' % \
                    (type(getattr(self, attr)), type(val))

                setattr(self, attr, val)
            except Exception as e:
                traceback.print_exc()
                raise ValueError('something wrong in “%s” entry' % attr)

        with open(filePath, 'w') as fd:
            cfg.write(fd)
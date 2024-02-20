class config_Boston_pmnet_V1_TL_3:
    def __init__(self,):
        # basics
        self.batch_size = 16
        self.exp_name = 'config_Boston_pmnet_V1_TL_3'
        self.num_epochs = 90
        self.val_freq = 1
        self.num_workers = 0

        self.train_ratio = 0.9
        self.validation_ratio = 0.1
        self.test_ratio = 0.1

        self.pre_trained_model = '/content/drive/MyDrive/Colab Notebooks/Joohan/PMNet_Extension/checkpoints/USC_pmnet_V1_model_0.00016.pt'

        self.dataset_settings()
        self.optim_settings()
        return

    def dataset_settings(self,):
        self.dataset = 'Boston'
        self.cityMap = 'complete'        # complete, height
        #self.antenna = 'complete'           # complete, height, building
        self.sampling = 'exclusive' # random, exclusive
        #self.flipping = True # True, False


    def optim_settings(self,):
        self.lr = 5 * 1e-4
        self.lr_decay = 0.5
        self.step = 20

    def get_train_parameters(self,):
      return {'exp_name':self.exp_name,
        'batch_size':self.batch_size,
        'num_epochs':self.num_epochs,
        'lr':self.lr,
        'lr_decay':self.lr_decay,
        'step':self.step,
        'sampling':self.sampling,
        'pre_trained_model': self.pre_trained_model}
        
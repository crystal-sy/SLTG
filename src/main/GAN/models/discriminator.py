from tensorflow.keras.models import load_model
model_dir = 'D:/sycode/SLTG/src/main/result/'

class Discriminator:
    def __init__(self, version):
        self.version = version

    def train(self, dataset, num_epochs, num_steps, **kwargs):
        return self.d_model.fit(dataset.repeat(num_epochs), verbose=1, epochs=num_epochs, steps_per_epoch=num_steps,
                                **kwargs)

    def save(self, filename):
        self.d_model.save(filename)

    def load(self):
        self.d_model= load_model(model_dir + self.version + '/lstm.h5')

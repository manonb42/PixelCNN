
import time
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from tqdm.notebook import tqdm

def trainer(dataset, model, optimizer, compute_loss, epochs=10, start_epoch=0, batch_size=1, rate=1e-4, status=None):

    for e in range(start_epoch, start_epoch+epochs):
        for input, target in dataset:
            model.train()
            optimizer.zero_grad()

            output = model(input)
            loss = compute_loss(output, target)
            loss.backward()
            optimizer.step()

            if status and status.should_update(e):
                status.update(e, model)

class Status: 
    def __init__(self, /, epoch_interval=-1, time_interval=-1):
        self.last_time = 0
        self.last_epoch = 0
        self.time_interval = time_interval
        self.epoch_interval = epoch_interval


    def should_update(self, epoch):
        return ((time.time() - self.last_time >= self.time_interval) 
            and (epoch - self.last_epoch >= self.epoch_interval))

    def update(self, epoch, _model):
        self.last_update = time.time()
        self.last_epoch = epoch


class LossStatus(Status):

    def __init__(self, score, train, test,  **kwargs):
        super().__init__(**kwargs)
        self.score = score
        self.test = test
        self.train = train
        self.plot = LossPlot()

    def update(self, epoch, model):
        super().update(epoch, model)
        print(time.time())

        self.plot.update(epoch, 
            self.score(model, self.train), 
            self.score(model, self.test))

        clear_output(wait=True)
        display(self.plot.fig) 

        


class LossPlot: 
    def __init__(self):

        fig, ax = plt.subplots(figsize=(8, 6))

        self.epochs = []
        self.train_losses = []
        self.test_losses = []

        self.train_line, = ax.plot([], [], label='Train score')
        self.test_line, = ax.plot([], [], label='Test score')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Score over Epochs')
        ax.legend()

        display(fig)

        self.fig, self.ax = fig, ax

    def update(self, epoch, train_loss, test_loss):
        
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        
        self.train_line.set_data(self.epochs, self.train_losses)
        self.test_line.set_data(self.epochs, self.test_losses)
        
        self.ax.relim()
        self.ax.autoscale_view()
        

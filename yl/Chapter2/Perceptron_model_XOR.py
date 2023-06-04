import torch
from torch import nn, optim
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy

print("torch version:", torch.__version__)
print("pytorch lightning version:", pl.__version__)

xor_input = [Variable(torch.Tensor([0, 0])),
             Variable(torch.Tensor([0, 1])),
             Variable(torch.Tensor([1, 0])),
             Variable(torch.Tensor([1, 1]))]

xor_target = [Variable(torch.Tensor([0])),
              Variable(torch.Tensor([1])),
              Variable(torch.Tensor([1])),
              Variable(torch.Tensor([0]))]

xor_data = list(zip(xor_input, xor_target))
train_loader = DataLoader(xor_data, batch_size=1000)


class XORModel(pl.LightningModule):
    def __init__(self):
        super(XORModel, self).__init__()
        self.input_layer = nn.Linear(2, 4)
        self.output_layer = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

    def forward(self, input):
        # print("INPUT:", input.shape)
        x = self.input_layer(input)
        # print("FIRST:", x.shape)
        x = self.sigmoid(x)
        # print("SECOND:", x.shape)
        output = self.output_layer(x)
        # print("THIRD:", output.shape)
        return output

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        xor_input, xor_target = batch
        # print("XOR INPUT:", xor_input.shape)
        # print("XOR TARGET:", xor_target.shape)
        outputs = self(xor_input)
        # print("XOR OUTPUT:", outputs.shape)
        loss = self.loss(outputs, xor_target)
        return loss


from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

checkpoint_callback = ModelCheckpoint()
model = XORModel()

trainer = pl.Trainer(max_epochs=500, callbacks=[checkpoint_callback])

trainer.fit(model, train_dataloaders=train_loader)

print(checkpoint_callback.best_model_path)
train_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
# test = torch.utils.data.DataLoader(xor_input, batch_size=1)
for val in xor_input:
    _ = train_model(val)
    print([int(val[0]), int(val[1])], int(_.round()))

# 7.accuracy 准确率  稳定性
total_accuracy = []
for xor_input, xor_target in train_loader:
    for i in range(100):
        output_tensor = train_model(xor_input)
        # https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html#multiclassaccuracy
        test_accuracy = accuracy(preds=output_tensor, target=xor_target, task="binary")
        total_accuracy.append(test_accuracy)
total_accuracy = torch.mean(torch.stack(total_accuracy))
print("TOTAL ACCURACY FOR 100 ITERATIONS: ", total_accuracy.item())

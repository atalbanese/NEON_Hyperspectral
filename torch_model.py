import torch.nn
import torch
import pytorch_lightning as pl
import numpy as np
import math
from einops import rearrange

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SimpleTransformer(pl.LightningModule):
    def __init__(
        self,
        lr,
        emb_size,
        scheduler,
        num_features,
        num_heads,
        num_layers,
        num_classes,
        sequence_length,
        weight,
        classes,
        dropout,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.emb_size = emb_size
        self.scheduler = scheduler
        self.classes = classes

        if weight is not None:
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weight))
        else:
            self.loss = torch.nn.CrossEntropyLoss()

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model = num_features,
            nhead = num_heads,
            dim_feedforward=emb_size,
            batch_first=True,
            dropout=dropout
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers = num_layers,
            enable_nested_tensor=False,
        )


        self.decoder = torch.nn.Sequential(torch.nn.Linear(num_features, num_features//2),
                                            torch.nn.BatchNorm1d(sequence_length),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(num_features//2, num_classes),
                                            )


    def calc_ova(self, x, target):
        predicted = torch.argmax(x, dim=1)
        #expected = torch.argmax(target, dim=1)
        expected = target

        conf_matrix = np.zeros((len(self.classes.keys()), len(self.classes.keys())), dtype=np.int32)
        #conf_matrix is: rows-predicted, columns-expected
        zip_count = 0
        for p, e in zip(predicted, expected):
            zip_count +=1
            conf_matrix[p, e] += 1

        ova = np.trace(conf_matrix)/conf_matrix.sum()
        assert conf_matrix.sum() == len(expected), "error calculating confusion matrix"

        return ova, conf_matrix

    def forward(self, batch, softmax = False):
        inp = batch['input']
        pad_mask= batch['pad_mask']
        target = rearrange(batch['target'], 'b p -> (b p)')

        x = self.encoder(inp, src_key_padding_mask = pad_mask)
        x = self.decoder(x)
        x = rearrange(x, 'b p c -> (b p) c')
        if softmax:
            x = torch.nn.functional.softmax(x, 1)
        return x, target

    def training_step(self, batch, batch_idx):
        x, target = self.forward(batch)

        loss = self.loss(x, target)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = self.forward(batch)

        loss = self.loss(x, target)
        self.log("val_loss", loss, prog_bar=True)
        ova, _ = self.calc_ova(x, target)
        self.log("val_ova", ova, prog_bar=True)
        self.log("hp_metric", ova)
        
        return loss

    def test_step(self, batch, batch_idx):
        x, target = self.forward(batch)

        loss = self.loss(x, target)

        self.log("test_loss", loss)
        ova, conf_matrix = self.calc_ova(x, target)
        self.log("test_ova", ova)
        print(self.classes)
        print(conf_matrix)
        tb = self.logger.experiment
        tb.add_text("class_labels", str(self.classes))
        tb.add_text("confusion_matrix", str(conf_matrix).replace("\n", "  \n"))
      
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        to_return = {"optimizer": optimizer, "monitor": "train_loss"}
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=40)
            to_return['lr_scheduler'] = scheduler

        return to_return


class SimpleLinearModel(pl.LightningModule):
    def __init__(
        self,
        lr,
        scheduler,
        num_features,
        num_classes,
        sequence_length,
        weight,
        classes,
        ):
        super().__init__()
        #self.save_hyperparameters()
        self.lr = lr
        self.scheduler = scheduler
        self.classes = classes

        if weight is not None:
            self.loss = torch.nn.CrossEntropyLoss(
                #label_smoothing=0.5, 
                weight=torch.FloatTensor(weight)
                )
        else:
            self.loss = torch.nn.CrossEntropyLoss()


        self.model = torch.nn.Sequential(torch.nn.Linear(num_features, num_features//2),
                                            torch.nn.BatchNorm1d(sequence_length),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(num_features//2, num_features//4),
                                            torch.nn.BatchNorm1d(sequence_length),
                                            torch.nn.ReLU(),
                                            torch.nn.Flatten(),
                                            torch.nn.Linear((num_features//4)*sequence_length, ((num_features//4)*sequence_length)//2),
                                            torch.nn.LayerNorm(((num_features//4)*sequence_length)//2),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(((num_features//4)*sequence_length)//2, ((num_features//4)*sequence_length)//4),
                                            torch.nn.LayerNorm(((num_features//4)*sequence_length)//4),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(((num_features//4)*sequence_length)//4, ((num_features//4)*sequence_length)//8),
                                            torch.nn.LayerNorm(((num_features//4)*sequence_length)//8),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(((num_features//4)*sequence_length)//8, num_classes),
                                            )

        # self.model = torch.nn.Sequential(       torch.nn.MaxPool1d(3),
        #                                         #torch.nn.LayerNorm(num_features),
        #                                         torch.nn.Flatten(),
        #                                         torch.nn.Linear(num_features*sequence_length, (num_features*sequence_length)//2),
        #                                         torch.nn.BatchNorm1d((num_features*sequence_length)//2),
        #                                         torch.nn.ReLU(),
        #                                         torch.nn.Linear((num_features*sequence_length)//2, num_classes)
        # )



    def calc_ova(self, x, target):
        predicted = torch.argmax(x, dim=1)
        expected = torch.argmax(target, dim=1)

        conf_matrix = np.zeros((len(self.classes.keys()), len(self.classes.keys())), dtype=np.int32)
        #conf_matrix is: rows-predicted, columns-expected
        zip_count = 0
        for p, e in zip(predicted, expected):
            zip_count +=1
            conf_matrix[p, e] += 1

        ova = np.trace(conf_matrix)/conf_matrix.sum()
        assert conf_matrix.sum() == len(expected), "error calculating confusion matrix"

        return ova, conf_matrix


    def forward(self, batch, softmax = False):
        x = batch['hs']

        target = batch['target_arr']
        x = self.model(x)
        if softmax:
            x = torch.nn.functional.softmax(x, 1)
        return x, target

    def training_step(self, batch, batch_idx):
        x, target = self.forward(batch)

        loss = self.loss(x, target)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = self.forward(batch)

        loss = self.loss(x, target)
        self.log("val_loss", loss, prog_bar=True)
        ova, _ = self.calc_ova(x, target)
        self.log("val_ova", ova, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, target = self.forward(batch)

        loss = self.loss(x, target)

        self.log("test_loss", loss)
        ova, conf_matrix = self.calc_ova(x, target)
        self.log("test_ova", ova)
        print(self.classes)
        print(conf_matrix)
        tb = self.logger.experiment
        tb.add_text("class_labels", str(self.classes))
        tb.add_text("confusion_matrix", str(conf_matrix).replace("\n", "  \n"))
      
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        to_return = {"optimizer": optimizer, "monitor": "train_loss"}
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=40)
            to_return['lr_scheduler'] = scheduler

        return to_return


class CombinedLinearModel(pl.LightningModule):
    def __init__(
        self,
        lr,
        scheduler,
        num_features,
        num_classes,
        sequence_length,
        weight,
        classes,
        pre_trained: pl.LightningModule

        ):
        super().__init__()
        #self.save_hyperparameters()
        self.lr = lr
        self.scheduler = scheduler
        self.classes = classes
        self.pre_trained = pre_trained
        self.pre_trained.freeze()
        self.pre_trained.eval()

        self.loss = torch.nn.CrossEntropyLoss(
            #label_smoothing=0.5, 
            weight=torch.FloatTensor(weight)
            )


        self.decoder = torch.nn.Sequential(torch.nn.Linear(num_features, num_features//2),
                                            torch.nn.BatchNorm1d(sequence_length),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(num_features//2, num_features//4),
                                            torch.nn.BatchNorm1d(sequence_length),
                                            torch.nn.ReLU(),
                                            torch.nn.Flatten(),
                                            torch.nn.Linear((num_features//4)*sequence_length, ((num_features//4)*sequence_length)//2),
                                            torch.nn.LayerNorm(((num_features//4)*sequence_length)//2),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(((num_features//4)*sequence_length)//2, ((num_features//4)*sequence_length)//4),
                                            torch.nn.LayerNorm(((num_features//4)*sequence_length)//4),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(((num_features//4)*sequence_length)//4, ((num_features//4)*sequence_length)//8),
                                            torch.nn.LayerNorm(((num_features//4)*sequence_length)//8),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(((num_features//4)*sequence_length)//8, num_classes),
                                            )


    def calc_ova(self, x, target):
        predicted = torch.argmax(x, dim=1)
        expected = torch.argmax(target, dim=1)

        conf_matrix = np.zeros((len(self.classes.keys()), len(self.classes.keys())), dtype=np.int32)
        #conf_matrix is: rows-predicted, columns-expected
        zip_count = 0
        for p, e in zip(predicted, expected):
            zip_count +=1
            conf_matrix[p, e] += 1

        ova = np.trace(conf_matrix)/conf_matrix.sum()
        assert conf_matrix.sum() == len(expected), "error calculating confusion matrix"

        return ova, conf_matrix


    def forward(self, batch, softmax = False):
        x = batch['hs']
        hs_pad_mask = batch['hs_pad_mask']
        target = batch['target_arr']
        x = self.pre_trained(batch)
        x = self.decoder(x)
        if softmax:
            x = torch.nn.functional.softmax(x, 1)
        return x, target

    def training_step(self, batch, batch_idx):
        x, target = self.forward(batch)

        loss = self.loss(x, target)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = self.forward(batch)

        loss = self.loss(x, target)
        self.log("val_loss", loss, prog_bar=True)
        ova, _ = self.calc_ova(x, target)
        self.log("val_ova", ova, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, target = self.forward(batch)

        loss = self.loss(x, target)

        self.log("test_loss", loss)
        ova, conf_matrix = self.calc_ova(x, target)
        self.log("test_ova", ova)
        print(self.classes)
        print(conf_matrix)
        tb = self.logger.experiment
        tb.add_text("class_labels", str(self.classes))
        tb.add_text("confusion_matrix", str(conf_matrix).replace("\n", "  \n"))
      
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        to_return = {"optimizer": optimizer, "monitor": "train_loss"}
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=40)
            to_return['lr_scheduler'] = scheduler

        return to_return
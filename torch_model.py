import torch.nn
import torch
import pytorch_lightning as pl
import numpy as np
from einops import rearrange
import os
import csv


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
        weight,
        classes,
        dropout,
        savedir,
        exp_number,
        trial_number
        ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.emb_size = emb_size
        self.scheduler = scheduler
        self.classes = classes
        self.savedir = savedir
        self.exp_number = exp_number
        self.trial_num = trial_number

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


        self.decoder = torch.nn.Sequential(torch.nn.Flatten(start_dim=0, end_dim=1),
                                            torch.nn.Linear(num_features, num_features),
                                            torch.nn.LazyBatchNorm1d(),
                                            torch.nn.ReLU(),
                                            #torch.nn.Dropout(dropout),
                                            torch.nn.Linear(num_features, num_features),
                                            torch.nn.LazyBatchNorm1d(),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(num_features, num_classes),
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
        target = batch['target']

        x = self.encoder(inp, src_key_padding_mask = pad_mask)
        x = self.decoder(x)

        pad_mask = rearrange(~pad_mask, 'b p -> (b p)')
        target = rearrange(target, 'b p -> (b p)')[pad_mask]
        x = x[pad_mask]
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
        self.write_conf_matrix(conf_matrix)
        tb = self.logger.experiment
        tb.add_text("class_labels", str(self.classes))
        tb.add_text("confusion_matrix", str(conf_matrix).replace("\n", "  \n"))
      
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

        to_return = {"optimizer": optimizer, "monitor": "train_loss"}
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=40)
            to_return['lr_scheduler'] = scheduler

        return to_return
    
    def write_conf_matrix(self, conf_matrix):
        rows = []
        for row in conf_matrix:
            rows = rows + [[f'{num}' for num in row]]
        classes = list(self.classes.keys())
        num_classes = len(classes)
        with open(os.path.join(self.savedir,f'{self.exp_number}_{self.trial_num}_conf_matrix.csv'), 'w') as conf_file:
            conf_writer = csv.writer(conf_file)
            header = ['' for x in range(num_classes+1)]
            header[1] = 'Expected'
            conf_writer.writerow(header)
            header_2 = ['Predicted'] + classes
            conf_writer.writerow(header_2)
            for ix, row in enumerate(rows):
                to_write = [classes[ix]] + row
                conf_writer.writerow(to_write)



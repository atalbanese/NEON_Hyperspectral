import torch.nn
import torch
import pytorch_lightning as pl
import numpy as np

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
        classes

        ):
        super().__init__()
        #self.save_hyperparameters()
        self.lr = lr
        self.emb_size = emb_size
        self.scheduler = scheduler
        self.classes = classes

        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=0.05, weight=torch.FloatTensor(weight))

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model = num_features,
            nhead = num_heads,
            dim_feedforward=emb_size,
            batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers = num_layers,
            enable_nested_tensor=False,
        )

        self.decoder = torch.nn.Sequential(torch.nn.Linear(num_features, num_features//2),
                                            torch.nn.BatchNorm1d(sequence_length),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(num_features//2, num_features//4),
                                            torch.nn.BatchNorm1d(sequence_length),
                                            torch.nn.ReLU(),
                                            torch.nn.Flatten(),
                                            torch.nn.Linear((num_features//4)*sequence_length, ((num_features//4)*sequence_length)//2),
                                            torch.nn.BatchNorm1d(((num_features//4)*sequence_length)//2),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(((num_features//4)*sequence_length)//2, ((num_features//4)*sequence_length)//4),
                                            torch.nn.BatchNorm1d(((num_features//4)*sequence_length)//4),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(((num_features//4)*sequence_length)//4, ((num_features//4)*sequence_length)//8),
                                            torch.nn.BatchNorm1d(((num_features//4)*sequence_length)//8),
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

        #print(conf_matrix)

        return ova, conf_matrix


    def forward(self, batch, softmax = False):
        hs = batch['hs']
        hs_pad_mask = batch['hs_pad_mask']
        target = batch['target_arr']

        x = self.encoder(hs, src_key_padding_mask = hs_pad_mask)
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



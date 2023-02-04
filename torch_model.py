import torch.nn
import pytorch_lightning as pl

class SimpleTransformer(pl.LightningModule):
    def __init__(
        self,
        lr,
        emb_size,
        #class_weights, 
        scheduler,
        num_features,
        num_heads,
        num_layers,
        num_classes,
        sequence_length

        ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.emb_size = emb_size
        #self.class_weights = class_weights
        self.scheduler = scheduler

        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

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

        post_pool_size = round((num_features - 4)/5) * sequence_length
        self.pooling = torch.nn.Sequential( torch.nn.MaxPool1d(5),
                                            torch.nn.ReLU())

        self.decoder = torch.nn.Sequential(torch.nn.Linear(post_pool_size, post_pool_size//2),
                                            torch.nn.BatchNorm1d(post_pool_size//2),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(post_pool_size//2, num_classes),
                                            #torch.nn.Softmax(1)
                                            )



    def forward(self, inp):
        pass

    def training_step(self, batch, batch_idx):
        hs = batch['hs']
        hs_pad_mask = batch['hs_pad_mask']
        target = batch['target_arr']

        x = self.encoder(hs, src_key_padding_mask = hs_pad_mask)
        #x = self.encoder(hs)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.decoder(x)

        loss = self.loss(x, target)
        self.log('loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        hs = batch['hs']
        hs_pad_mask = batch['hs_pad_mask']
        target = batch['target_arr']

        x = self.encoder(hs, src_key_padding_mask = hs_pad_mask)
        #x = self.encoder(hs)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.decoder(x)

        loss = self.loss(x, target)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        to_return = {"optimizer": optimizer, "monitor": "train_loss"}
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=40)
            to_return['lr_scheduler'] = scheduler

        return to_return



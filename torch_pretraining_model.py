import torch.nn
import torch
import pytorch_lightning as pl
import transforms
from einops.layers.torch import Rearrange

class PreTrainingModel(pl.LightningModule):
    def __init__(self,
        lr,
        emb_size,
        scheduler,
        num_features,
        num_heads,
        num_layers,
        sequence_length,
        temp=0.1, 
        epsilon=0.05,  
        sinkhorn_iters=3
                ):
        super(PreTrainingModel, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.emb_size = emb_size
        self.scheduler = scheduler
        self.num_features = num_features
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.temp = temp
        self.epsilon = epsilon
        self.niters = sinkhorn_iters

        self.transforms_main = torch.nn.Sequential(
                                        transforms.Blit(p=0.5),
                                        transforms.Block(p=0.5)
                                       )

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

        self.projector = torch.nn.Sequential(torch.nn.Linear(num_features, num_features),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(num_features, num_features),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(num_features, num_features))

        self.prototypes = torch.nn.Linear(num_features, num_features, bias=False)

        self.softmax = torch.nn.LogSoftmax(dim=1)

        self.ra = Rearrange('b s f -> (b s) f')

    @torch.no_grad()
    def norm_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = torch.nn.functional.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    @torch.no_grad()
    def sinkhorn(self, scores):
        Q = torch.exp(scores/self.epsilon).t()
        B = Q.shape[1] 
        K = Q.shape[0]

        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.niters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        return Q.t()
    
    def training_step(self, inp, batch_idx):

        x = inp['input']
        pad_mask = inp['pad_mask']

        b = x.shape[0]
        x_s = self.transforms_main(x)

        inp = torch.cat((x, x_s))
        double_mask = torch.cat((pad_mask, pad_mask))
        inp = self.encoder(inp, src_key_padding_mask = double_mask)
        inp = self.projector(inp)
        inp = torch.nn.functional.normalize(inp, dim=1, p=2)

        scores = self.prototypes(inp)

        scores_t = scores[:b][~pad_mask]
        scores_s = scores[b:][~pad_mask]

        t = scores_t.detach()
        s = scores_s.detach()

        # t = self.ra(t)
        # s = self.ra(s)

        # scores_t = self.ra(scores_t)
        # scores_s = self.ra(scores_s)

        b = scores_t.shape[0]


        q_t = self.sinkhorn(t)
        q_s = self.sinkhorn(s)

        p_t = self.softmax(scores_t/self.temp)
        p_s = self.softmax(scores_s/self.temp)

        loss = -0.5 * torch.mean(q_t * p_s + q_s * p_t)
        loss = loss/b
        self.log("train_loss", loss)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        to_return = {"optimizer": optimizer, "monitor": "train_loss"}
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=40)
            to_return['lr_scheduler'] = scheduler

        return to_return
    
    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if (self.current_epoch < 1):
            for name, p in self.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
    
    def on_train_batch_start(self, batch, batch_idx):
        self.norm_prototypes()

    def forward(self, batch):
        with torch.no_grad():
            inp = batch['hs']
            hs_pad_mask = batch['hs_pad_mask']
            inp = self.encoder(inp, src_key_padding_mask = hs_pad_mask)
            inp = self.projector(inp)
            inp = torch.nn.functional.normalize(inp, dim=1, p=2)

            scores = self.prototypes(inp)
            return scores


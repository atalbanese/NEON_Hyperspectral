import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
from splitting import SiteData
from torch_data import PaddedTreeDataSet, SyntheticPaddedTreeDataSet

class SimpleTransformer(nn.Module):
    def __init__(
        self,
        emb_size,
        num_features,
        num_heads,
        num_layers,
        num_classes,
        sequence_length,
        #weight,
        #classes
        ):
        super(SimpleTransformer, self).__init__()
        self.emb_size = emb_size
        #self.classes = classes

        

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



    def forward(self, hs, hs_pad_mask, softmax = False):
        # hs = batch['hs']
        # hs_pad_mask = batch['hs_pad_mask']
        #target = batch['target_arr']

        x = self.encoder(hs, src_key_padding_mask = hs_pad_mask)
        x = self.decoder(x)
        if softmax:
            x = torch.nn.functional.softmax(x, 1)
        return x


def train_loop(dataloader, model: SimpleTransformer, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, inp in enumerate(dataloader):
        # Compute prediction and loss
        hs, hs_pad_mask = inp['hs'].to(torch.device('cuda:0')), inp['hs_pad_mask'].to(torch.device('cuda:0'))
        pred = model(hs, hs_pad_mask)
        loss = loss_fn(pred, inp['target_arr'].to(torch.device('cuda:0')))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 40 == 0:
            loss, current = loss.item(), batch * len(inp['hs'])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model:SimpleTransformer, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for inp in dataloader:
            hs, hs_pad_mask = inp['hs'].to(torch.device('cuda:0')), inp['hs_pad_mask'].to(torch.device('cuda:0'))
            pred = model(hs, hs_pad_mask)
            target = inp['target_arr'].to(torch.device('cuda:0'))
            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) == target.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    #torch.multiprocessing.set_start_method('spawn')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    niwo = SiteData(
        site_dir = '/home/tony/thesis/data/NIWO',
        random_seed=42,
        train = 0.6,
        test= 0.3,
        valid = 0.1)



    niwo.make_splits('plot_level')
    train_data = niwo.get_data('training', ['hs', 'origin'], 16, make_key=True)
    valid_data = niwo.get_data('validation', ['hs', 'origin'], 16, make_key=True)
    test_data = niwo.get_data('testing', ['hs', 'origin'], 16, make_key=True)

    # with open('niwo_data.pkl', 'wb') as f:
    #     pickle.dump(dict(train_data=train_data, valid_data=valid_data, test_data=test_data),f, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('niwo_data.pkl', 'rb') as f:
    #     all_data = pickle.load(f)
    # train_data = all_data['train_data']
    # valid_data = all_data['valid_data']
    # test_data = all_data['test_data']

    train_set = SyntheticPaddedTreeDataSet(
        tree_list=train_data,
        pad_length=16,
        num_synth_trees=5120,
        num_features=372,
        stats='/home/tony/thesis/data/stats/niwo_stats.npz',
        augments_list=["normalize"]
    )
    train_loader = DataLoader(train_set, batch_size=128, num_workers=4)

    valid_set = PaddedTreeDataSet(valid_data, pad_length=16, stats='/home/tony/thesis/data/stats/niwo_stats.npz', augments_list=["normalize"])
    valid_loader = DataLoader(valid_set, batch_size=38, num_workers=0)

    test_set = PaddedTreeDataSet(test_data, pad_length=16, stats='/home/tony/thesis/data/stats/niwo_stats.npz', augments_list=["normalize"])
    test_loader = DataLoader(test_set, batch_size = len(test_set), num_workers=0)

    train_model = SimpleTransformer(
        emb_size = 378,
        num_features=372,
        num_heads=12,
        num_layers=6,
        num_classes=4,
        sequence_length=16,
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05, weight=torch.FloatTensor([1.05,0.744,2.75,0.753]).to(device))
    optimizer = torch.optim.SGD(train_model.parameters(), lr = 6.246186465873744e-05)

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, train_model, loss_fn,optimizer)

        test_loop(valid_loader, train_model, loss_fn)
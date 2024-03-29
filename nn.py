from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Linear
from torch_geometric.nn import global_mean_pool
import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=2, num_attention_heads=1):
        super(GAT, self).__init__()
        torch.manual_seed(12345)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GATConv((-1, -1), hidden_channels, heads=num_attention_heads, add_self_loops=True)
            self.convs.append(conv)

        self.lin = torch.nn.Linear(hidden_channels * num_attention_heads, out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


def train_model(model, train_loader, loss_fct, optimizer):
    model.train()
    for batch_idx, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        data.to(DEVICE)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = loss_fct(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


@torch.no_grad()
def eval_model(model, test_loader, print_classification_report=False):
    model.eval()
    correct = 0
    true_y = []
    pred_y = []
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        data.to(DEVICE)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        pred_y.append(pred.cpu().detach().numpy())
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        true_y.append(data.y.cpu().detach().numpy())
    if print_classification_report:
        print(classification_report(np.concatenate(true_y), np.concatenate(pred_y), digits=5))
    return (accuracy_score(np.concatenate(true_y), np.concatenate(pred_y)),
            precision_score(np.concatenate(true_y), np.concatenate(pred_y), average='macro'),
            recall_score(np.concatenate(true_y), np.concatenate(pred_y), average='macro'),
            f1_score(np.concatenate(true_y), np.concatenate(pred_y), average='macro'))


def train_eval_model(model, train_loader, test_loader, loss_fct, optimizer, num_epochs=1, verbose=1):
    model.to(DEVICE)
    for epoch in range(1, num_epochs+1):
        train_model(model=model, train_loader=train_loader, loss_fct=loss_fct, optimizer=optimizer)
        train_acc, train_p, train_r, train_f1 = eval_model(model, train_loader)
        if epoch == num_epochs:
            test_acc, test_p, test_r, test_f1 = eval_model(model, test_loader, print_classification_report=True)
            return test_acc, test_p, test_r, test_f1
        else:
            test_acc, test_p, test_r, test_f1 = eval_model(model, test_loader)
            if verbose == 1:
                print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f},'
                      f' Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')

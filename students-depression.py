import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import marimo as mo
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    import torch
    from torch.utils.data import DataLoader
    from torch import nn
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
    from sklearn.metrics import roc_auc_score, f1_score

    return (
        DataLoader,
        OrdinalEncoder,
        StandardScaler,
        mo,
        nn,
        np,
        os,
        pd,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data processing
    """)
    return


@app.cell
def _():
    from torch.utils.data import Dataset

    class DepressionDataset(Dataset):
        def __init__(self, data, labels, transform=None, target_transform=None):
            self.labels = labels
            self.data = data
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return(len(self.labels))

        def __getitem__(self, idx):
            label = self.labels.iloc[idx]
            entry = self.data[idx]
            if self.target_transform:
                label = self.target_transform(label)
            if self.transform:
                entry = self.transform(entry)
            return entry, label

    return (DepressionDataset,)


@app.cell
def _():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("mkhoirulmustaqim/student-depression-detection-dataset")

    print("Path to dataset files:", path)
    return (path,)


@app.cell
def _(os, path, pd):
    file = os.path.join(path, 'Student Depression Detection Dataset', 'Student_Depression_Dataset.csv')
    data = pd.read_csv(file)
    data.shape
    return (data,)


@app.cell
def _(data):
    for col in data.columns:
        col_uniques = data[col].unique()
        print(f'{col}({len(col_uniques)}): {col_uniques}')

    categorical_features = ['Family_background', 'Relationship_status', 'Marital_status', 'Health_condition', 'Home_division']
    binary_features = ['Gender', 'Living_style', 'friend_circle', ]
    numerical_features = ['Age', 'Current_semester', 'CGPA', 'Cost_without_study']
    other_features = []
    return binary_features, categorical_features, numerical_features


@app.cell
def _(data):
    # \u26a0\ufe0f Runnig this cell twice will lead to all Nans in column
    data['Depression'] = data['Depression'].map({'No': 0, 'Yes': 1})
    print(data["Depression"].unique())
    return


@app.cell
def _(data):
    train_data = data.sample(frac=0.8)
    test_data = data.drop(train_data.index)
    y_train = train_data['Depression']
    y_train = y_train[:-1]
    y_test = test_data['Depression']
    train_data = train_data.drop('Depression', axis=1)
    test_data = test_data.drop('Depression', axis=1)
    train_data.shape, y_train.shape, test_data.shape, y_test.shape
    return test_data, train_data, y_test, y_train


@app.cell
def _(
    OrdinalEncoder,
    StandardScaler,
    binary_features,
    categorical_features,
    np,
    numerical_features,
    test_data,
    train_data,
):
    ord_encoder = OrdinalEncoder()
    X_train_cat = ord_encoder.fit_transform(train_data[(categorical_features + binary_features)])
    X_test_cat = ord_encoder.transform(test_data[(categorical_features + binary_features)])

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_data[numerical_features])
    X_test_num = scaler.transform(test_data[numerical_features])

    X_train = np.hstack([X_train_cat, X_train_num])
    X_test = np.hstack([X_test_cat, X_test_num])

    one_example = X_train[-1]
    X_train = X_train[:-1]

    X_train.shape, X_test.shape
    return X_test, X_train, one_example, ord_encoder, scaler


@app.cell
def _(np):
    def transform(x):
        return x.astype(np.float32)

    return (transform,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model implementation
    """)
    return


@app.cell
def _(torch):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    return (device,)


@app.cell
def _(nn):
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.stack = nn.Sequential(
                nn.Linear(12, 12),
                nn.ReLU(),
                nn.Linear(12, 6),
                nn.ReLU(),
                nn.Linear(6, 1),
            )
        def forward(self, x):
            logits = self.stack(x)
            return logits.squeeze(-1)

    return (NeuralNetwork,)


@app.cell
def _(NeuralNetwork, device):
    model = NeuralNetwork().to(device)
    model
    return (model,)


@app.cell
def _(device, model, nn, torch):
    # Model test
    X = torch.rand((64, 12), device=device)
    logits = model(X)
    prob = nn.Softmax(dim=0)(logits)
    y_pred = prob.argmax(0)
    logits, prob, y_pred
    return


@app.cell
def _(device, model, nn, torch, y_train):
    learning_rate = 1e-3
    batch_size = 64
    epochs = 100
    patience = 10
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    pos_weight = torch.tensor(
        [(y_train == 0).sum() / (y_train == 1).sum()],
        dtype=torch.float32,
    ).to(device)
    print(f"pos_weight = {pos_weight.item():.2f}")   # ожидается ~2.77
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return batch_size, epochs, loss_fn, optimizer, patience


@app.cell
def _(DepressionDataset, X_test, X_train, transform, y_test, y_train):
    dataset_train = DepressionDataset(X_train, y_train, transform=transform, target_transform=transform)
    dataset_test = DepressionDataset(X_test, y_test, transform=transform, target_transform=transform)
    return dataset_test, dataset_train


@app.cell
def _(DataLoader, batch_size, dataset_test, dataset_train):
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    return loader_test, loader_train


@app.cell
def _(batch_size, torch):
    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()
        num_batches = len(dataloader)
        train_loss = 0
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if batch % 100 == 0:
            loss_, current = loss.item(), (batch * batch_size + len(X))
            train_loss += loss_

        train_loss /= num_batches
        print(f"loss: {train_loss:>7f}")
        return train_loss


    def test_loop(dataloader, model, loss_fn):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                pred = model(X)
                test_loss += loss_fn(pred, y).item() * len(y)

                probs = torch.sigmoid(pred)
                preds = (probs > 0.5).to(torch.float32)
                correct += (preds == y).sum().item()

                current = (batch * batch_size + len(X))

        test_loss /= num_batches * batch_size
        test_accuracy = 100 * correct/size
        print(f"Test Error: \nAccuracy: {test_accuracy:>0.1f}%, Avg loss: {test_loss:>8f}")
        return test_loss, test_accuracy

    return test_loop, train_loop


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training initialization
    """)
    return


@app.cell
def _(
    epochs,
    loader_test,
    loader_train,
    loss_fn,
    model,
    optimizer,
    patience,
    test_loop,
    train_loop,
):
    train_history, test_history = [], []
    counter, best_accuracy = 0, 0
    logs = [str(model)]

    for e in range(epochs):
        print(f'\nEpoch: {e+1}')
        train_history.append(train_loop(loader_train, model, loss_fn, optimizer))
        test_loss, test_accuracy = test_loop(loader_test, model, loss_fn)
        test_history.append((test_loss, test_accuracy))

        delta = test_accuracy - best_accuracy
        print(f'Accuracy delta = {delta}')
        if delta < 0.1:
            counter += 1
   
            if counter == patience:
                print('Early stopping interupted')
                logs.append(f'Early stopping interupted after epoch {e+1}')
                break
            continue
        best_accuracy = test_accuracy
        counter = 0
    return test_history, train_history


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Results
    """)
    return


@app.cell
def _(np, plt, test_history, train_history):
    test_history_np = np.array(test_history)

    fig, (ax_train, ax_test) = plt.subplots(2,1, figsize=(10, 16))

    l1, = ax_train.plot(train_history, color='tab:red', label='Loss')

    ax_train.set_xlabel('Epoch')
    ax_train.set_ylabel('Loss', color='tab:red')
    ax_train.set_title('Train history')

    ax_test_acc = ax_test.twinx()

    l1, = ax_test.plot(test_history_np[:, 0], color='tab:red', label='Loss')
    l2, = ax_test_acc.plot(test_history_np[:, 1], color='tab:blue', label='Accuracy')

    ax_test.set_xlabel('Epoch')
    ax_test.set_ylabel('Loss', color='tab:red')
    ax_test_acc.set_ylabel('Accuracy', color='tab:blue')
    ax_test.set_title('Test history')

    plt.tight_layout()
    plt.savefig('weight_decay.png')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prediction example
    """)
    return


@app.cell
def _(
    binary_features,
    categorical_features,
    model,
    np,
    numerical_features,
    one_example,
    ord_encoder,
    scaler,
    torch,
):
    one_prediction = torch.sigmoid(model(torch.tensor(one_example).to(torch.float32)))

    parametrs = np.hstack([ord_encoder.inverse_transform(one_example[:len(categorical_features + binary_features)].reshape(1, -1))[0], scaler.inverse_transform(one_example[len(categorical_features + binary_features):].reshape(1, -1))[0]])

    columns = categorical_features + binary_features + numerical_features

    for p in zip(columns, parametrs):
        print(f'{p[0]}: {p[1]}')

    print(f'\nPrognosis: will {'not ' * (one_prediction.item()<0.5)}have depression')
    return


if __name__ == "__main__":
    app.run()

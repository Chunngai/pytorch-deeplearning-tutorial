import argparse
import copy
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from tqdm import tqdm

from data.ag_news_dataset import AGNewsDataset, collate_func
from models.rnn_text_classifier import RNNTextClassifier
from utils.data_utils import make_class_index_mapping, make_vocab
from utils.training_utils import device


def train(model, criterion, optimizer, train_loader):
    """Train the model."""

    model.train()

    losses = []
    for batch in tqdm(train_loader):
        texts = batch["texts"].to(device)
        labels = batch["labels"].to(device)

        predictions = model(texts)

        loss = criterion(predictions, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = torch.tensor(losses).mean()
    print(f"Train Loss : {train_loss:.3f}")


def valid(model, criterion, dev_loader):
    """Validate the model performance."""

    model.eval()

    all_labels = []
    all_predictions = []
    losses = []
    with torch.no_grad():
        for batch in dev_loader:
            texts = batch["texts"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(texts)

            loss = criterion(predictions, labels)
            losses.append(loss.item())

            all_labels.append(labels)
            all_predictions.append(predictions.argmax(dim=-1))

    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)

    valid_loss = torch.tensor(losses).mean()
    valid_acc = accuracy_score(
        y_true=all_labels.detach().cpu().numpy(),
        y_pred=all_predictions.detach().cpu().numpy()
    )
    print(f"Valid Loss : {valid_loss:.3f}")
    print(f"Valid Acc  : {valid_acc:.3f}")

    return valid_loss.item()


def test(model, test_loader, class_index_mapping):
    """Test the model"""

    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            texts = batch["texts"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(texts)

            all_labels.append(labels)
            all_predictions.append(predictions)

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

    all_labels = all_labels.detach().cpu().numpy()
    all_predictions = F.softmax(all_predictions, dim=-1).argmax(dim=-1).detach().cpu().numpy()

    test_acc = accuracy_score(
        y_true=all_labels,
        y_pred=all_predictions
    )
    print(f"Test Acc   : {test_acc:.3f}")

    print("\nClassification Report : ")
    print(classification_report(all_labels, all_predictions, target_names=class_index_mapping.keys()))

    print("\nConfusion Matrix : ")
    print(confusion_matrix(all_labels, all_predictions))


def predict(model, text, tokenizer, vocab, class_index_mapping):
    """Predict the label of the given text."""

    tokens = tokenizer(text)
    token_ids = vocab(tokens)
    texts = torch.tensor([token_ids])
    with torch.no_grad():
        predictions = model(texts)

    prediction_index = predictions[0].argmax(dim=0).item()
    prediction_class = {
        index: class_
        for class_, index in class_index_mapping.items()
    }[prediction_index]

    print(f"text: {text}")
    print(f"prediction: {prediction_class}")


def main(args: argparse.Namespace):
    train_dataset = AGNewsDataset(fp="train.csv")
    dev_dataset = AGNewsDataset(fp="dev.csv")
    test_dataset = AGNewsDataset(fp="test.csv")

    tokenizer = get_tokenizer('basic_english')
    # tokenizer = get_tokenizer(word_tokenize)
    vocab = make_vocab(
        texts=train_dataset.texts,
        tokenizer=tokenizer,
        min_freq=args.min_freq,
    )
    class_index_mapping = make_class_index_mapping(labels=train_dataset.labels)

    model = RNNTextClassifier(
        vocab_len=len(vocab),
        class_num=len(class_index_mapping),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    )
    model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        params=model.parameters(),
        lr=args.lr,
    )

    collate_fn = lambda batch: collate_func(
        batch=batch,
        tokenizer=tokenizer,
        vocab=vocab,
        max_len=args.max_len,
        class_index_mapping=class_index_mapping,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_bz,
        collate_fn=collate_fn,
        shuffle=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_bz,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_bz,
        collate_fn=collate_fn
    )

    best_valid_loss = float("inf")
    best_model = None
    for i in range(args.num_epochs):
        print(f"Epoch {i + 1}")

        train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader
        )

        valid_loss = valid(
            model=model,
            criterion=criterion,
            dev_loader=dev_loader
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model)

    test(
        model=best_model,
        test_loader=test_loader,
        class_index_mapping=class_index_mapping
    )

    predict(
        model=best_model,
        text="Apple Tops in Customer Satisfaction \"Dell comes in a close second, while Gateway shows improvement, study says.\"",
        tokenizer=tokenizer,
        vocab=vocab,
        class_index_mapping=class_index_mapping,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data args.
    parser.add_argument("--min-freq", type=int, default=10, help="Min frequency of the word added to the vocab.")
    parser.add_argument("--max-len", type=int, default=25, help="Max sentence len.")
    # Model args.
    parser.add_argument("--embed-dim", type=int, default=50, help="Embedding dimension.")
    parser.add_argument("--hidden-dim", type=int, default=50, help="Hidden dimension of the RNN.")
    # Training args.
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--train-bz", type=int, default=64, help="Training batch size.")
    parser.add_argument("--eval-bz", type=int, default=64, help="Validation/testing batch size.")

    args = parser.parse_args()

    main(args)

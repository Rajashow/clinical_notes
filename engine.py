import torch
from tqdm import tqdm


def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    train_fn train a model

    Given a dataloader, model, optimizer, device and scheduler train the model.

    Parameters
    ----------
    data_loader : torch.Dataloader
        torch data loader
    model : torch.model
        torch model
    optimizer : torch.optimizer
        torch optimizer
    device : str
        device
    scheduler : torch.scheduler
        torch scheduler

    Returns
    -------
    float
        final loss
    """
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    """
    eval_fn evaluate a model

    given a model, dataloader and device get loss

    Parameters
    ----------
    data_loader : torch.dataloader
        dataloader
    model : torch.model
        model
    device : str
        torch device

    Returns
    -------
    float
        final loss
    """
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        _, loss = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)

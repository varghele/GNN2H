import torch
from misc_modules.data_loader import data_loader


def train(model, criterion, optimizer, batch_size, device, database, latent_graph_dict, latent_graph_list):
    # Set model to training mode
    #model = model.train()
    # --- Backward hook with gradient clipping
    clip_value = 0.1
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    # Set up loss
    loss_avg = 0

    # Get batches from data loader
    batch_p1, batch_p2, batch_pm, batch_target, batch_Temperature = data_loader(database, latent_graph_dict,
                                                                                latent_graph_list, batch_size,
                                                                                shuffle=True
                                                                                )
    for btc in range(len(batch_p1)):
        # Get molecules 1 and two, as well as the membrane in batches, together with targets and temperature
        p1 = batch_p1[btc].to(device)
        p2 = batch_p2[btc].to(device)
        pm = batch_pm[btc].to(device)
        targ = batch_target[btc].to(device)
        Temperature = batch_Temperature[btc].to(device)

        # Forward pass and gradient descent
        out = model([p1, p2, pm, Temperature])
        loss = criterion(out, targ)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        loss_avg += loss.item()

        # Empty CUDA cache
        torch.cuda.empty_cache()

    return loss_avg/len(batch_p1)


def evaluate(model, criterion, batch_size, device, database, latent_graph_dict, latent_graph_list):
    # Set model to evaluation
    #model = model.eval()
    # Set up loss
    loss_avg = 0

    with torch.no_grad():
        # Get batches from data loader
        batch_p1, batch_p2, batch_pm, batch_target, batch_Temperature = data_loader(database, latent_graph_dict,
                                                                                    latent_graph_list, batch_size,
                                                                                    shuffle=False
                                                                                    )
        for btc in range(len(batch_p1)):
            # Get molecules 1 and two, as well as the membrane in batches, together with targets and temperature
            p1 = batch_p1[btc].to(device)
            p2 = batch_p2[btc].to(device)
            pm = batch_pm[btc].to(device)
            targ = batch_target[btc].to(device)
            Temperature = batch_Temperature[btc].to(device)

            # Calculate loss
            out = model([p1, p2, pm, Temperature])
            loss = criterion(out, targ)

            loss_avg += loss.item()

            # Empty CUDA cache
            torch.cuda.empty_cache()

    return loss_avg / len(batch_p1)

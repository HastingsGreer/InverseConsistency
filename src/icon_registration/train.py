import torch


def train2d(net, optimizer, d1, d2, epochs=400):
    batch_size = net.identityMap.shape[0]
    loss_history = []
    print("[", end="")
    for epoch in range(epochs):
        print("-", end="", flush=True)
        if (epoch + 1) % 50 == 0:
            print("]", end="\n[")
        for A, B in list(zip(d1, d2)):
            if A[0].size()[0] == batch_size:
                image_A = A[0].cuda()
                image_B = B[0].cuda()
                optimizer.zero_grad()
                (
                    loss,
                    inverse_consistency_loss,
                    similarity_loss,
                    transform_magnitude,
                    flips,
                ) = net(image_A, image_B)

                loss.backward()
                optimizer.step()
        du = (
            (
                net.phi_AB_vectorfield[:, :, 1:, :-1]
                - net.phi_AB_vectorfield[:, :, :-1, :-1]
            )
            .detach()
            .cpu()
        )
        dv = (
            (
                net.phi_AB_vectorfield[:, :, :-1, 1:]
                - net.phi_AB_vectorfield[:, :, :-1, :-1]
            )
            .detach()
            .cpu()
        )
        dA = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]

        loss_history.append(
            [
                inverse_consistency_loss.item(),
                similarity_loss.item(),
                transform_magnitude.item(),
                torch.log(torch.sum(dA < 0) + 0.1),
            ]
        )
    print("]")
    return loss_history

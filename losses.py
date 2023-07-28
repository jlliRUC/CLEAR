import torch
import torch.nn.functional as F
import numpy as np


def get_loss(name):
    if name == "NCE":
        return InfoNCE_default
    elif name == "pos-in":
        return InfoNCE_pos_in
    elif name == "pos-out-single":
        return InfoNCE_pos_out_single
    elif name == 'pos-rank-out-all':
        return InfoNCE_pos_rank_out_all
    elif name == 'pos-rank-sim':
        return InfoNCE_pos_rank_sim


def InfoNCE_default(batch_size, n_views, temperature, features, device):
    # Here args.n_views = 2
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    # For each subject, it has (n_views-1) positive and (batch_size-1)*n_views negatives
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features.type(torch.float32), dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # Discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)

    # labels: (batch_size, batch_size-1)
    labels = labels[~mask].view(labels.shape[0], -1)
    # similarity_matrix: (batch_size, batch_size-1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # Select positives
    # positives: (batch_size, n_views-1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select negatives
    # negatives: (batch_size, batch_size-n_views)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # Now for each subject, its positive is the first column
    logits = torch.cat([positives, negatives], dim=1).to(device)
    # So the ground truth label should all be zero
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature

    criteration = torch.nn.CrossEntropyLoss().to(device)
    loss = criteration(logits, labels)

    return loss


def InfoNCE_pos_in(batch_size, n_views, temperature, features, device):
    """
    Each time we only consider one pos pair in numerator, and one pos pair + all neg pairs in denominator
    :param batch_size:
    :param n_views:
    :param temperature:
    :param features:
    :param device:
    :return:
    """
    features = F.normalize(features.type(torch.float32), dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # labels: (batch_size * n_views, batch_size * n_views)
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    loss_total = 0.0

    # Each time, for each sample, its postive is different, but negatives are constant. That means we only need to select
    # one positive pair each time and mask the others.
    for n in range(0, n_views-1):
        # Suppose we first mask all positive pairs, then we only need to recover the selected pos-pairs according to n.
        mask = torch.eye(batch_size, dtype=bool).repeat(n_views, n_views)
        # Below is not a 2-layer loop, just for ease of traversal.
        for i in range(0, batch_size):
            for j in range(0, n_views):
                row = i + j * batch_size
                # the i-th pos for each anchor is its i-th following samples augmented from the same ancestor
                # Example: We augment A for three times and get A1, A2 and A3, which means n_views = 3. For A1, its 1st
                #          pos is A2, 2nd is A3, for A2, its 1st pos is A3, 2nd is A1, for A3, its 1st pos is A1,
                #          2nd pos is A2.
                column = i + ((j+n+1) % n_views) * batch_size
                mask[row][column] = False
        mask = ~mask
        # labels_temp: (batch_size*n_views, 1 + (batch_size-1) * n_views)
        labels_temp = labels[mask].bool().view(batch_size*n_views, -1)
        # similarity_matrix_temp: (batch_size*n_views, 1 + (batch_size-1) * n_views)
        similarity_matrix_temp = similarity_matrix[mask].view(batch_size*n_views, -1)

        pos = similarity_matrix_temp[labels_temp.bool()].view(similarity_matrix_temp.size()[0], -1) / temperature
        neg = similarity_matrix_temp[~labels_temp.bool()].view(similarity_matrix_temp.shape[0], -1) / temperature

        logits = torch.cat([pos, neg], dim=1).to(device)
        logits = F.softmax(logits, dim=1)

        pos = logits[:, 0]
        loss_total += pos

    # calculate the average inside the log
    loss = -torch.log(loss_total / (n_views - 1)).mean()

    return loss


def InfoNCE_pos_out_single(batch_size, n_views, temperature, features, device):
    """
    Each time we only consider one pos pair in numerator, and one pos pair + all neg pairs in denominator
    :param batch_size:
    :param n_views:
    :param temperature:
    :param features:
    :param device:
    :return:
    """
    features = F.normalize(features.type(torch.float32), dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # labels: (batch_size * n_views, batch_size * n_views)
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    loss_total = 0.0

    # Each time, for each sample, its postive is different, but negatives are constant. That means we only need to select
    # one positive pair each time and mask the others.
    for n in range(0, n_views-1):
        # Suppose we first mask all positive pairs, then we only need to recover the selected pos-pairs according to n.
        mask = torch.eye(batch_size, dtype=bool).repeat(n_views, n_views)
        # Below is not a 2-layer loop, just for ease of traversal.
        for i in range(0, batch_size):
            for j in range(0, n_views):
                row = i + j * batch_size
                # the i-th pos for each anchor is its i-th following samples augmented from the same ancestor
                # Example: We augment A for three times and get A1, A2 and A3, which means n_views = 3. For A1, its 1st
                #          pos is A2, 2nd is A3, for A2, its 1st pos is A3, 2nd is A1, for A3, its 1st pos is A1,
                #          2nd pos is A2.
                column = i + ((j+n+1) % n_views) * batch_size
                mask[row][column] = False
        mask = ~mask
        # labels_temp: (batch_size*n_views, 1 + (batch_size-1) * n_views)
        labels_temp = labels[mask].bool().view(batch_size*n_views, -1)
        # similarity_matrix_temp: (batch_size*n_views, 1 + (batch_size-1) * n_views)
        similarity_matrix_temp = similarity_matrix[mask].view(batch_size*n_views, -1)

        pos = similarity_matrix_temp[labels_temp.bool()].view(similarity_matrix_temp.size()[0], -1) / temperature
        neg = similarity_matrix_temp[~labels_temp.bool()].view(similarity_matrix_temp.shape[0], -1) / temperature

        logits = torch.cat([pos, neg], dim=1).to(device)
        logits = F.softmax(logits, dim=1)

        pos = logits[:, 0]
        loss_total += (-torch.log(pos).mean())

    # calculate the average outside the log
    loss = loss_total / (n_views - 1)

    return loss


def InfoNCE_pos_rank_out_all(batch_size, n_views, temperature, features, device, ranks=None):
    """
    1. The features are stored by an ascending order of data augmentation, i.e.,
    the features from (i+1)-th view is from stronger data augmentation compared with i-th view.
    2. When we deal with i-th views, the former views will be discarded since they are more similar to anchor

    :param batch_size:
    :param n_views:
    :param temperature:
    :param features:
    :param device:
    :return:
    """
    # ranks: list (n_views)
    features = F.normalize(features.type(torch.float32), dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    if ranks is None:
        a = [i for i in range(n_views)]
        ranks = []
        for i in range(len(a)):
            ranks_sub = sorted(range(len(a)), key=lambda k: abs(a[i] - a[k]), reverse=False)
            ranks_row = [item for item in ranks_sub for idx in range(batch_size)]
            for j in range(batch_size):
                ranks.append(ranks_row)
    ranks = torch.tensor(ranks)
    num_rank = len(ranks[0, :].unique())

    loss_total = 0.0

    for rank in range(1, num_rank):
        # For default CL, there are only two types of samples for each anchor, i.e., pos and neg
        # We need a Bool Tensor 'label' to select them.
        # But for ranked CL, there are three types of pos, i.e. easy pos, rank (current) pos, and hard pos
        # Here we need to drop out the easy pos, keep the rank pos and the hard pos.
        # Also, different from pos-out and pos-in who consider only one pos in the denominator,
        # here we always keep all positives in the denominator.
        # By iteratively ranking positives, these hard positives get more chances to be trained by the model.
        # Note that for unsupervised CL, there is no class label, each anchor still belongs to one unique class.
        # Thus we don't need 'label', only positives and negatives

        # Discard easy pos
        # all pos_positions
        mask_pos = torch.eye(batch_size).bool().repeat(n_views, n_views)
        # mask for the easy pos (also itself)
        mask_pos_easy = torch.where(ranks < rank, True, False)
        mask_pos_easy = mask_pos * mask_pos_easy

        # Locate rank pos (rank pos + hard pos, the others are neg (neg)
        mask_pos_rank = torch.where(ranks >= rank, True, False)
        mask_pos_rank = mask_pos * mask_pos_rank
        mask_pos_rank = mask_pos_rank[~mask_pos_easy].view(similarity_matrix.size()[0], -1)

        similarity_rank = similarity_matrix[~mask_pos_easy].view(similarity_matrix.size()[0], -1)

        positives = similarity_rank[mask_pos_rank].view(similarity_rank.shape[0], -1) / temperature
        negatives = similarity_rank[~mask_pos_rank].view(similarity_rank.shape[0], -1) / temperature

        logits = torch.cat([positives, negatives], dim=1).to(device)
        logits = F.softmax(logits, dim=1)

        # for each rank, there maybe multi positives.
        pos = logits[:, 0:positives.shape[1]]

        loss_total += -torch.log(pos).mean()

    loss = loss_total / num_rank

    return loss


def InfoNCE_pos_rank_sim(batch_size, n_views, temperature, features, device, ranks=None):
    """
    rank each positive based on similarity.
    :param batch_size:
    :param n_views:
    :param temperature:
    :param features:
    :param device:
    :return:
    """
    # ranks: list (n_views)
    features = F.normalize(features.type(torch.float32), dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    if ranks is None:
        ranks = []
        for i in range(batch_size * n_views):
            sim_pos = [similarity_matrix[i, (i + batch_size * j) % (batch_size * n_views)] for j in range(n_views)]
            _, ranks_temp = torch.sort(torch.tensor(sim_pos), descending=True)
            ranks_row = [item for item in ranks_temp for idx in range(batch_size)]
            ranks.append(ranks_row)
    ranks = torch.tensor(ranks)
    num_rank = len(ranks[0, :].unique())

    loss_total = 0.0

    for rank in range(1, num_rank):
        # For default CL, there are only two types of samples for each anchor, i.e., pos and neg
        # We need a Bool Tensor 'label' to select them.
        # But for ranked CL, there are three types of pos, i.e. easy pos, rank pos, and hard pos
        # Here we need to drop out the easy pos, keep the rank pos and the hard pos.
        # Also, different from pos-out and pos-in who consider only one pos in the denominator,
        # here we always keep all positives in the denominator.
        # By iteratively ranking positives, these hard positives get more chances to be trained by the model.
        # Note that for unsupervised CL, there is no class label, each anchor still belongs to one unique class.
        # Thus we don't need 'label', only positives and negatives

        # Discard easy pos
        # all pos_positions
        mask_pos = torch.eye(batch_size).bool().repeat(n_views, n_views)
        # mask for the easy pos (also itself)
        mask_pos_easy = torch.where(ranks < rank, True, False)
        mask_pos_easy = mask_pos * mask_pos_easy

        # Locate rank pos (rank pos + hard pos, the others are neg (neg)
        mask_pos_rank = torch.where(ranks >= rank, True, False)
        mask_pos_rank = mask_pos * mask_pos_rank
        mask_pos_rank = mask_pos_rank[~mask_pos_easy].view(similarity_matrix.size()[0], -1)

        similarity_rank = similarity_matrix[~mask_pos_easy].view(similarity_matrix.size()[0], -1)

        positives = similarity_rank[mask_pos_rank].view(similarity_rank.shape[0], -1) / temperature
        negatives = similarity_rank[~mask_pos_rank].view(similarity_rank.shape[0], -1) / temperature

        logits = torch.cat([positives, negatives], dim=1).to(device)
        logits = F.softmax(logits, dim=1)

        # for each rank, there maybe multi positives.
        pos = logits[:, 0:positives.shape[1]]

        loss_total += -torch.log(pos).mean()

    loss = loss_total / num_rank

    return loss


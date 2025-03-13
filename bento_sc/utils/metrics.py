def pearson_batch_masked(preds, trues):

    mask = trues == -1

    preds = preds.clone()
    trues = trues.clone()

    preds[mask] = 0
    trues[mask] = 0

    preds_mean = preds.sum(1) / (~mask).sum(1)
    trues_mean = trues.sum(1) / (~mask).sum(1)

    pred_min_mean = preds - preds_mean[:, None]
    true_min_mean = trues - trues_mean[:, None]

    pred_min_mean[mask] = 0
    true_min_mean[mask] = 0

    numer = (pred_min_mean * true_min_mean).sum(1)
    denom = ((pred_min_mean**2).sum(1) * (true_min_mean**2).sum(1)) ** 0.5

    return numer / denom

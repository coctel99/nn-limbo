def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    num_samples = prediction.shape[0]

    for i in range(num_samples):
        if ground_truth[i]:
            if prediction[i]:
                tp += 1
            else:
                fn += 1
        else:
            if prediction[i]:
                fp += 1
            else:
                tn += 1

    if prediction.shape[0] != 0:
        accuracy = (tp + tn) / (tp + fp + fn + tn)
    else:
        accuracy = 0
    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    s = ground_truth.shape[0]
    tp = 0
    for i in range(s):
        if prediction[i] == ground_truth[i]:
            tp += 1

    if s != 0:
        accuracy = tp / s
    else:
        accuracy = 0

    return accuracy

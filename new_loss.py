class AutoscaleFocalLoss:
    def __init__(self, threshold, alpha):
        self.threshold = threshold
        self.alpha = alpha
    
    def gamma(self, y_pred):
        return self.threshold/2 * (torch.cos(np.pi*(y_pred+1)) + 1)

    def __call__(self, y_pred, y_true):
        y_true = F.one_hot(y_true, 3)
        CE = F.binary_cross_entropy_with_logits(y_pred, y_true,reduction = 'none')
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_true * y_pred + (1-y_true) * (1-y_pred)
        loss = ((1 - y_pred)**self.gamma(y_pred)) * CE
        loss = loss.mean()
        return loss
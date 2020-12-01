






class Optimizer(object):
    pass


class SGD(Optimizer):
    def __init__(self,
                 lr,
                 aaa=0,
                 bbb=0):
        super(SGD, self).__init__()
        self.lr = lr

    def update(self, param, dparam, param_lr=1.0, decay_type=None, decay_coeff=0.0):
        assert decay_type in ['L1Decay', 'L2Decay', None]
        lr = self.lr   # 优化器的学习率
        # lr = param_lr * lr
        if decay_type is None:
            param = param - param_lr * lr * dparam
        elif decay_type == 'L2Decay':
            # p = (1.0 - 1.0*param_lr * lr *decay_coeff)
            # param = p * param - param_lr * lr * dparam
            p = (1.0 - 1.0*param_lr * lr *decay_coeff)
            param = p * param - param_lr * lr * dparam
        return param
















class Optimizer(object):
    pass


class SGD(Optimizer):
    def __init__(self,
                 lr,
                 aaa=0,
                 bbb=0):
        super(SGD, self).__init__()
        self.lr = lr

    def update(self, param, dparam, layer_lr=1.0):
        lr = self.lr   # 优化器的学习率
        param = param - layer_lr * lr * dparam
        return param









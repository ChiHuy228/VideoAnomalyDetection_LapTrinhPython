class Config(object):
    def __init__(self, args):
        self.lr = eval(args.lr)
        self.lr_str = args.lr

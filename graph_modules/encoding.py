import torch
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, BatchNorm1d as BN


class EncodingMLP(torch.nn.Module):
    def __init__(self, NO_INP, NO_OUT, NO_LAYERS, NORM=True):
        super(EncodingMLP, self).__init__()

        def LIN_block(ins, outs, norm=True, last=False):
            if norm == True and last == False:
                return Seq(Lin(ins, outs), LeakyReLU(), BN(outs))  # BATCHNORM1D
            elif norm == False and last == False:
                return Seq(Lin(ins, outs), LeakyReLU())
            if last == True:
                return Seq(Lin(ins, outs))

        mlp = []
        if NO_LAYERS == 0:
            mlp.append(LIN_block(NO_INP, NO_OUT, NORM, True))
        else:
            mlp.append(LIN_block(NO_INP, NO_OUT, NORM))
            for _ in range(NO_LAYERS)[:-1]:
                mlp.append(LIN_block(NO_OUT, NO_OUT, NORM))
            mlp.append(LIN_block(NO_OUT, NO_OUT, False, True))

        self.mlp = Seq(*mlp)

    def forward(self, x):
        return self.mlp(x)


class LastMLP(torch.nn.Module):
    def __init__(self, NO_INP, NO_OUT, NO_LAYERS, NORM=True):
        super(LastMLP, self).__init__()

        def LIN_block(ins, outs, norm=True, last=False):
            if norm == True and last == False:
                return Seq(Lin(ins, outs), LeakyReLU(), BN(outs))  # BATCHNORM1D
            elif norm == False and last == False:
                return Seq(Lin(ins, outs), LeakyReLU())
            if last == True:
                return Seq(Lin(ins, outs))

        mlp = []
        if NO_LAYERS != 0:
            mlp.append(LIN_block(NO_INP, NO_INP, NORM))
            for _ in range(NO_LAYERS)[:-1]:
                mlp.append(LIN_block(NO_INP, NO_INP, NORM))
            mlp.append(LIN_block(NO_INP, NO_OUT, False, True))
        else:
            mlp.append(LIN_block(NO_INP, NO_OUT, False, True))

        self.mlp = Seq(*mlp)

    def forward(self, x):
        return self.mlp(x)

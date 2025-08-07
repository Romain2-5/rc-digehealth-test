from scipy import signal


class Filter:
    def __init__(self, cutoff, fs, btype, axis):
        self.cutoff = cutoff
        self.fs = fs
        self.btype = btype
        self.axis = axis

    def apply(self, data):
        raise NotImplementedError()


class ButterFilter(Filter):

    def __init__(self, cutoff, fs, btype, axis, order=2, rs=20):
        super().__init__(cutoff, fs, btype, axis)

        self.sos = signal.cheby2(N=order, rs=rs, Wn=self.cutoff, fs=fs, btype=btype, output='sos')


    def apply(self, data):
        sig = signal.sosfiltfilt(sos=self.sos, x=data, axis=self.axis)

        return sig

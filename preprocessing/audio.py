''' MFCC, FBank, LogFBank calculations were removed from python_speech_features package
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common.sigproc import delta
from common import sigproc

import numpy as np
import logging

from scipy import signal
from scipy.fftpack import dct


class Feature(object):
    """ Base class for features calculation
    This class is an interface and must not be instantiated. All children class must implement __str__ and __call__ function.
    """

    def __init__(self):
        raise NotImplementedError, "This class must not be instantiated"

    def __call__(self):
        raise NotImplementedError, "__call__ must be overrided"

class FBank(Feature):
    """Compute Mel-filterbank energy features from an audio signal.
    """

    def __init__(self, fs=16e3, win_len=0.025, win_step=0.01,
                 num_filt=40, nfft=512, low_freq=20, high_freq=7800,
                 pre_emph=0.97, win_fun=signal.hamming, d=False, dd=False, eps=1e-14):
        """Constructor
        """

        if high_freq > fs/2:
            raise ValueError, "high_freq must be less or equal than fs/2"

        self.fs = fs
        self.win_len = win_len
        self.win_step = win_step
        self.num_filt = num_filt
        self.nfft = nfft
        self.low_freq = low_freq
        self.high_freq = high_freq or self.fs/2
        self.pre_emph = pre_emph
        self.win_fun = win_fun
        self.eps = eps

        self._filterbanks = self._get_filterbanks()

        self._logger = logging.getLogger('%s.%s' % (__name__, self.__class__.__name__))


    @property
    def mel_points(self):
        return np.linspace(self._low_mel, self._high_mel, self.num_filt + 2)

    @property
    def low_freq(self):
        return self._low_freq

    @low_freq.setter
    def low_freq(self, value):
        self._low_mel = self._hz2mel(value)
        self._low_freq = value

    @property
    def high_freq(self):
        return self._high_freq

    @high_freq.setter
    def high_freq(self, value):
        self._high_mel = self._hz2mel(value)
        self._high_freq = value


    def __call__(self, signal):
        """Compute Mel-filterbank energy features from an audio signal.
        :param signal: the audio signal from which to compute features. Should be an N*1 array

        Returns:
            2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
            second return value is the energy in each frame (total energy, unwindowed)
        """

        signal = sigproc.preemphasis(signal, self.pre_emph)

        frames = sigproc.framesig(signal,
                                  self.win_len*self.fs,
                                  self.win_step*self.fs,
                                  self.win_fun)

        pspec = sigproc.powspec(frames, self.nfft)
        # this stores the total energy in each frame
        energy = np.sum(pspec, 1)
        # if energy is zero, we get problems with log
        energy = np.where(energy == 0, np.finfo(float).eps, energy)

        # compute the filterbank energies
        feat = np.dot(pspec, self._filterbanks.T)
        # if feat is zero, we get problems with log
        feat = np.where(feat == 0, np.finfo(float).eps, feat)

        return feat, energy

    def _get_filterbanks(self):
        """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

        Returns:
            A numpy array of size num_filt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """

        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = np.floor((self.nfft+1)*self._mel2hz(self.mel_points)/self.fs)

        fbank = np.zeros([self.num_filt, int(self.nfft/2+1)])
        for j in xrange(0, self.num_filt):
            for i in xrange(int(bin[j]), int(bin[j+1])):
                fbank[j, i] = (i - bin[j])/(bin[j+1]-bin[j])
            for i in xrange(int(bin[j+1]), int(bin[j+2])):
                fbank[j, i] = (bin[j+2]-i)/(bin[j+2]-bin[j+1])
        return fbank

    def _hz2mel(self, hz):
        """Convert a value in Hertz to Mels

        Args:
            hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.

        Returns:
            A value in Mels. If an array was passed in, an identical sized array is returned.
        """
        return 2595 * np.log10(1+hz/700.0)

    def _mel2hz(self, mel):
        """Convert a value in Mels to Hertz

        Args:
            mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.

        Returns:
            A value in Hertz. If an array was passed in, an identical sized array is returned.
        """
        return 700*(10**(mel/2595.0)-1)


class MFCC(FBank):
    """Compute MFCC features from an audio signal.
    """

    def __init__(self, fs=16e3, win_len=0.025, win_step=0.01, num_cep=13,
                 num_filt=40, nfft=512, low_freq=20, high_freq=7800,
                 pre_emph=0.97, cep_lifter=22, append_energy=False,
                 win_fun=signal.hamming, d=True, dd=True, eps=1e-14):
        """ Constructor of class

            Args:
                fs: the samplerate of the signal we are working with.
                    Default is 8e3
                win_len: the length of the analysis window in seconds.
                    Default  is 0.025s (25 milliseconds)
                win_step: the step between successive windows in seconds.
                    Default is 0.01s (10 milliseconds)
                num_cep: the number of cepstrum to return. Default 13.
                num_filt: the number of filters in the filterbank, default 40.
                nfft: the FFT size. Default is 512.
                low_freq: lowest band edge of mel filters in Hz.
                    Default is 20.
                high_freq: highest band edge of mel filters in Hz.
                    Default is 7800
                pre_emph: apply preemphasis filter with preemph as coefficient.
                0 is no filter. Default is 0.97.
                cep_lifter: apply a lifter to final cepstral coefficients. 0 is
                no lifter. Default is 22.
                append_energy: if this is true, the zeroth cepstral coefficient
                is replaced with the log of the total frame energy.
                win_func: the analysis window to apply to each frame.
                    By default hamming window is applied.
                d: if True add deltas coeficients. Default True
                dd: if True add delta-deltas coeficients. Default True
        """

        super(MFCC, self).__init__(fs=fs, win_len=win_len, win_step=win_step,
                                   num_filt=num_filt, nfft=nfft,
                                   low_freq=low_freq, high_freq=high_freq,
                                   pre_emph=pre_emph, win_fun=win_fun, eps=eps)

        self.num_cep = num_cep
        self.cep_lifter = cep_lifter
        self.append_energy = append_energy
        self.d = d
        self.dd = dd

        self._logger = logging.getLogger('%s.%s' % (__name__, self.__class__.__name__))

    def __call__(self, signal):
        """Compute MFCC features from an audio signal.

        Args:
            signal: the audio signal from which to compute features. Should be
            an N*1 array

        Returns:
            A numpy array of size (NUMFRAMES by numcep) containing features.
            Each row holds 1 feature vector.
        """
        feat, energy = super(MFCC, self).__call__(signal)

        feat = np.log(feat)
        feat = dct(feat, type=2, axis=1, norm='ortho')[:, :self.num_cep]
        feat = self._lifter(feat, self.cep_lifter)

        if self.d:
            d = delta(feat, 2)
            feat = np.hstack([feat, d])

            if self.dd:
                feat = np.hstack([feat, delta(d, 2)])

        if self.append_energy:
            # replace first cepstral coefficient with log of frame energy
            feat[:, 0] = np.log(energy + self.eps)

        return feat

    def _lifter(self, cepstra, L=22):
        """Apply a cepstral lifter the the matrix of cepstra.

        This has the effect of increasing the magnitude of the high frequency
        DCT coeffs.

        Args:
            cepstra: the matrix of mel-cepstra, will be numframes * numcep in
            size.
            L: the liftering coefficient to use. Default is 22. L <= 0 disables
            lifter.
        """
        if L > 0:
            nframes, ncoeff = np.shape(cepstra)
            n = np.arange(ncoeff)
            lift = 1 + (L/2)*np.sin(np.pi*n/L)
            return lift*cepstra
        else:
            # values of L <= 0, do nothing
            return cepstra

class LogFbank(FBank):
    """Compute Mel-filterbank energy features from an audio signal.
    """

    def __init__(self, fs=16e3, win_len=0.025, win_step=0.01,
                 num_filt=40, nfft=512, low_freq=20, high_freq=7800,
                 pre_emph=0.97, win_fun=signal.hamming, append_energy=False, d=True, dd=True):
        """Constructor
        """

        super(LogFbank, self).__init__(fs=fs, win_len=win_len, win_step=win_step,
                                   num_filt=num_filt, nfft=nfft,
                                   low_freq=low_freq, high_freq=high_freq,
                                   pre_emph=pre_emph, win_fun=win_fun)

        self.d = d
        self.dd = dd
        self.append_energy = append_energy

        self._logger = logging.getLogger('%s.%s' % (__name__, self.__class__.__name__))

    def __call__(self, signal):
        """Compute log Mel-filterbank energy features from an audio signal.
        :param signal: the audio signal from which to compute features. Should be an N*1 array

        Returns:
             A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
        """

        feat, energy = super(LogFbank, self).__call__(signal)

        feat = np.log(feat)

        if self.append_energy:
            feat = np.hstack([feat, np.log(energy + self.eps)[:, np.newaxis]])


        if self.d:
            d = delta(feat, 2)
            feat = np.hstack([feat, d])

            if self.dd:
                feat = np.hstack([feat, delta(d, 2)])

        return feat

class Raw(Feature):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, signal):
        return signal

raw = lambda x: Raw()(x)

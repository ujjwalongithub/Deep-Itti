import typing

import kornia.geometry.transform as KT
import torch


class IttiKochParams(object):
    """
    This class is just a convenience class definition to encapsulate
    the parameters of Itti-Koch Saliency
    """

    def __init__(self,
                 num_levels: int = 4,
                 cs_upper: tuple = (1, 2, 3),
                 cs_lower: tuple = (0, 1, 2),
                 normalization_iterations: int = 5
                 ):
        """
        Class initializer.
        :param num_levels: Number of levels in the gaussian pyramid
        :param cs_upper: Pyramid levels to subtract from. Please see _check_params()
        :param cs_lower: Pyramid levels to subtract. Please see _check_params()
        :param normalization_iterations: Number of iterations of DoG filter + normalization
        """
        super(IttiKochParams, self).__init__()
        self._check_params(num_levels, cs_upper, cs_lower, normalization_iterations)
        self._num_levels = num_levels
        self._cs_upper = cs_upper
        self._cs_lower = cs_lower
        self._normalization_iterations = normalization_iterations

    @property
    def num_levels(self):
        """
        Returns number of levels in the gaussian pyramid
        :return: number of levels in the gaussian pyramid
        """
        return self._num_levels

    @property
    def cs_lower(self):
        """
        Returns the pyramid levels to subtract from.
        :return: Tuple of pyramid levels to subtract from.
        """
        return self._cs_lower

    @property
    def cs_upper(self):
        """
        Returns the pyramid levels to subtract for laplacian
        :return: Tuple of pyramid levels to subtract.
        """
        return self._cs_upper

    @property
    def normalization_iterations(self):
        """
        Returns the number of times DoG filter application + normalization is carried out.
        :return: Number of normalization iterations
        """
        return self._normalization_iterations

    def _check_params(self, num_levels, cs_upper, cs_lower, normalization_iterations):
        """
        Checks the correctness of the initializer arguments
         :param num_levels: Number of levels in the gaussian pyramid
        :param cs_upper: Pyramid levels to subtract from. Please see _check_params()
        :param cs_lower: Pyramid levels to subtract. Please see _check_params()
        :param normalization_iterations: Number of iterations of DoG filter + normalization
        :return: None if everything is fine. Else raises a ValueError
        """
        if not cs_upper == tuple(sorted(cs_upper)):
            raise ValueError('cs_upper must be a tuple with elements in '
                             'ascending order.')

        if not cs_lower == tuple(sorted(cs_lower)):
            raise ValueError('cs_lower must be a tuple with elements in '
                             'ascending order.')

        if not num_levels > 0:
            raise ValueError('num_levels must be a positive integer.')

        if not max(cs_upper) == num_levels - 1:
            raise ValueError('Maximum entry in cs_upper must be num_levels-1.')

        if not max(cs_lower) == num_levels - 2:
            raise ValueError('Maximum entry in cs_lower must be num_levels-2.')

        if not normalization_iterations > 0:
            raise ValueError('normalization_iterations must be a positive integer.')

        return None


class IttiKochSaliency(torch.nn.Module):
    """
    PyTorch module implementing the Itti-Koch Saliency
    """

    def __init__(self,
                 params: typing.Union[IttiKochParams, None] = None
                 ):
        """
        Class initializer
        :param params: None or an instance IttiKochParams. If None, the module
        instantiates IttiKochParams based on default values
        """
        super(IttiKochSaliency, self).__init__()
        if params is None:
            self._params = IttiKochParams()
        else:
            self._params = params

    def forward(self, x):
        # We first normalize the input per-channel in [0,1].
        x_normalized = self._normalize_per_channel(x)
        # We then build the Gaussian Pyramid from the normalized input
        gp = KT.build_pyramid(x_normalized, max_level=self._params.num_levels)

        pass

    @staticmethod
    def _normalize_per_channel(x):
        """
        Given a tensor [B,C,H,W], normalizes each channel
        in the range [0,1].
        :param x: A Tensor of shape [B,C,H,W]
        :return: A normalized version of x, normalized in [0,1] per channel.
        """
        max_pc = torch.amax(x, dim=(2, 3), keepdim=True)
        min_pc = torch.amin(x, dim=(2, 3), keepdim=True)
        x = (x - min_pc) / (max_pc - min_pc)
        return x

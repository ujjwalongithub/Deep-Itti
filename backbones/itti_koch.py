import typing
from loguru import logger
import kornia.geometry.transform as KT
import torch
import torch.nn.functional as F


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
        self._check_params(num_levels, cs_upper, cs_lower,
                           normalization_iterations)
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

    def _check_params(self, num_levels, cs_upper, cs_lower,
                      normalization_iterations):
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
            raise ValueError(
                'normalization_iterations must be a positive integer.')

        return None


class IttiKochSaliency(torch.nn.Module):
    """
    PyTorch module implementing the Itti-Koch Saliency
    Inp: [B,C,H,W]
    Output: [B,H,W]
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
        inp = self._normalize_per_channel(x)
        # We then build the Gaussian Pyramid from the normalized input
        max_level = int(inp.shape[2] / 1).bit_length() - 1
        u = max_level - 1
        l = max_level - 1

        # gaussian pyramid
        gau_pyr = KT.build_pyramid(inp, max_level)
        #         gau_pyr = KT.build_pyramid(inp, max_level=self._params.num_levels)

        # center surround
        dst = list()
        for i in range(u - 1):
            h, w = gau_pyr[i + 1].shape[2], gau_pyr[i + 1].shape[3]
            # for j in range(l-u):
            #     tmp = cv2.resize(gau_pyr[-j-1], (w, h))
            #     nowdst = cv2.absdiff(gau_pyr[u], tmp)
            #     dst.append(nowdst)
            tmp: torch.Tensor = F.interpolate(
                gau_pyr[i + 2], size=(h, w), mode='nearest'
            )
            nowdst = torch.abs(gau_pyr[i + 1] - tmp)
            dst.append(nowdst)
        logger.info(dst)

        # normalisation
        for i in range(len(dst)):
            logger.info('Before normalization dst[{}]={}'.format(i, dst[i]))
            dst[i] = self.normalizeFeatureMaps(dst[i], gau_pyr[0].shape[2],
                                               gau_pyr[0].shape[3])
            logger.info('After normalization dst[{}]={}'.format(i, dst[i]))

        # add
        dst = torch.stack(dst, dim=0)
        dst = torch.sum(dst, dim=0)

        # mean across channels
        dst = torch.mean(dst, dim=1)

        return dst

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

    def SMAvgLocalMax(self,src):

        # size
        stepsize = 16
        width = src.shape[3]
        height = src.shape[2]
        # find local maxima
        numlocal = torch.zeros(src.shape[0], src.shape[1], 1, 1)
        lmaxmean = torch.zeros(src.shape[0], src.shape[1], 1, 1)
        for y in range(0, height - stepsize, stepsize):
            for x in range(0, width - stepsize, stepsize):
                localimg = src[:, :, y:y + stepsize, x:x + stepsize]
                lmax = torch.amax(localimg, dim=(2, 3), keepdim=True)
                lmin = torch.amin(localimg, dim=(2, 3), keepdim=True)
                # lmin, lmax, dummy1, dummy2 = cv2.minMaxLoc(localimg)
                lmaxmean += lmax
                numlocal += 1
        # averaging over all the local regions
        logger.info("lmaxmean={}\t numlocal={}.".format(lmaxmean, numlocal))
        output = lmaxmean / numlocal

        return torch.nan_to_num(output)

    ## normalization specific for the saliency map model
    def SMNormalization(self, src):
        dst = self._normalize_per_channel(src)
        lmaxmean = self.SMAvgLocalMax(dst)
        normcoeff = (1 - lmaxmean) * (1 - lmaxmean)
        return dst * normcoeff

    ## normalizing feature maps
    def normalizeFeatureMaps(self, x, h, w):
        normalizedImage = self.SMNormalization(x)
        normalizedImage = F.interpolate(
            normalizedImage, size=(h, w), mode='bilinear'
        )
        return normalizedImage
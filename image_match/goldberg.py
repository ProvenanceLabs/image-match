from skimage.color import rgb2gray
from skimage.io import imread
from PIL import Image
from PIL.MpoImagePlugin import MpoImageFile
try:
    from cairosvg import svg2png
except ImportError:
    pass
from six import string_types, text_type
from io import BytesIO
import numpy as np
import xml.etree


class CorruptImageError(RuntimeError):
    pass


class ImageSignature(object):
    """Image signature generator.

    Based on the method of Goldberg, et al. Available at http://www.cs.cmu.edu/~hcwong/Pdfs/icip02.ps
    """

    def __init__(self, n=9, crop_percentiles=(5, 95), P=None, diagonal_neighbors=True,
                 identical_tolerance=2/255., n_levels=2, fix_ratio=False):
        """Initialize the signature generator.

        The default parameters match those given in Goldberg's paper.

        Note:
            Non-default parameters have not been extensively tested. Use carefully.

        Args:
            n (Optional[int]): size of grid imposed on image. Grid is n x n (default 9)
            crop_percentiles (Optional[Tuple[int]]): lower and upper bounds when considering how much
                variance to keep in the image (default (5, 95))
            P (Optional[int]): size of sample region, P x P. If none, uses a sample region based
                on the size of the image (default None)
            diagonal_neighbors (Optional[boolean]): whether to include diagonal grid neighbors
                (default True)
            identical_tolerance (Optional[float]): cutoff difference for declaring two adjacent
                grid points identical (default 2/255)
            n_levels (Optional[int]): number of positive and negative groups to stratify neighbor
                differences into. n = 2 -> [-2, -1, 0, 1, 2] (default 2)

        """

        # check inputs
        assert crop_percentiles is None or len(crop_percentiles) == 2,\
            'crop_percentiles should be a two-value tuple, or None'
        if crop_percentiles is not None:
            assert crop_percentiles[0] >= 0,\
                'Lower crop_percentiles limit should be > 0 (%r given)'\
                % crop_percentiles[0]
            assert crop_percentiles[1] <= 100,\
                'Upper crop_percentiles limit should be < 100 (%r given)'\
                % crop_percentiles[1]
            assert crop_percentiles[0] < crop_percentiles[1],\
                'Upper crop_percentile limit should be greater than lower limit.'
            self.lower_percentile = crop_percentiles[0]
            self.upper_percentile = crop_percentiles[1]
            self.crop_percentiles = crop_percentiles
        else:
            self.crop_percentiles = crop_percentiles
            self.lower_percentile = 0
            self.upper_percentile = 100

        assert type(n) is int, 'n should be an integer > 1'
        assert n > 1, 'n should be greater than 1 (%r given)' % n
        self.n = n

        assert type(P) is int or P is None, 'P should be an integer >= 1, or None'
        if P is not None:
            assert P >= 1, 'P should be greater than 0 (%r given)' % n
        self.P = P

        assert type(diagonal_neighbors) is bool, 'diagonal_neighbors should be boolean'
        self.diagonal_neighbors = diagonal_neighbors
        self.sig_length = self.n ** 2 * (4 + self.diagonal_neighbors * 4)

        assert type(fix_ratio) is bool, 'fix_ratio should be boolean'
        self.fix_ratio = fix_ratio

        assert type(identical_tolerance) is float or type(identical_tolerance) is int,\
            'identical_tolerance should be a number between 1 and 0'
        assert 0. <= identical_tolerance <= 1.,\
            'identical_tolerance should be greater than zero and less than one (%r given)' % identical_tolerance
        self.identical_tolerance = identical_tolerance

        assert type(n_levels) is int, 'n_levels should be an integer'
        assert n_levels > 0
        'n_levels should be > 0 (%r given)' % n_levels
        self.n_levels = n_levels

        self.handle_mpo = True

    def generate_signature(self, path_or_image, bytestream=False):
        """Generates an image signature.

        See section 3 of Goldberg, et al.

        Args:
            path_or_image (string or numpy.ndarray): image path, or image array
            bytestream (Optional[boolean]): will the image be passed as raw bytes?
                That is, is the 'path_or_image' argument an in-memory image?
                (default False)

        Returns:
            The image signature: A rank 1 numpy array of length n x n x 8
                (or n x n x 4 if diagonal_neighbors == False)

        Examples:
            >>> from image_match.goldberg import ImageSignature
            >>> gis = ImageSignature()
            >>> gis.generate_signature('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
            array([ 0,  0,  0,  0,  0,  0,  2,  2,  0,  0,  0,  0,  0,  2,  2,  2,  0,
                    0,  0,  0,  2,  2,  2,  2,  0,  0,  0, -2,  2,  2,  1,  2,  0,  0,
                    0, -2,  2, -1, -1,  2,  0,  0,  0, -2, -2, -2, -2, -1,  0,  0,  0,
                    2, -1,  2,  2,  2,  0,  0,  0,  1, -1,  2,  2, -1,  0,  0,  0,  1,
                    0,  2, -1,  0,  0, -2, -2,  0, -2,  0,  2,  2, -2, -2, -2,  2,  2,
                    2,  2,  2, -2, -2, -2, -2, -2,  1,  2, -2, -2, -1,  1,  2,  1,  2,
                   -1,  1, -2,  1,  2, -1,  2, -1,  0,  2, -2,  2, -2, -2,  1, -2,  1,
                    2,  1, -2, -2, -1, -2,  1,  1, -1, -2, -2, -2,  2, -2,  2,  2,  2,
                    1,  1,  0,  2,  0,  2,  2,  0,  0, -2, -2,  0,  1,  0, -1,  1, -2,
                   -2, -1, -1,  1, -1,  1,  1, -2, -2, -2, -1, -2, -1,  1, -1,  2,  1,
                    1,  2,  1,  2,  2, -1, -1,  0,  2, -1,  2,  2, -1, -1, -2, -1, -1,
                   -2,  1, -2, -2, -1, -2, -1, -2, -1, -2, -2, -1, -1,  1, -2, -2,  2,
                   -1,  1,  1, -2, -2, -2,  0,  1,  0,  1, -1,  0,  0,  1,  1,  0,  1,
                    0,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -2, -1, -1, -1,
                   -2, -2,  1, -2, -2,  1, -2, -2, -2, -2,  1,  1,  2,  2,  1, -2, -2,
                   -1,  1,  2,  2, -1,  2, -2, -1,  1,  1,  1, -1, -2, -1, -2, -2,  0,
                    1, -1, -1,  1, -2, -2,  0,  1,  2,  1,  0,  2,  0,  2,  2,  0,  0,
                   -1,  1,  0,  1,  0,  1,  2, -1, -1,  1, -1, -1, -1,  2,  1,  1,  2,
                    2,  1, -2,  2,  2,  1,  2,  2,  2,  2, -1,  2,  2,  2,  2,  2,  2,
                    1,  2,  2,  2,  2,  1,  1,  2, -2,  2,  2,  2,  2, -1,  2,  2, -2,
                    2,  2,  2,  2,  0,  0, -2, -2,  1,  0, -1,  1, -1, -2,  0, -1,  0,
                   -1,  1,  0,  0, -1,  1,  0,  2,  0,  2,  2, -2, -2, -2, -2, -1, -1,
                   -1,  0, -1, -2, -2,  1, -1, -1,  1,  1, -1, -2, -2,  1,  1,  1,  1,
                    2, -2, -2, -2, -1,  1,  1,  1,  2, -2, -2, -2, -1, -1,  0,  1,  1,
                   -2, -2,  0,  1, -1,  1,  1,  1, -2,  1,  1,  1,  2,  2,  2,  2, -1,
                   -1,  0, -2,  0,  0,  1,  0,  0, -2,  1,  0, -1,  0, -1, -2, -2,  1,
                    1,  1,  1, -1, -1, -2,  0, -1, -1, -1, -1, -2, -2, -2, -1, -1, -1,
                    1,  1, -2, -2,  1, -2, -1,  0, -1,  0, -2, -1,  1, -2, -1, -1,  0,
                    0, -1,  0,  0, -1, -1, -2,  0, -1,  0,  0, -1, -1, -2,  0,  1,  1,
                    1,  0,  1, -2, -1,  0, -1,  0, -1,  0,  0,  0,  1,  1,  0, -1,  0,
                    2, -1,  2,  1,  2,  1, -2,  2, -1, -2,  2,  2,  2,  2,  2,  2,  2,
                    2,  2,  2,  2, -2,  2,  1,  2,  2, -1,  1,  1, -2,  1, -2, -2, -1,
                   -1,  0,  0, -1,  0, -2, -1, -1,  0,  0, -1,  0, -1, -1, -1, -1,  1,
                    0,  1,  1,  1, -1,  0,  1, -1,  0,  0, -1,  0, -1,  0,  0,  0, -2,
                   -2,  0, -2,  0,  0,  0,  1,  1, -2,  2, -2,  0,  0,  0,  2, -2, -1,
                    2,  2,  0,  0,  0, -2, -2,  2, -2,  1,  0,  0,  0, -2,  2,  2, -1,
                    2,  0,  0,  0,  1,  1,  1, -2,  1,  0,  0,  0,  1,  1,  1, -1,  1,
                    0,  0,  0,  1,  0,  1, -1,  1,  0,  0,  0, -1,  0,  0, -1,  0,  0,
                    0,  0], dtype=int8)

        """

        # Step 1:    Load image as array of grey-levels
        im_array = self.preprocess_image(path_or_image, handle_mpo=self.handle_mpo, bytestream=bytestream)

        # Step 2a:   Determine cropping boundaries
        if self.crop_percentiles is not None:
            image_limits = self.crop_image(im_array,
                                           lower_percentile=self.lower_percentile,
                                           upper_percentile=self.upper_percentile,
                                           fix_ratio=self.fix_ratio)
        else:
            image_limits = None

        # Step 2b:   Generate grid centers
        x_coords, y_coords = self.compute_grid_points(im_array,
                                                      n=self.n, window=image_limits)

        # Step 3:    Compute grey level mean of each P x P
        #           square centered at each grid point
        avg_grey = self.compute_mean_level(im_array, x_coords, y_coords, P=self.P)

        # Step 4a:   Compute array of differences for each
        #           grid point vis-a-vis each neighbor
        diff_mat = self.compute_differentials(avg_grey,
                                              diagonal_neighbors=self.diagonal_neighbors)

        # Step 4b: Bin differences to only 2n+1 values
        self.normalize_and_threshold(diff_mat,
                                     identical_tolerance=self.identical_tolerance,
                                     n_levels=self.n_levels)

        # Step 5: Flatten array and return signature
        return np.ravel(diff_mat).astype('int8')

    @staticmethod
    def preprocess_image(image_or_path, bytestream=False, handle_mpo=False):
        """Loads an image and converts to greyscale.

        Corresponds to 'step 1' in Goldberg's paper

        Args:
            image_or_path (string or numpy.ndarray): image path, or image array
            bytestream (Optional[boolean]): will the image be passed as raw bytes?
                That is, is the 'path_or_image' argument an in-memory image?
                (default False)
            handle_mpo (Optional[boolean]): try to compute a signature for steroscopic
                images by extracting the first image of the set (default False)

        Returns:
            Array of floats corresponding to greyscale level at each pixel

        Examples:
            >>> gis.preprocess_image('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
            array([[ 0.26344431,  0.32423294,  0.30406745, ...,  0.35069725,
                     0.36499961,  0.36361569],
                   [ 0.29676627,  0.28640118,  0.34523255, ...,  0.3703051 ,
                     0.34931333,  0.31655686],
                   [ 0.35305216,  0.31858431,  0.36202   , ...,  0.40588196,
                     0.37284275,  0.30871373],
                   ...,
                   [ 0.05932863,  0.05540706,  0.05540706, ...,  0.01954745,
                     0.01954745,  0.01562588],
                   [ 0.0632502 ,  0.05540706,  0.05148549, ...,  0.01954745,
                     0.02346902,  0.01562588],
                   [ 0.06717176,  0.05540706,  0.05148549, ...,  0.02346902,
                     0.02739059,  0.01954745]])

        """
        if bytestream:
            try:
                img = Image.open(BytesIO(image_or_path))
            except IOError:
                # could be an svg, attempt to convert
                try:
                    img = Image.open(BytesIO(svg2png(image_or_path)))
                except (NameError, xml.etree.ElementTree.ParseError):
                    raise CorruptImageError()
            img = img.convert('RGB')
            return rgb2gray(np.asarray(img, dtype=np.uint8))
        elif type(image_or_path) in string_types or \
             type(image_or_path) is text_type:
            return imread(image_or_path, as_gray=True)
        elif type(image_or_path) is bytes:
            try:
                img = Image.open(image_or_path)
                arr = np.array(img.convert('RGB'))
            except IOError:
                # try again due to PIL weirdness
                return imread(image_or_path, as_grey=True)
            if handle_mpo:
                # take the first images from the MPO
                if arr.shape == (2,) and isinstance(arr[1].tolist(), MpoImageFile):
                    return rgb2gray(arr[0])
                else:
                    return rgb2gray(arr)
            else:
                return rgb2gray(arr)
        elif type(image_or_path) is np.ndarray:
            return rgb2gray(image_or_path)
        else:
            raise TypeError('Path or image required.')

    @staticmethod
    def crop_image(image, lower_percentile=5, upper_percentile=95, fix_ratio=False):
        """Crops an image, removing featureless border regions.

        Corresponds to the first part of 'step 2' in Goldberg's paper

        Args:
            image (numpy.ndarray): n x m array of floats -- the greyscale image. Typically, the
                output of preprocess_image
            lower_percentile (Optional[int]): crop image by percentage of difference (default 5)
            upper_percentile (Optional[int]): as lower_percentile (default 95)
            fix_ratio (Optional[boolean]): use the larger ratio for both directions. This is useful
                for using the fast signature transforms on sparse but very similar images (e.g.
                renderings from fixed directions). Use with care -- only use if you can guarantee the
                incoming image is square (default False).

        Returns:
            A pair of tuples describing the 'window' of the image to use in analysis: [(top, bottom), (left, right)]

        Examples:
            >>> img = gis.preprocess_image('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
            >>> gis.crop_image(img)
            [(36, 684), (24, 452)]

        """
        # row-wise differences
        rw = np.cumsum(np.sum(np.abs(np.diff(image, axis=1)), axis=1))
        # column-wise differences
        cw = np.cumsum(np.sum(np.abs(np.diff(image, axis=0)), axis=0))

        # compute percentiles
        upper_column_limit = np.searchsorted(cw,
                                             np.percentile(cw, upper_percentile),
                                             side='left')
        lower_column_limit = np.searchsorted(cw,
                                             np.percentile(cw, lower_percentile),
                                             side='right')
        upper_row_limit = np.searchsorted(rw,
                                          np.percentile(rw, upper_percentile),
                                          side='left')
        lower_row_limit = np.searchsorted(rw,
                                          np.percentile(rw, lower_percentile),
                                          side='right')

        # if image is nearly featureless, use default region
        if lower_row_limit > upper_row_limit:
            lower_row_limit = int(lower_percentile/100.*image.shape[0])
            upper_row_limit = int(upper_percentile/100.*image.shape[0])
        if lower_column_limit > upper_column_limit:
            lower_column_limit = int(lower_percentile/100.*image.shape[1])
            upper_column_limit = int(upper_percentile/100.*image.shape[1])

        # if fix_ratio, return both limits as the larger range
        if fix_ratio:
            if (upper_row_limit - lower_row_limit) > (upper_column_limit - lower_column_limit):
                return [(lower_row_limit, upper_row_limit),
                        (lower_row_limit, upper_row_limit)]
            else:
                return [(lower_column_limit, upper_column_limit),
                        (lower_column_limit, upper_column_limit)]

        # otherwise, proceed as normal
        return [(lower_row_limit, upper_row_limit),
                (lower_column_limit, upper_column_limit)]

    @staticmethod
    def compute_grid_points(image, n=9, window=None):
        """Computes grid points for image analysis.

        Corresponds to the second part of 'step 2' in the paper

        Args:
            image (numpy.ndarray): n x m array of floats -- the greyscale image. Typically,
                the output of preprocess_image
            n (Optional[int]): number of gridpoints in each direction (default 9)
            window (Optional[List[Tuple[int]]]): limiting coordinates [(t, b), (l, r)], typically the
                output of (default None)

        Returns:
            tuple of arrays indicating the vertical and horizontal locations of the grid points

        Examples:
            >>> img = gis.preprocess_image('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
            >>> window = gis.crop_image(img)
            >>> gis.compute_grid_points(img, window=window)
            (array([100, 165, 230, 295, 360, 424, 489, 554, 619]),
             array([ 66, 109, 152, 195, 238, 280, 323, 366, 409]))

        """

        # if no limits are provided, use the entire image
        if window is None:
            window = [(0, image.shape[0]), (0, image.shape[1])]

        x_coords = np.linspace(window[0][0], window[0][1], n + 2, dtype=int)[1:-1]
        y_coords = np.linspace(window[1][0], window[1][1], n + 2, dtype=int)[1:-1]

        return x_coords, y_coords      # return pairs

    @staticmethod
    def compute_mean_level(image, x_coords, y_coords, P=None):
        """Computes array of greyness means.

        Corresponds to 'step 3'

        Args:
            image (numpy.ndarray): n x m array of floats -- the greyscale image. Typically,
                the output of preprocess_image
            x_coords (numpy.ndarray): array of row numbers
            y_coords (numpy.ndarray): array of column numbers
            P (Optional[int]): size of boxes in pixels (default None)

        Returns:
            an N x N array of average greyscale around the gridpoint, where N is the
                number of grid points

        Examples:
            >>> img = gis.preprocess_image('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
            >>> window = gis.crop_image(img)
            >>> grid = gis.compute_grid_points(img, window=window)
            >>> gis.compute_mean_level(img, grid[0], grid[1])
            array([[ 0.62746325,  0.62563642,  0.62348078,  0.50651686,  0.37438874,
                     0.0644063 ,  0.55968952,  0.59356148,  0.60473832],
                   [ 0.35337797,  0.50272543,  0.27711346,  0.42384226,  0.39006181,
                     0.16773968,  0.10471924,  0.33647144,  0.62902124],
                   [ 0.20307514,  0.19021892,  0.12435402,  0.44990121,  0.38527996,
                     0.08339507,  0.05530059,  0.18469107,  0.21125228],
                   [ 0.25727387,  0.1669419 ,  0.08964046,  0.1372754 ,  0.48529236,
                     0.39894004,  0.10387907,  0.11282135,  0.30014612],
                   [ 0.23447867,  0.15702549,  0.25232943,  0.75172715,  0.79488688,
                     0.4943538 ,  0.29645163,  0.10714578,  0.0629376 ],
                   [ 0.22167555,  0.04839472,  0.10125833,  0.1550749 ,  0.14346914,
                     0.04713144,  0.10095568,  0.15349296,  0.04456733],
                   [ 0.09233709,  0.11210942,  0.05361996,  0.07066566,  0.04191625,
                     0.03548839,  0.03420656,  0.05025029,  0.03519956],
                   [ 0.19226873,  0.20647194,  0.62972106,  0.45514529,  0.05620413,
                     0.03383168,  0.03413588,  0.04741828,  0.02987698],
                   [ 0.05799523,  0.23310153,  0.43719717,  0.27666873,  0.25106573,
                     0.11094163,  0.10180622,  0.04633349,  0.02704855]])

        """

        if P is None:
            P = max([2.0, int(0.5 + min(image.shape)/20.)])     # per the paper

        avg_grey = np.zeros((x_coords.shape[0], y_coords.shape[0]))

        for i, x in enumerate(x_coords):        # not the fastest implementation
            lower_x_lim = int(max([x - P/2, 0]))
            upper_x_lim = int(min([lower_x_lim + P, image.shape[0]]))
            for j, y in enumerate(y_coords):
                lower_y_lim = int(max([y - P/2, 0]))
                upper_y_lim = int(min([lower_y_lim + P, image.shape[1]]))

                avg_grey[i, j] = np.mean(image[lower_x_lim:upper_x_lim,
                                        lower_y_lim:upper_y_lim])  # no smoothing here as in the paper

        return avg_grey

    @staticmethod
    def compute_differentials(grey_level_matrix,  diagonal_neighbors=True):
        """Computes differences in greylevels for neighboring grid points.

        First part of 'step 4' in the paper.

        Returns n x n x 8 rank 3 array for an n x n grid (if diagonal_neighbors == True)

        The n x nth coordinate corresponds to a grid point.  The eight values are
        the differences between neighboring grid points, in this order:

        upper left
        upper
        upper right
        left
        right
        lower left
        lower
        lower right

        Args:
            grey_level_matrix (numpy.ndarray): grid of values sampled from image
            diagonal_neighbors (Optional[boolean]): whether or not to use diagonal
                neighbors (default True)

        Returns:
            a n x n x 8 rank 3 numpy array for an n x n grid (if diagonal_neighbors == True)

        Examples:
            >>> img = gis.preprocess_image('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
            >>> window = gis.crop_image(img)
            >>> grid = gis.compute_grid_points(img, window=window)
            >>> grey_levels = gis.compute_mean_level(img, grid[0], grid[1])
            >>> gis.compute_differentials(grey_levels)
            array([[[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                       0.00000000e+00,   1.82683143e-03,  -0.00000000e+00,
                       2.74085276e-01,   1.24737821e-01],
                    [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                      -1.82683143e-03,   2.15563930e-03,   2.72258444e-01,
                       1.22910990e-01,   3.48522956e-01],
                    [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                      -2.15563930e-03,   1.16963917e-01,   1.20755351e-01,
                       3.46367317e-01,   1.99638513e-01],
                    [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                      -1.16963917e-01,   1.32128118e-01,   2.29403399e-01,
                       8.26745956e-02,   1.16455050e-01],
                    ...

        """
        right_neighbors = -np.concatenate((np.diff(grey_level_matrix),
                                           np.zeros(grey_level_matrix.shape[0]).
                                           reshape((grey_level_matrix.shape[0], 1))),
                                          axis=1)
        left_neighbors = -np.concatenate((right_neighbors[:, -1:],
                                          right_neighbors[:, :-1]),
                                         axis=1)

        down_neighbors = -np.concatenate((np.diff(grey_level_matrix, axis=0),
                                          np.zeros(grey_level_matrix.shape[1]).
                                          reshape((1, grey_level_matrix.shape[1]))))

        up_neighbors = -np.concatenate((down_neighbors[-1:], down_neighbors[:-1]))

        if diagonal_neighbors:
            # this implementation will only work for a square (m x m) grid
            diagonals = np.arange(-grey_level_matrix.shape[0] + 1,
                                  grey_level_matrix.shape[0])

            upper_left_neighbors = sum(
                [np.diagflat(np.insert(np.diff(np.diag(grey_level_matrix, i)), 0, 0), i)
                 for i in diagonals])
            lower_right_neighbors = -np.pad(upper_left_neighbors[1:, 1:],
                                            (0, 1), mode='constant')

            # flip for anti-diagonal differences
            flipped = np.fliplr(grey_level_matrix)
            upper_right_neighbors = sum([np.diagflat(np.insert(
                np.diff(np.diag(flipped, i)), 0, 0), i) for i in diagonals])
            lower_left_neighbors = -np.pad(upper_right_neighbors[1:, 1:],
                                           (0, 1), mode='constant')

            return np.dstack(np.array([
                upper_left_neighbors,
                up_neighbors,
                np.fliplr(upper_right_neighbors),
                left_neighbors,
                right_neighbors,
                np.fliplr(lower_left_neighbors),
                down_neighbors,
                lower_right_neighbors]))

        return np.dstack(np.array([
            up_neighbors,
            left_neighbors,
            right_neighbors,
            down_neighbors]))

    @staticmethod
    def normalize_and_threshold(difference_array,
                                identical_tolerance=2/255., n_levels=2):
        """Normalizes difference matrix in place.

        'Step 4' of the paper.  The flattened version of this array is the image signature.

        Args:
            difference_array (numpy.ndarray): n x n x l array, where l are the differences between
                the grid point and its neighbors. Typically the output of compute_differentials
            identical_tolerance (Optional[float]): maximum amount two gray values can differ and
                still be considered equivalent (default 2/255)
            n_levels (Optional[int]): bin differences into 2 n + 1 bins (e.g. n_levels=2 -> [-2, -1,
                0, 1, 2])

        Examples:
            >>> img = gis.preprocess_image('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
            >>> window = gis.crop_image(img)
            >>> grid = gis.compute_grid_points(img, window=window)
            >>> grey_levels = gis.compute_mean_level(img, grid[0], grid[1])
            >>> m = gis.compute_differentials(grey_levels)
            >>> m
            array([[[ 0.,  0.,  0.,  0.,  0.,  0.,  2.,  2.],
                    [ 0.,  0.,  0.,  0.,  0.,  2.,  2.,  2.],
                    [ 0.,  0.,  0.,  0.,  2.,  2.,  2.,  2.],
                    [ 0.,  0.,  0., -2.,  2.,  2.,  1.,  2.],
                    [ 0.,  0.,  0., -2.,  2., -1., -1.,  2.],
                    [ 0.,  0.,  0., -2., -2., -2., -2., -1.],
                    [ 0.,  0.,  0.,  2., -1.,  2.,  2.,  2.],
                    [ 0.,  0.,  0.,  1., -1.,  2.,  2., -1.],
                    [ 0.,  0.,  0.,  1.,  0.,  2., -1.,  0.]],

                   [[ 0., -2., -2.,  0., -2.,  0.,  2.,  2.],
                    [-2., -2., -2.,  2.,  2.,  2.,  2.,  2.],
                    [-2., -2., -2., -2., -2.,  1.,  2., -2.],
                    [-2., -1.,  1.,  2.,  1.,  2., -1.,  1.],
                    [-2.,  1.,  2., -1.,  2., -1.,  0.,  2.],
                    ...

        """

        # set very close values as equivalent
        mask = np.abs(difference_array) < identical_tolerance
        difference_array[mask] = 0.

        # if image is essentially featureless, exit here
        if np.all(mask):
            return None

        # bin so that size of bins on each side of zero are equivalent
        positive_cutoffs = np.percentile(difference_array[difference_array > 0.],
                                         np.linspace(0, 100, n_levels+1))
        negative_cutoffs = np.percentile(difference_array[difference_array < 0.],
                                         np.linspace(100, 0, n_levels+1))

        for level, interval in enumerate([positive_cutoffs[i:i+2]
                                          for i in range(positive_cutoffs.shape[0] - 1)]):
            difference_array[(difference_array >= interval[0]) &
                             (difference_array <= interval[1])] = level + 1

        for level, interval in enumerate([negative_cutoffs[i:i+2]
                                          for i in range(negative_cutoffs.shape[0] - 1)]):
            difference_array[(difference_array <= interval[0]) &
                             (difference_array >= interval[1])] = -(level + 1)

        return None

    @staticmethod
    def normalized_distance(_a, _b):
        """Compute normalized distance between two points.

        Computes || b - a || / ( ||b|| + ||a||)

        Args:
            _a (numpy.ndarray): array of size m
            _b (numpy.ndarray): array of size m

        Returns:
            normalized distance between signatures (float)

        Examples:
            >>> a = gis.generate_signature('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
            >>> b = gis.generate_signature('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
            >>> gis.normalized_distance(a, b)
            0.22095170140933634

        """
        b = _b.astype(int)
        a = _a.astype(int)
        norm_diff = np.linalg.norm(b - a)
        norm1 = np.linalg.norm(b)
        norm2 = np.linalg.norm(a)
        return norm_diff / (norm1 + norm2)

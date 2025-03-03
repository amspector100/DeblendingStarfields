import numpy as np
import torch

import fitsio

from astropy.io import fits
from astropy.wcs import WCS

import deblending_runjingdev.sdss_dataset_lib as sdss_dataset_lib
from deblending_runjingdev.sdss_dataset_lib import _get_mgrid2

def load_data(catalog_file = '../coadd_field_catalog_runjing_liu.fit',
                sdss_dir = '../sdss_stage_dir/',
                run = 94, camcol = 1, field = 12, bands = [2, 3],
                align_bands = True):

    n_bands = len(bands)

    band_letters = ['ugriz'[bands[i]] for i in range(n_bands)]


    ##################
    # load sdss data
    ##################
    sdss_data = sdss_dataset_lib.SloanDigitalSkySurvey(sdssdir = sdss_dir,
                                      run = run, camcol = camcol,
                                      field = field, bands = bands)

    image = torch.Tensor(sdss_data[0]['image'])
    slen0 = image.shape[-2]
    slen1 = image.shape[-1]

    ##################
    # load coordinate files
    ##################
    frame_names = ["frame-{}-{:06d}-{:d}-{:04d}.fits".format(band_letters[i],
                        run, camcol, field) for i in range(n_bands)]

    wcs_list = []
    for i in range(n_bands):
        hdulist = fits.open(sdss_dir + str(run) + '/' + str(camcol) + '/' + str(field) + \
                            '/' + frame_names[i])
        wcs_list += [WCS(hdulist['primary'].header)]

    min_coords = wcs_list[0].wcs_pix2world(np.array([[0, 0]]), 0)
    max_coords = wcs_list[0].wcs_pix2world(np.array([[slen1, slen0]]), 0)

    ##################
    # load catalog
    ##################
    fits_file = fitsio.FITS(catalog_file)[1]
    true_ra = fits_file['ra'][:]
    true_decl = fits_file['dec'][:]

    # make sure our catalog covers the whole image
    assert true_ra.min() < min_coords[0, 0]
    assert true_ra.max() > max_coords[0, 0]

    assert true_decl.min() < min_coords[0, 1]
    assert true_decl.max() > max_coords[0, 1]

    ##################
    # align image
    ##################
    if align_bands:
        pix_coords_list = [wcs_list[i].wcs_world2pix(true_ra, true_decl, 0, \
                                                    ra_dec_order = True) \
                           for i in range(n_bands)]


        for i in range(1, n_bands):
            shift_x0 = np.median(pix_coords_list[0][1] - pix_coords_list[i][1])
            shift_x1 = np.median(pix_coords_list[0][0] - pix_coords_list[i][0])

            grid = _get_mgrid2(slen0, slen1).unsqueeze(0) - \
                        torch.Tensor([[[[shift_x1 / (slen1 - 1),
                                        shift_x0 / (slen0 - 1)]]]]) * 2

            image_i = image[i].unsqueeze(0).unsqueeze(0)
            band_aligned = torch.nn.functional.grid_sample(image_i, grid,
                                mode = 'nearest', align_corners=True).squeeze()

            image[i] = band_aligned

    return image, fits_file, wcs_list, sdss_data

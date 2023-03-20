import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import cg
import scipy.interpolate as interp


def bicubic(source_img, scaling_factor):
    source_img_size = source_img.shape[0]
    x_or_y = np.array(list(range(0, int(source_img_size)))).astype(float)
    int_img = RectBivariateSpline(x_or_y, x_or_y, source_img)
    x_or_y_up = np.array(list(range(0, source_img_size * scaling_factor))).astype(float) / scaling_factor - 0.5

    x_grid, y_grid = np.meshgrid(x_or_y_up, x_or_y_up, indexing="ij")
    return int_img.ev(x_grid, y_grid)


def bicubic_with_mask(source,mask, scaling_factor):
    
    source_size = source.shape[0]

    H,W = source.shape[0] * scaling_factor, source.shape[1] * scaling_factor

    source_r = source.flatten()
    mask_r = mask.flatten()

    ### this part has been fixed to improve bicubic performance, nefore was not perfectly aligned
    x = np.arange((scaling_factor-1)/2,H,scaling_factor) #(0,source.shape[0])
    y = np.arange((scaling_factor-1)/2,H,scaling_factor) #(0,source.shape[1])

    x_g,y_g = np.meshgrid(x,y) # indexing="ij"

    x_g_r = x_g.flatten()
    y_g_r = y_g.flatten()

    source_r = source_r[mask_r==1]
    x_g_r = x_g_r[mask_r==1]
    y_g_r = y_g_r[mask_r==1]

    xy_g_r = np.concatenate([x_g_r[:,None],y_g_r[:,None]],axis=1)

    # second part being fixed
    #x_HR = np.linspace(0,source.shape[0]-1,endpoint=False,num=source.shape[0]*scaling_factor)
    #y_HR = np.linspace(0,source.shape[1]-1,endpoint=False,num=source.shape[1]*scaling_factor)
    x_HR = np.linspace(0,W,endpoint=False,num=W)
    y_HR = np.linspace(0,H,endpoint=False,num=H)

    x_HR_g,y_HR_g = np.meshgrid(x_HR,y_HR)
    x_HR_g,y_HR_g = x_HR_g.flatten(),y_HR_g.flatten()
    xy_HR_g_r = np.concatenate([x_HR_g[:,None],y_HR_g[:,None]],axis=1)

    depth_HR = interp.griddata(xy_g_r,source_r,xy_HR_g_r,method="cubic")
    depth_HR_nearest = interp.griddata(xy_g_r,source_r,xy_HR_g_r,method="nearest")

    depth_HR[np.isnan(depth_HR)] = depth_HR_nearest[np.isnan(depth_HR)]

    depth_HR = depth_HR.reshape(source_size*scaling_factor,-1)

    return depth_HR


def bicubic_with_mask_sparse(source,mask):
    
    source_size = source.shape[0]

    H,W = source.shape[0], source.shape[1]

    source_r = source.flatten()
    mask_r = mask.flatten()

    ### this part has been fixed to improve bicubic performance, nefore was not perfectly aligned
    x = np.arange(0,H) #(0,source.shape[0])
    y = np.arange(0,W) #(0,source.shape[1])

    x_g,y_g = np.meshgrid(x,y) # indexing="ij"

    x_g_r = x_g.flatten()
    y_g_r = y_g.flatten()

    source_r = source_r[mask_r==1]
    x_g_r = x_g_r[mask_r==1]
    y_g_r = y_g_r[mask_r==1]

    xy_g_r = np.concatenate([x_g_r[:,None],y_g_r[:,None]],axis=1)

    # second part being fixed
    x_HR = np.linspace(0,W,endpoint=False,num=W)
    y_HR = np.linspace(0,H,endpoint=False,num=H)

    x_HR_g,y_HR_g = np.meshgrid(x_HR,y_HR)
    x_HR_g,y_HR_g = x_HR_g.flatten(),y_HR_g.flatten()
    xy_HR_g_r = np.concatenate([x_HR_g[:,None],y_HR_g[:,None]],axis=1)

    depth_HR = interp.griddata(xy_g_r,source_r,xy_HR_g_r,method="cubic")
    depth_HR_nearest = interp.griddata(xy_g_r,source_r,xy_HR_g_r,method="nearest")

    depth_HR[np.isnan(depth_HR)] = depth_HR_nearest[np.isnan(depth_HR)]

    depth_HR = depth_HR.reshape(source_size,-1)

    return depth_HR


RGB_TO_YUV = np.array([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312]])
YUV_TO_RGB = np.array([
    [1.0, 0.0, 1.402],
    [1.0, -0.34414, -0.71414],
    [1.0, 1.772, 0.0]])
YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)


def rgb2yuv(im):
    return np.tensordot(im, RGB_TO_YUV, ([2], [1])) + YUV_OFFSET


def yuv2rgb(im):
    return np.tensordot(im.astype(float) - YUV_OFFSET, YUV_TO_RGB, ([2], [1]))



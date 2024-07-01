import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import rfft, irfft, fft2, ifft2, fftshift, ifftshift
lenna = np.load("lenna.npy")
# watermark = np.load("watermark.npy")
watermark = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
watermark = np.column_stack([watermark for _ in range(512//4)])
watermark = np.row_stack([watermark for _ in range(512//4)])
watermark = watermark * 128
DIM = 512

def plot_pic(pic):
    plt.gray()
    plt.imshow(pic)
    plt.show()

def high_pass_filter(data,axis=1,freq=20):
    mask = np.ones_like(data)
    if axis == 1:
        mask[:,:freq] = 0.0
    else:
        mask[:freq,:] = 0.0
    return (mask * data)
def low_pass_filter(data,axis=1,freq=20):
    mask = np.ones_like(data)
    if axis == 1:
        mask[:,freq:] = 0.0
    else:
        mask[freq:,:] = 0.0
    return (mask * data)

def row_fft(data):
    return rfft(data, axis=1)
def row_ifft(data):
    return irfft(data, axis=1)
def col_fft(data):
    return rfft(data, axis=0)
def col_ifft(data):
    return irfft(data, axis=0)

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

if __name__ == "__main__":
    # plot_pic(np.abs(row_ifft(high_pass_filter(row_fft(lenna)))))
    # plot_pic(np.abs(row_ifft(low_pass_filter(row_fft(lenna)))))
    # plot_pic(np.abs(col_ifft(high_pass_filter(col_fft(lenna), axis=0))))
    # plot_pic(np.abs(col_ifft(low_pass_filter(col_fft(lenna), axis=0))))
    mask1 = create_circular_mask(512,512,(128,128),30)
    mask2 = create_circular_mask(512,512,(384,384),30)
    mask = mask1 | mask2
    plot_pic(watermark)
    watermarked = (watermark / 255) * watermark + (1 - watermark / 255) * lenna
    plot_pic(watermarked)
    data = fftshift(fft2(watermarked, norm="ortho"))
    plot_pic(np.log(np.abs(data)))
    mu = np.median(np.log(np.abs(data)))
    std = np.std(np.log(np.abs(data)))
    print(f"{mu=}")
    print(f"{std=}")
    data = np.exp(mask * ((np.random.uniform(-1,1,data.shape) * std) + mu)) + (1 - mask) * data
    plot_pic(np.log(np.abs(data)))
    new_lenna = np.abs(ifft2(ifftshift(data), norm="ortho"))
    original_fft = fftshift(fft2(lenna, norm="ortho"))
    plot_pic(np.log(np.abs(original_fft)))

    plot_pic(new_lenna)
    plot_pic(lenna)

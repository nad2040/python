import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import rfft, irfft, fft2, ifft2
lenna = np.load("lenna.npy")
watermark = np.load("watermark.npy") / 255
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

if __name__ == "__main__":
    # plot_pic(np.abs(row_ifft(high_pass_filter(row_fft(lenna)))))
    # plot_pic(np.abs(row_ifft(low_pass_filter(row_fft(lenna)))))
    # plot_pic(np.abs(col_ifft(high_pass_filter(col_fft(lenna), axis=0))))
    # plot_pic(np.abs(col_ifft(low_pass_filter(col_fft(lenna), axis=0))))
    print(lenna.shape)
    watermark = np.column_stack([watermark,np.fliplr(watermark)])
    watermark = np.row_stack([watermark,np.flipud(watermark)])
    watermark = watermark / 10 + 0.95
    print(watermark.shape)
    data = np.array(fft2(lenna))
    print(data.shape)
    plot_pic(np.log(np.abs(data)))
    watermarked = watermark * data
    plot_pic(np.log(np.abs(watermarked)))
    new_lenna = np.abs(ifft2(watermarked))
    plot_pic(lenna)
    plot_pic(new_lenna)
    plot_pic(new_lenna - lenna)

    plot_pic(np.abs(fft2(new_lenna) / fft2(lenna)))



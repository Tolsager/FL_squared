import matplotlib.pyplot as plt

def plot_simsiam_images(dataset, index):
    im1, im2, label = dataset[index]
    plot_two_images_side_by_side(im1, im2)



def plot_two_images_side_by_side(im1, im2):
    fig, axs = plt.subplots(1, 2)
    im1 = im1.permute(1, 2, 0).numpy()
    im2 = im2.permute(1, 2, 0).numpy()
    axs[0].imshow(im1, vmin=im1.min(), vmax=im1.max())
    axs[1].imshow(im2, vmin=im2.min(), vmax=im2.max())
    plt.show()
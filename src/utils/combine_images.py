from PIL import Image

if __name__ == "__main__":
    # Open each image
    fc_image = Image.open("src/GAN_FC/final_images/ls_interpolation.png")
    wgan_image = Image.open("src/WGAN_OLD/final_images/ls_interpolation.png")
    acgan_image = Image.open("src/ACGAN_BEST/final_images/ls_interpolation_horse.png")
    dcgan_image = Image.open("src/GAN_EXT/final_images/ls_interpolation_v2.png")

    # Resize images if necessary to ensure they have the same dimensions
    width, height = min(im.size for im in [fc_image, wgan_image, acgan_image, dcgan_image])
    fc_image = fc_image.resize((width, height))
    wgan_image = wgan_image.resize((width, height))
    acgan_image = acgan_image.resize((width, height))
    dcgan_image = dcgan_image.resize((width, height))

    # Create a new blank image with the same dimensions as the input images
    stacked_image = Image.new("RGB", (width, height * 4))

    # Paste each image onto the blank image
    stacked_image.paste(fc_image, (0, 0))
    stacked_image.paste(wgan_image, (0, height))
    stacked_image.paste(acgan_image, (0, height * 2))
    stacked_image.paste(dcgan_image, (0, height * 3))

    # Save the stacked image
    stacked_image.save("src/ls_interpolation_images.png")

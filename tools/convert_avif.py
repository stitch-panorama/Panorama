from PIL import Image
import pillow_avif  # enables AVIF support in Pillow

im = Image.open("data/synthetic/source.avif").convert("RGB")
print(im.size)
im.save("data/synthetic/source.jpg", quality=95)
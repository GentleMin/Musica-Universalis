# -*- coding: utf-8 -*-


from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()

image = Image.open(args.file)
image.show()



from PIL import Image
import math
import os


class image_splitter_merger():
    def __init__(self,img_split_size, buffer= 50):
        self.out_w = img_split_size[0]
        self.out_h = img_split_size[1]
        self.buf = int(buffer)
        self.buf_hf = math.ceil(buffer/2.0)
        self.subsection_list = []

    def image_splitter(self, img):
        try:
            self.in_w, self.in_h = img.size
        except TypeError as e:
            print("please provide PIL.Image type input")
            raise e
            return 1

        #determine the number of sub images for each dimension
        self.N_w = math.ceil((self.in_w - self.buf) / float(self.out_w - self.buf))
        self.N_h = math.ceil((self.in_h - self.buf) / float(self.out_h - self.buf))

        for i in range(self.N_h * self.N_w):
            bbox = self._index2bbox(i)
            subsection = img.crop(bbox)
            self.subsection_list.append(subsection)
        return self.subsection_list


    def image_merger(self, image_list):
        new_im = Image.new('RGB', (self.in_w, self.in_h))
        index = 0
        for image in image_list:

            offset = self._index2imageoffset(index)

            offset = self._substract_half_buffer(offset)

            bbox = self._index2bbox_half_buffer_on_subimage(index)

            new_im.paste(image.crop(bbox), offset)


            index += 1

        return new_im

    def _index2bbox(self, i):
        n_w, n_h = self._index2subposition(i)

        left = n_w * (self.out_w - self.buf)
        right = left + self.out_w
        upper = n_h *(self.out_h - self.buf)
        lower = upper + self.out_h
        bbox = (left, upper, right, lower)
        return bbox

    def _index2subposition(self,i):
        n_h = math.floor(i / float(self.N_w))
        n_w = i % self.N_w
        return (n_w, n_h)

    def _index2imageoffset(self, i):
        n_w, n_h = self._index2subposition(i)

        w_offset = n_w * (self.out_w - self.buf)
        h_offset = n_h * (self.out_h - self.buf)

        return (w_offset, h_offset)

    def _substract_half_buffer(self, offset):
        off_w = 0 if offset[0] == 0 else offset[0] + self.buf_hf
        off_h = 0 if offset[1] == 0 else offset[1] + self.buf_hf
        return (off_w, off_h)

    def _index2bbox_half_buffer_on_subimage(self, i):
        n_w, n_h = self._index2subposition(i)

        left  = 0 if n_w == 0         else self.buf_hf
        right = self.out_w if n_w == self.N_w  else self.out_w - self.buf_hf
        upper = 0 if n_h == 0         else self.buf_hf
        lower = self.out_h if n_h == self.N_h  else self.out_h - self.buf_hf

        bbox = (left, upper, right, lower)
        return bbox

    def _toPILImage(self, array):
        return Image.fromarray(array.astype('uint8'), 'RGB')






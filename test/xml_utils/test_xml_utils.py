from experiments.exp_utils import ImageScribble
import unittest
import numpy as np
from PIL import Image, ImageDraw
import random
from pathlib import Path

class TestImageScribble(unittest.TestCase):
    def setUp(self):
        self.xml_file_path = 'test/xml_utils/test_2008_003147.xml'
        self.image_scribble = ImageScribble(self.xml_file_path)


    def test_image_scribble(self):
        self.assertEqual(self.image_scribble.filename, 'test_2008_003147.jpg')
        self.assertEqual(self.image_scribble.width, 500)
        self.assertEqual(self.image_scribble.height, 375)
        self.assertEqual(self.image_scribble.depth, 4)

        self.assertEqual(len(self.image_scribble.scribble_coordinates.keys()), 11)
        self.assertEqual(set(self.image_scribble.scribble_coordinates.keys()), set(self.image_scribble.scribble_mask.keys()))
        self.assertEqual(set(self.image_scribble.scribble_coordinates.keys()),
                         set(['sheep_0', 'sheep_1', 'sheep_2',
                              'grass_0',
                              'fence_0',
                              'tree_0', 'tree_1',
                              'sky_0', 'sky_1', 'sky_2', 'sky_3',])
                         )
        self.assertEqual(self.image_scribble.scribble_coordinates['sheep_0'].shape, (77, 2))
        self.assertTrue(np.array_equal(self.image_scribble.scribble_coordinates['sheep_0'][0], np.array([127, 209])),
                        f"Expected [127, 209], got {self.image_scribble.scribble_coordinates['sheep_0'][0]}")
        self.assertEqual(self.image_scribble.scribble_mask['sheep_0'].shape, (375, 500)) # height, width
        self.assertEqual(self.image_scribble.scribble_mask['sheep_0'][209][127], 1) # y, x

        return
    

    def test_image_scribble_draw(self):
        self.image_scribble.draw_scribble(object_name = 'sheep_0',
                                          image_save_path = 'test/xml_utils/sheep_0_scribble_only.jpg')
        self.image_scribble.draw_scribble(object_name = 'sheep_0',
                                          image_save_path = 'test/xml_utils/sheep_0_scribble_with_original.jpg',
                                          source_image_path = f'test/xml_utils/{self.image_scribble.filename}')

        return


if __name__ == '__main__':
    unittest.main()
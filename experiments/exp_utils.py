import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from datasets.pascal_utils import get_classname_plural
from utils.vis_utils import resize_and_pad

class ImageScribble:
    def __init__(self, xml_file_path: Union[str, Path], max_width=64, max_height=64, test_mode=False) -> None:
        assert Path(xml_file_path).exists(), f"File {xml_file_path} does not exist."
        assert Path(xml_file_path).suffix == ".xml", f"File {xml_file_path} is not an xml file."

        self.xml_file_path = xml_file_path
        self.max_width = max_width
        self.max_height = max_height

        if not test_mode:
            self._get_scribble()

        return


    def draw_scribble(self, 
                      object_name: Union[str, list],  
                      line_width=20, 
                      source_image_path=None
                    ) -> None:
        """
        Draw scribble of the selected objects.
        
        Args:
            object_name (str or list): Name of the object(s) to draw.
            image_save_path (str or Path): Path to save the image.
            line_width (int): Width of the line to draw.
            source_image_path (str or Path): Path to the original image. If None, a black image will be created.

        Returns:
            None
        """

        # Check arguments
        if isinstance(object_name, str):
            object_name = [object_name]
        for name in object_name:
            assert name in self.scribble_coordinates.keys(), f"Object {object_coordinate} not found."

        # Create image
        if source_image_path:
            assert Path(source_image_path).exists(), f"File {source_image_path} does not exist."
            image = Image.open(source_image_path)
            assert image.size == (self.width, self.height), f"Image size {image.size} does not match scribble size ({self.width}, {self.height})."
        else:
            image = Image.new('RGB', (self.width, self.height), color = 'black')
        image_draw = ImageDraw.Draw(image)

        for name in object_name:
            for object_coordinate in self.scribble_coordinates[name]:
                flat_coords_list = object_coordinate.reshape(-1).tolist()
                image_draw.line(flat_coords_list, fill = 'white', width = line_width) # TODO(minigb): Implement edge softening for the line.

        image = resize_and_pad(image, self.max_width)

        return image
        

    def _get_scribble(self) -> None:
        """
        Get scribble from xml file.
        """
        tree = ET.parse(self.xml_file_path)
        root = tree.getroot()
        
        # metadata
        self.filename = root.find('filename').text

        # image metadata
        self.width = int(root.find('size/width').text)
        self.height = int(root.find('size/height').text)
        self.depth = int(root.find('size/depth').text)
        self.segmented = int(root.find('segmented').text)

        # scribbles
        self.scribble_coordinates = {}
        self.scribble_mask = {}
        for polygon in root.findall('polygon'):
            object_name = polygon.find('tag').text
            #print('OBJ NAME :', object_name)
            
            # we do not consider multiple objects with the same name since we focus on semantic segmentation.
            # num_of_same_object = sum(1 for key in self.scribble_coordinates.keys() if object_name in key)
            # object_name = f'{object_name}_{num_of_same_object}'

            points = []

            if self.scribble_mask.get(object_name) is None:
                self.scribble_mask[object_name] = np.zeros((self.height, self.width))
            
            for point in polygon.findall('point'):
                x = int(point.find('X').text)
                y = int(point.find('Y').text)

                # ignore
                if not (0 <= x < self.width) or not (0 <= y < self.height):
                    continue
 
                assert 0 <= x < self.width, f"X coordinate must be between 0 and {self.width}. Got {x}."
                assert 0 <= y < self.height, f"Y coordinate must be between 0 and {self.height}. Got {y}."
                
                
                points.append([x, y])
                self.scribble_mask[object_name][y, x] = 1

            if self.scribble_coordinates.get(object_name) is None:
                self.scribble_coordinates[object_name] = [np.array(points)]
            else:
                self.scribble_coordinates[object_name].append(np.array(points))

        #print(f'scr_coords : {self.scribble_coordinates}')
         
        keys_to_delete = []

        for key in list(self.scribble_coordinates.keys()):
            if len(self.scribble_coordinates[key]) > 1:
                classname = get_classname_plural(key)
                if classname != key:
                    self.scribble_coordinates[classname] = self.scribble_coordinates[key]
                    keys_to_delete.append(key)

        for key in keys_to_delete:
            del self.scribble_coordinates[key]

        return
    

class ImageAnnotation:
    def __init__(self, xml_file_path: Union[str, Path], max_width=64, max_height=64, test_mode=False) -> None:
        assert Path(xml_file_path).exists(), f"File {xml_file_path} does not exist."
        assert Path(xml_file_path).suffix == ".xml", f"File {xml_file_path} is not an xml file."

        self.xml_file_path = xml_file_path
        self.max_width = max_width
        self.max_height = max_height

        if not test_mode:
            self._get_annotations()

    def draw_annotations(self, image_path: Union[str, Path], output_path: Union[str, Path] = None) -> Image.Image:
        """
        Draw bounding box annotations on the image.

        Args:
            image_path (str or Path): Path to the original image.
            output_path (str or Path): Path to save the output image. If None, the image won't be saved.

        Returns:
            Image.Image: PIL Image with bounding boxes overlay.
        """
        assert Path(image_path).exists(), f"File {image_path} does not exist."

        image = Image.open(image_path)

        # Draw bounding boxes on the image
        image_draw = ImageDraw.Draw(image)
        for annotation in self.annotations:
            xmin, ymin, xmax, ymax = annotation["bbox"]
            image_draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

        # Resize and pad the image
        image = self._resize_and_pad(image)

        # Save the image if output_path is provided
        if output_path:
            image.save(output_path)

        return image

    def _resize_and_pad(self, image: Image.Image) -> Image.Image:
        """
        Resize and pad the image to fit within the specified maximum width and height.

        Args:
            image (Image.Image): Original image.

        Returns:
            Image.Image: Resized and padded image.
        """
        return image.resize((self.max_width, self.max_height), Image.LANCZOS)

    def _get_annotations(self) -> None:
        """
        Get bounding box annotations from the XML file.
        """
        tree = ET.parse(self.xml_file_path)
        root = tree.getroot()

        self.annotations = []

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            annotation = {"bbox": (xmin, ymin, xmax, ymax)}
            self.annotations.append(annotation)

'''
# Example usage:
xml_file_path = "path/to/A.xml"
image_path = "path/to/original_image.jpg"
output_image_path = "path/to/output_image.jpg"

image_annotations = ImageAnnotations(xml_file_path)
output_image = image_annotations.draw_annotations(image_path, output_path=output_image_path)

# Display the image using matplotlib
plt.imshow(output_image)
plt.axis("off")
plt.show()
'''

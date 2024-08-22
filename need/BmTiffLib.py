import os

vipsbin = r'D:\work\python\vips-dev-8.15\bin'
os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']
import pyvips
import xml.etree.ElementTree as ET


def rotate_and_cinter_crop(im, angle):
    return im.rotate(angle).smartcrop(im.width, im.height, interesting="centre")

def read_pyramid_from_file(*args, **kwargs):
    return pyvips.Image.new_from_file(*args, **kwargs)


def draw_black(width, height, bands=1):
    return pyvips.Image.black(width, height, bands=bands)


def save_pyramid_tif(im, save_path, compression="jpeg",  tile_width=512, tile_height=512):
    # openslide will add an alpha ... drop it
    if im.hasalpha():
        im = im[:-1]

    image_height = im.height
    image_bands = im.bands

    # split to separate image planes and stack vertically ready for OME
    im = pyvips.Image.arrayjoin(im.bandsplit(), across=1)

    # set minimal OME metadata
    # before we can modify an image (set metadata in this case), we must take a
    # private copy
    im = im.copy()
    im.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    im.set_type(pyvips.GValue.gstr_type, "image-description",
                f"""<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
        <Image ID="Image:0">
            <!-- Minimum required fields about image dimensions -->
            <Pixels DimensionOrder="XYCZT"
                    ID="Pixels:0"
                    SizeC="{image_bands}"
                    SizeT="1"
                    SizeX="{im.width}"
                    SizeY="{image_height}"
                    SizeZ="1"
                    Type="uint8">
            </Pixels>
        </Image>
    </OME>""")

    im.tiffsave(save_path, compression=compression, tile=True,
                tile_width=tile_width, tile_height=tile_height,
                pyramid=True, properties=True, bigtiff=True)


def get_property_value(im, property_name):
    xml_string = im.get("image-description")
    return get_property_value_from_xml(xml_string, property_name)


def get_property_value_from_xml(xml_string, property_name):
    """
    从XML字符串中提取给定property名称的值。

    参数:
        xml_string (str): 包含XML数据的字符串。
        property_name (str): 要查询的property的名称。

    返回:
        str: 对应property的值。如果找不到，返回None。
    """
    # 移除命名空间声明
    cleaned_xml_string = xml_string.replace('xmlns="http://www.vips.ecs.soton.ac.uk//dzsave"', '')

    # 解析 XML 字符串
    root = ET.fromstring(cleaned_xml_string)

    # 找到所有 property 元素
    properties = root.find('properties')
    if properties is not None:
        for prop in properties.findall('property'):
            name = prop.find('name').text
            if name == property_name:
                return prop.find('value').text

    # 如果没有找到对应的property，返回None
    return None

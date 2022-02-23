import numpy as onp
import png
import io
import base64
from IPython.display import SVG
import math
import jax.numpy as np
from jax import lax, ops, jit, random, vmap

def tensorplot(data, labels, columns, pixels_per_cell=1):
  # todo: off-by-1 here somewhere, getting slight aliasing at pixels_per_cell=1
  data_urls = [get_data_url(i) for i in data]  
  label_height = 10
  array_height, array_width = data.shape[1] * pixels_per_cell,  data.shape[2] * pixels_per_cell 
  y_offset_multiplier = array_height + label_height + 2
  x_offset_multiplier = array_width + label_height + 2 # for symmetry
  index = onp.arange(0, data.shape[0])
  index_y = onp.floor(onp.divide(index, columns))
  index_x = onp.floor(onp.mod(index, columns))
  y_offsets = index_y * y_offset_multiplier
  x_offsets = index_x * x_offset_multiplier
  rows = math.ceil(data.shape[0] / columns)
  y_total = y_offset_multiplier * len(data) / columns
  x_total = x_offset_multiplier * columns
  
  svgs = [f'<svg width="{x_total}" height="{y_total}">']
  for (data_url, label, x_offset, y_offset) in zip(data_urls, labels, x_offsets, y_offsets):
    svgs.append(f'<g transform="translate({x_offset},{y_offset})">')
    svgs.append(svg_item(array_width, array_height, label_height, str(label), data_url))
    svgs.append('</g>')
  svgs.append("</svg>")  
  return "".join(svgs)


def svg_item(width, height, label_height, label, data_url):
  return f'''
    <rect x="1" y="{label_height}" width="{width}" height="{height}" style="stroke-width:1; stroke:rgb(0, 0, 0);"/>
    <image x="{1}" y="{label_height}" width="{width}" height="{height}" style="image-rendering:pixelated" href="{data_url}"/>
    <text style="fill:black" x="{width/2}" y="{label_height}" font-size="10" dominant-baseline="text-after-edge" text-anchor="middle">{label}</text>
  '''  

def array_plot(x, m=1):
  svg_str = "<svg>"+svg_image_item(x, m*x.shape[0],m*x.shape[1])+"</svg>"
  return SVG(svg_str)




def svg_image_item(data, width, height):
  data_url = get_data_url(data)
  return f'''
  <image width="{width}" height="{height}"
   style="image-rendering:pixelated" 
   href="{data_url}"
   />
  '''

def png_bytes(a):
  output = io.BytesIO()
  b = (1 - a)
  # todo: support for more colors
  png.from_array((b*255).astype(onp.uint8), 'L').write(output)
  return output.getvalue()

def get_data_url(x):
  return (b'data:image/png;base64,'+base64.b64encode(png_bytes(x))).decode("utf-8")  





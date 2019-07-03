import string
import pkg_resources

from matplotlib import font_manager
import numpy as np
from scipy import ndimage

from PIL import Image
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype


def rotate_points(xy, radians, origin=(0, 0)):
  """Rotate a point around a given point.
  """
  x, y = xy
  offset_x, offset_y = origin
  adjusted_x = (x - offset_x)
  adjusted_y = (y - offset_y)
  cos_rad = np.cos(radians)
  sin_rad = np.sin(radians)
  qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
  qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
  return np.array([qx, qy])


def elastic_transformations(image, alpha=2000, sigma=50, interpolation_order=1):
  """Returns a function to elastically transform multiple images.

  Good starting values:
    - alpha: 2000
    - sigma: between 40 and 60
  """
  # Take measurements
  image_shape = image.shape[:2]
  # Make random fields
  dx = np.random.uniform(-1, 1, image_shape) * alpha
  dy = np.random.uniform(-1, 1, image_shape) * alpha

  # Smooth dx and dy
  sdx = ndimage.filters.gaussian_filter(dx, sigma=sigma, mode='reflect')
  sdy = ndimage.filters.gaussian_filter(dy, sigma=sigma, mode='reflect')

  # Make meshgrid
  x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))

  # Distort meshgrid indices
  distorted_indices = (y + sdy).reshape(-1, 1), (x + sdx).reshape(-1, 1)

  # Map cooordinates from image to distorted index set
  def map_fn(a):
    return ndimage.interpolation.map_coordinates(a, distorted_indices,
                                                 mode='reflect',
                                                 order=interpolation_order)

  if image.ndim == 2:
    transformed_image = map_fn(image)
    transformed_image = transformed_image.reshape(image_shape)
  elif image.ndim == 3:
    transformed_image = [map_fn(im).reshape(image_shape) for im in np.moveaxis(image, -1, 0)]
    transformed_image = np.asarray(transformed_image)
    transformed_image = np.moveaxis(transformed_image, 0, -1)
  else:
    raise ValueError(f"Wrong image dimension: {image.shape}")
  return transformed_image


class CaptchaGenerator:

  def __init__(self, image_size, font_size=60, font_path=None, font_name=None, alphabet=None):

    self.image_size = image_size
    self.font_size = font_size

    if font_path:
      self.font_path = str(font_path)
    else:
      if not font_name:
        self.font_name = font_manager.fontManager.defaultFamily['ttf']
        self.font_path = font_manager.fontManager.findfont(self.font_name)
      else:
        self.font_name = font_name
        self.font_path = pkg_resources.resource_filename('captchanet', f'fonts/{self.font_name}')

    self.font_type = self._make_font_type(font_size)

    if not alphabet:
      self.alphabet = list(string.ascii_letters + string.digits)
      self.alphabet.remove('0')
      self.alphabet.remove('o')
      self.alphabet.remove('O')
    else:
      self.alphabet = alphabet

    self.table = [i * 19.7 for i in range(256)]

  def __call__(self, n_min=6, n_max=10, watermark=None, font_color=(140, 140, 140), background=(255, 255, 255, 0)):
    """Generate a captcha image for a random word of size `n`.
    """
    n = np.random.randint(n_min, n_max + 1)
    word = self.generate_word(n)
    return word, self.generate_image_from_word(word, watermark, font_color, background)

  def generate_image_from_word(self, word, watermark=None, font_color=(140, 140, 140), background=(255, 255, 255, 0)):

    image_width = self.image_size[0]
    image_height = self.image_size[1]

    image = Image.new('RGBA', self.image_size, background)
    draw = Draw(image)

    # Create patch for all letters.
    patches = []
    for letter in word:
      patches.append(self._draw_character(letter, draw, font_color, background))

    word_width = np.sum([im.size[0] for im in patches])
    width_offset = int((image_width - word_width) / 2)

    for patch in patches:
      patch_width, patch_height = patch.size
      mask = patch.convert('L').point(self.table)
      image.paste(patch, (width_offset, int((image_height - patch_height) / 2)), mask)
      width_offset = width_offset + patch_width

    if word_width > image_width:
      image = image.resize((image_width, image_height))

    image = np.asarray(image)
    image = elastic_transformations(image, alpha=1200, sigma=40)
    image = Image.fromarray(image)

    color = (112, 112, 112)
    self._create_noise_dots(image, color=color, n_min=450, n_max=500)
    self._create_noise_curves(image, color=color, width_min=1, width_max=3, n_min=8, n_max=12)

    if watermark:
      self._add_watermark(image, watermark, font_color=(112, 112, 112), font_size=13)

    image = image.convert('RGB')
    return image

  def _make_font_type(self, font_size):
    return truetype(self.font_path, font_size)

  def _draw_character(self, character, draw, font_color=(140, 140, 140), background=(255, 255, 255, 0)):

    patch_width, patch_height = draw.textsize(character, font=self.font_type)
    patch = Image.new('RGBA', (patch_width, patch_height), color=background)
    Draw(patch).text((0, 0), character, font=self.font_type, fill=font_color)
    return patch

  def generate_word(self, n):
    """Generate a random word of size `n` from a given alphabet.
    """
    letters = np.random.choice(self.alphabet, size=n)
    return ''.join(letters)

  def _create_noise_curves(self, image, color, width_min=1, width_max=4, n_min=7, n_max=10):
    draw = Draw(image)
    w, h = image.size
    n = np.random.randint(n_min, n_max)
    for _ in range(n):
      A = np.random.randint(0, h / 2)
      f = 1 / np.random.randint(1, 20)
      A *= 0.1

      x = np.arange(0, w, 1)
      size = np.random.randint(w * 0.2, 0.8 * w)
      x = x[:size]
      x += np.random.randint(0, w)
      y = A * np.sin(2*np.pi*f*x)
      y += np.random.randint(0, h / 2)
      points = np.array([x, y]).T

      # Random rotation
      degree = np.random.randint(0, 360)
      radian = np.deg2rad(degree)
      points = rotate_points(points.T, radian, origin=points.mean(axis=0))
      points = points.T
      points = [tuple(p) for p in points]

      width = np.random.randint(width_min, width_max)
      draw.line(points, width=width, fill=color)
    return image

  def _create_noise_dots(self, image, color, n_min=80, n_max=120):
    draw = Draw(image)
    w, h = image.size
    n = np.random.randint(n_min, n_max)
    for _ in range(n):
      x1 = np.random.randint(0, w)
      y1 = np.random.randint(0, h)
      draw.point((x1, y1), fill=color)

      if np.random.random() < 0.5:
        draw.point((x1 - 1, y1), fill=color)
      if np.random.random() < 0.5:
        draw.point((x1 - 1, y1 - 1), fill=color)
      if np.random.random() < 0.5:
        draw.point((x1, y1 - 1), fill=color)

    return image

  def _add_watermark(self, image, text, font_color=(140, 140, 140), font_size=12):
    font_type = self._make_font_type(font_size)
    draw = Draw(image)
    w, h = draw.textsize(text, font=font_type)

    dx = (image.width - w) - 4.5
    dy = image.height - h - 0
    draw.text((dx, dy), text, font=font_type, fill=font_color)

    return image

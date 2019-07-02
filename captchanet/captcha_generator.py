import string

from matplotlib import font_manager
import numpy as np

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


class CaptchaGenerator:

  def __init__(self, image_size, font_size=60, font_path=None, font_name=None, alphabet=None):

    self.image_size = image_size
    self.font_size = font_size

    if font_path:
      self.font_path = str(font_path)
    else:
      if not font_name:
        self.font_name = font_manager.fontManager.defaultFamily['ttf']
      else:
        self.font_name = font_name
      self.font_path = font_manager.fontManager.findfont(self.font_name)
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

    # average = int(word_width / len(word))
    #rand = int(0.05 * average)
    offset = int((image_width - word_width + 5 * len(patches)) / 2)

    for patch in patches:
      w, h = patch.size
      mask = patch.convert('L').point(self.table)
      image.paste(patch, (offset, int((image_height - h) / 2) - 10), mask)
      offset = offset + w - 5  # + np.random.randint(-rand, 0)

    if word_width > image_width:
      image = image.resize((image_width, image_height))

    color = (112, 112, 112)
    self._create_noise_dots(image, color=color, n_min=450, n_max=500)
    self._create_noise_curves(image, color=color, width_min=1, width_max=3, n_min=5, n_max=9)

    if watermark:
      self._add_watermark(image, watermark, font_color=(112, 112, 112), font_size=13)

    return image

  def _make_font_type(self, font_size):
    return truetype(self.font_path, font_size)

  def _draw_character(self, character, draw, font_color=(140, 140, 140), background=(255, 255, 255, 0)):
    w, h = draw.textsize(character, font=self.font_type)

    # Write and draw
    dx = np.random.randint(0, 2)
    dy = np.random.randint(0, 2)
    patch = Image.new('RGBA', (w + dx, h + dy), color=background)
    Draw(patch).text((dx, dy), character, font=self.font_type, fill=font_color)

    # Rotate
    patch = patch.crop(patch.getbbox())
    #patch = patch.rotate(np.random.uniform(-5, 5), Image.BILINEAR, expand=1)

    # Warp
    dx = w * np.random.uniform(0, 0.1)
    dy = h * np.random.uniform(0, 0.1)
    x1 = int(np.random.uniform(-dx, dx))
    y1 = int(np.random.uniform(-dy, dy))
    x2 = int(np.random.uniform(-dx, dx))
    y2 = int(np.random.uniform(-dy, dy))
    w2 = w + abs(x1) + abs(x2)
    h2 = h + abs(y1) + abs(y2)
    data = (x1, y1, -x1, h2 - y2, w2 + x2, h2 + y2, w2 - x2, -y1)
    patch = patch.resize((w2, h2))
    patch = patch.transform((w, h), Image.QUAD, data)

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

    dx = (image.width - w) - 2
    dy = image.height - h - 2
    draw.text((dx, dy), text, font=font_type, fill=font_color)

    return image

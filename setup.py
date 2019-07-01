from setuptools import setup
from setuptools import find_packages


setup(name='captchanet',
      version='0.1.0',
      author='Hadrien Mary',
      author_email='hadrien.mary@gmail.com',
      url='https://github.com/hadim/captchanet/',
      description='A simple but yet efficient neural networks to solve captcha images.',
      long_description_content_type='text/markdown',
      packages=find_packages(),
      classifiers=[
              'Development Status :: 5 - Production/Stable',
              'Intended Audience :: Developers',
              'Natural Language :: English',
              'License :: OSI Approved :: BSD License',
              'Operating System :: OS Independent',
              'Programming Language :: Python',
              'Programming Language :: Python :: 3',
              ])

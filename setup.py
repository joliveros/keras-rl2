from setuptools import setup
from setuptools import find_packages

setup(name='keras-rl2',
      version='1.0.4',
      description='Deep Reinforcement Learning for Tensorflow 2 Keras',
      author='Taylor McNally',
      author_email='taylor.mcnally@emory.edu',
      url='https://github.com/wau/keras-rl2',
      license='MIT',
      # install_requires=['exchange-data'],
      extras_require={
          'gym': ['gym'],
      },
      packages=find_packages())

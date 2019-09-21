from setuptools import setup


setup(name='ctlearn_optimizer',
      version='1.0.0',
      description='Optimization framework for CTLearn v0.3.0',
      url='https://github.com/ctlearn-project/ctlearn_optimizer',
      author='Juan Alfonso Redondo Pizarro',
      license='BSD-3-Clause',
      packages=['ctlearn_optimizer'],
      package_dir={'ctlearn_optimizer': 'src/ctlearn_optimizer'},
      include_package_data=True,
      zip_safe=False)

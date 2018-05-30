from setuptools import setup

setup(name='dias',
      version='0.01',
      description='Disaster Impact Analysis System (DIAS) provides a set of methods for modeling disaster and decision modeling using Q-Analysis and Multi-Criteria Decision Analysis',
      classifiers=[
          'Development Status :: 1 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Flood Modeling :: System Dynamics :: Q-Analysis :: Complex Systems :: Topology',
      ],
      url='https://github.com/dr-jgsmith/Disaster-Impact-Analysis-System',
      author='Justin G. Smith',
      author_email='justingriffis@wsu.edu',
      license='MIT',
      packages=['dias',
                'dias.core',
                'dias.storage',
                'dias.scripts',
                'dias.notebooks',
                'dias.visual'],
      install_requires=[
          'dataset',
          'stuf',
          'seaborn',
          'networkx',
          'lxml',
          'dbfread',
          'googlemaps',
          'numba'

      ],
      include_package_data=True,
      zip_safe=False)

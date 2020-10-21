from setuptools import setup, Extension, find_packages
from distutils.core import setup



from codecs import open
import os
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'kernelml-README.md'), encoding='utf-8') as f:
    long_description = f.read()


# setup(
#     name = "Kernel Machine Learning",
#     ext_modules = cythonize('kernelml.pyx'),  # accepts a glob pattern
# )

def find_pyx(path='.'):
    pyx_files = []
    for root, dirs, filenames in os.walk(path):
        for fname in filenames:
            if fname.endswith('.c'):
                pyx_files.append(os.path.join(root, fname))
    return pyx_files

setup(
      # Information
      name = "kernelml",
      version = "3.38",
      description='generalized machine learning algorithm for complex loss functions and non-linear coefficients',
      url = "https://github.com/Freedomtowin/kernelml",
      author = "Rohan Kotwani",
      email = "rohankotwani@gmail.com",
      
      
      license = "MIT",
      classifiers=[
                   "Development Status :: 4 - Beta",
                   
                   # Indicate who your project is intended for
                   "Intended Audience :: Developers",
                   "Intended Audience :: Science/Research",
                   "Topic :: Software Development",
                   "Topic :: Scientific/Engineering",
                   
                   # Pick your license as you wish
                   'License :: OSI Approved :: MIT License',
                   
                   # Specify the Python versions you support here. In particular, ensure
                   # that you indicate whether you support Python 2, Python 3 or both.
                   'Programming Language :: Python :: 3'
                   ],
      keywords = "kernel machine learning nonlinear",
      #    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
  packages=['kernelml','kernelml.kernelml_bycython',
            'kernelml.hdre','kernelml.hdre.hdr_helpers_bycython'
            ],  # Required
      install_requires = ["numpy","cython","numba"],
      python_requires = ">=3.6",
      #     data_files=[('my_data', ['kc_house_test_data.csv','kc_house_train_data.csv'])],
      #    ext_modules=cythonize(find_pyx('kernelml'), language_level=3),
      #    # Build instructions
      ext_modules = [Extension("kernelml.kernelml_bycython.kernelml",
                               find_pyx('kernelml/kernelml_bycython'),
                               include_dirs=[]),
                     Extension("kernelml.hdre.hdr_helpers_bycython.hdr_helpers",
                               find_pyx('kernelml/hdre/hdr_helpers_bycython'),
                               include_dirs=[])
                               ]
      )

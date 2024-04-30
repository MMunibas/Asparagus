Installation
===================================


Asparagus Bundle is a package that depends majorly on Pytorch and the Atomic Simulation Environment (ASE).
Other than that, it is designed to rely on just a few of packages and is written fully in Python without the need for compilation.
We recommend that you use a virtual environment to install Asparagus.

Dependencies
--------------

Asparagus depends on the following packages:

   .. Python: https://www.python.org/ :math:`\geq` 3.8 (**DO NOT** (for the moment) use 3.9)
   .. PyTorch :math:`\geq` 2.0
   .. Torch-ema :math:`\geq` 0.3
   .. TensorBoard :math:`\geq` 2.4
   .. Atomic Simulation Environment (ASE)  :math:`\geq` 3.22

Optional:

   .. _Python: https://www.python.org/


Setting up the environment
--------------------------

We recommend to use virtual environment sucha as `mamba`_ or `miniconda`_.

.. _mamba: https://mamba.readthedocs.io/en/latest/user_guide/mamba.html
.. _miniconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html

In `mamba`_, you can create a virtual enviroment called *asparagus* by

.. code-block:: bash

      mamba create --name asparagus python=3.11

and activate the virtual environment by the command:

.. code-block:: bash

       mamba activate asparagus

**Note**: If you are using Miniconda, just replace ``mamba`` by ``conda``.

Installation
-------------

To install Asparagus first clone the repository:

.. code-block:: bash

      git clone https://github.com/LIVazquezS/Asparagus

Then, go to the folder where you cloned the repository and install via pip:

.. code-block:: bash

      pip install -e .

Alternatively, install via ``setup.py``:

.. code-block:: bash

      python setup.py install

**BEWARE**: With this command any modification that is done to the code in the folder *asparagus* will be automatically reflected
in the modules that you import.



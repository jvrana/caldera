Caldera
=======

.. toctree::
   :maxdepth: 2
   :glob:
   :hidden:

   getting_started
   api
   examples/examples
   gallery/gallery
   narratives/narratives.rst
   developer

.. toctree::
   :maxdepth: 1
   :caption: API

    data
    dataset
    blocks
    models
    transforms
    utils
    defaults
    exceptions

.. jupyter-execute::

  import caldera
  from caldera.data import GraphData

  data = GraphData.random(5, 4, 3)
  print(data)
  name = 'world'
  print('hello ' + name + '!')
  data

.. automodule:: caldera.data
    :members:



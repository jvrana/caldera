Caldera
=======

.. toctree::
   :name: NavBar
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
   :caption: Python API
   :maxdepth: 1
   :glob:

   api/gnn
   api/data
   api/dataset
   api/blocks
   api/models
   api/transforms
   api/utils
   api/defaults
   api/exceptions

.. jupyter-execute::

  import caldera
  from caldera.data import GraphData

  data = GraphData.random(5, 4, 3)
  print(data)
  name = 'world'
  print('hello ' + name + '!')
  data
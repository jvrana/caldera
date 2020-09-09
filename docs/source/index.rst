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
   narratives/data-processing
   narratives/shortest_path.rst
   developer

.. jupyter-execute::

  import caldera
  from caldera.data import GraphData

  data = GraphData.random(5, 4, 3)
  print(data)
  name = 'world'
  print('hello ' + name + '!')
  data
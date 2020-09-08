Caldera
=======

.. toctree::
   :maxdepth: 2
   :glob:
   :hidden:

   getting_started
   api
   examples/examples
   narratives/data-processing
   _nb_generated/data-processing

.. jupyter-execute::

  import caldera
  from caldera.data import GraphData

  data = GraphData.random(5, 4, 3)
  print(data)
  name = 'world'
  print('hello ' + name + '!')
  data
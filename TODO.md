# TODO

1. Summary statistics for GraphDataLoader
1. Summary for GraphData (in degree, out degree, etc.)
1. t comGlobal information is not being passed correctly. It should be a global array per GRAPH not per batch. This needs
to be corrected in the global block (and core?). We should be selecting values from some `global_idx` value that indicates the index of
the individual graphs.
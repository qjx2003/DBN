from pgmpy.models import DynamicBayesianNetwork as DBN
dbn = DBN()
dbn.add_edges_from([(('D', 0), ('G', 0)), (('I', 0), ('G', 0)),
                    (('G', 0), ('L', 0)), (('D', 0), ('D', 1)),
                    (('I', 0), ('I', 1)), (('G', 0), ('G', 1)),
                    (('G', 0), ('L', 1)), (('L', 0), ('L', 1))])
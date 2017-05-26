import tensorflow as tf


class AbstractModel(object):
    def __init__(self, model_path):
        if self.__class__ is AbstractModel:
            raise TypeError('Abstract class cannot be instantiated.')
        self.inception_model_path = model_path
        try:
            print("Import graph")
            self.graph = self._get_graph()
            print("Graph loaded.")
        except:
            print('Error while loading model.')
            exit()

    def _get_graph(self):
        # read model file
        with tf.gfile.GFile(self.inception_model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # import a graph_def into the current default Graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="",
                op_dict=None,
                producer_op_list=None
            )
        return graph

    def extract_embeddings(self, augmentation_flag=False):
        raise TypeError('Abstract method extract() must be overridden.')

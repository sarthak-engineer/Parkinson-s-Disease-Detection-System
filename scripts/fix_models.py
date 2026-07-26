import os
import joblib
import joblib.numpy_pickle
import numpy as np
import pickle

class CustomUnpickler(joblib.numpy_pickle.NumpyUnpickler):
    def dispatch_build(self):
        # Peak at the stack to see what object is being built
        state = self.stack.pop()
        obj = self.stack[-1]
        
        # We need to detect sklearn.tree._tree.Tree
        if type(obj).__name__ == 'Tree':
            if isinstance(state, dict) and 'nodes' in state:
                nodes = state['nodes']
                if hasattr(nodes, 'dtype') and 'missing_go_to_left' not in nodes.dtype.names:
                    # Create new dtype
                    new_dtype = np.dtype([
                        ('left_child', '<i8'),
                        ('right_child', '<i8'),
                        ('feature', '<i8'),
                        ('threshold', '<f8'),
                        ('impurity', '<f8'),
                        ('n_node_samples', '<i8'),
                        ('weighted_n_node_samples', '<f8'),
                        ('missing_go_to_left', 'u1')
                    ])
                    new_nodes = np.zeros(nodes.shape, dtype=new_dtype)
                    for name in nodes.dtype.names:
                        new_nodes[name] = nodes[name]
                    new_nodes['missing_go_to_left'] = 0
                    state['nodes'] = new_nodes
        
        self.stack.append(state)
        # Call the original build handler (which is BUILD in Python's core Unpickler)
        super().dispatch[pickle.BUILD](self)

# We need to monkey-patch the dispatch table for our unpickler
CustomUnpickler.dispatch = joblib.numpy_pickle.NumpyUnpickler.dispatch.copy()
CustomUnpickler.dispatch[pickle.BUILD] = CustomUnpickler.dispatch_build

def load_and_save(filename):
    print(f"Fixing {filename}...")
    with open(filename, 'rb') as f:
        unpickler = CustomUnpickler(f)
        obj = unpickler.load()
    
    with open(filename, 'wb') as f:
        joblib.dump(obj, f)
    print(f"Saved {filename} successfully.")

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
    models = [
        os.path.join(PROJECT_ROOT, "backend", "models", "drawing_model.pkl"),
        os.path.join(PROJECT_ROOT, "backend", "models", "voice_modell.pkl"),
        os.path.join(PROJECT_ROOT, "backend", "models", "Rf_edit.pkl")
    ]
    for model in models:
        try:
            load_and_save(model)
        except Exception as e:
            print(f"Error processing {model}: {e}")

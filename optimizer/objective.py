import numpy as np

class Objective():
    def __init__(self, objective_functions, num_objectives=None) -> None:
        self.objective_functions = objective_functions
        if num_objectives is None:
            self.num_objectives = len(self.objective_functions)
        else:
            self.num_objectives = num_objectives
        pass

    def evaluate(self, items, mask):
        outputs = np.array([objective_function(items[mask]) for objective_function in self.objective_functions]).T
        return np.array(self.populate_matrix(outputs, mask))
    
    def populate_matrix(self, outputs, mask):
        result = []
        output_id = 0
        for m in mask:
            if m :
                result.append(outputs[output_id])
                output_id += 1
            else:
                result.append([np.inf] * self.num_objectives) # Maybe len(self.objective_function) depending on how self.num_objectives is defined
        return result

    def type(self):
        return self.__class__.__name__


class ElementWiseObjective(Objective):
    def __init__(self, objective_functions, num_objectives=None) -> None:
        super().__init__(objective_functions, num_objectives)

    def evaluate(self, items, mask):
        outputs = [[obj_func(item) for obj_func in self.objective_functions] for item in items[mask]]
        return np.array(self.populate_matrix(outputs, mask))

class BatchObjective(Objective):
    def __init__(self, objective_functions, num_objectives=None) -> None:
        super().__init__(objective_functions, num_objectives)

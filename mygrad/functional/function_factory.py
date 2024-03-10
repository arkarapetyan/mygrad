# functional/function_factory.py
import mygrad.functional as F


class __SingletonMeta(type):
    __instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            instance = super().__call__(*args, **kwargs)
            cls.__instances[cls] = instance
        return cls.__instances[cls]


class FunctionFactory(metaclass=__SingletonMeta):
    __active_instances = {}
    __free_instances = {}

    def get_new_function_of_type(self, f_type):
        if len(self.__free_instances) != 0:
            name, func = self.__free_instances[f_type].popitem()
        else:
            name, func = self.__create_function_of_type(f_type)

        if f_type not in self.__active_instances:
            self.__active_instances[f_type] = {}

        self.__active_instances[f_type][name] = func

        return func

    def get_active_function_by_name(self, name):
        func = None
        for t in self.__active_instances.keys():
            if name not in self.__active_instances[t]:
                continue

            func = self.__active_instances[t][name]

        if func is None:
            print(f"Active Function not Found: {name}")

        return func

    def __create_function_of_type(self, f_type):
        num = 0
        if f_type in self.__active_instances.keys():
            num += len(self.__active_instances[f_type])
        if f_type in self.__free_instances.keys():
            num += len(self.__free_instances[f_type])

        print(f"Creating Function of type {f_type} with name", end=" ")
        if f_type == F.Add:
            name = f"add_{num}"
            print(name)
            return name, F.Add(name)
        elif f_type == F.Mul:
            name = f"mul_{num}"
            print(name)
            return name, F.Mul(name)
        elif f_type == F.Matmul:
            name = f"matmul_{num}"
            print(name)
            return name, F.Matmul(name)
        elif f_type == F.Exp:
            name = f"exp_{num}"
            print(name)
            return name, F.Exp(name)
        elif f_type == F.Sigmoid:
            name = f"sigmoid_{num}"
            print(name)
            return name, F.Sigmoid(name)
        elif f_type == F.Tanh:
            name = f"tanh_{num}"
            print(name)
            return name, F.Tanh(name)
        elif f_type == F.ReLU:
            name = f"relu_{num}"
            print(name)
            return name, F.ReLU(name)
        elif f_type == F.Linear:
            name = f"linear_{num}"
            print(name)
            return name, F.Linear(name)
        elif f_type == F.BCELossWithLogits:
            name = f"bce_loss_logit_{num}"
            print(name)
            return name, F.BCELossWithLogits(name)
        elif f_type == F.MSELoss:
            name = f"mse_loss_{num}"
            print(name)
            return name, F.MSELoss(name)

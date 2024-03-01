import lightning
import torch


class ParameterGradientLogger(lightning.pytorch.callbacks.Callback):

        def __init__(self, pl_module):
            self.parameter_names = set()
            stack = [("gpt", pl_module.model)]
            while len(stack) > 0:
                module_name, module = stack.pop()
                assert not isinstance(module, torch.nn.Parameter)
                assert len(list(module.children())) == len(list(module.named_children()))
                for child_name, child in module.named_children():
                    stack.append((f"{module_name}-{child_name}", child))
                assert len(list(module.parameters())) == len(list(module.named_parameters()))
                for parameter_name, parameter in module.named_parameters(recurse=False):
                    if not parameter.requires_grad:
                        continue
                    parameter_name = f"{module_name}-{parameter_name}"
                    assert not parameter_name in self.parameter_names
                    self.parameter_names.add(parameter_name)
        
        def on_after_backward(self, trainer, pl_module):
            stack = [("gpt", pl_module.model)]
            while len(stack) > 0:
                module_name, module = stack.pop()
                assert not isinstance(module, torch.nn.Parameter)
                assert len(list(module.children())) == len(list(module.named_children()))
                for child_name, child in module.named_children():
                    stack.append((f"{module_name}-{child_name}", child))
                assert len(list(module.parameters())) == len(list(module.named_parameters()))
                for parameter_name, parameter in module.named_parameters(recurse=False):
                    if not parameter.requires_grad:
                        continue
                    parameter_name = f"{module_name}-{parameter_name}"
                    assert parameter_name in self.parameter_names
                    pl_module.log(f"{parameter_name}-mean", parameter.mean())
                    pl_module.log(f"{parameter_name}-abs", parameter.abs().mean())
                    pl_module.log(f"{parameter_name}-grad-mean", parameter.grad.mean())
                    pl_module.log(f"{parameter_name}-grad-abs", parameter.grad.abs().mean())
                    # pl_module.log(f"{parameter_name}-max", parameter.max())
                    # pl_module.log(f"{parameter_name}-min", parameter.min())

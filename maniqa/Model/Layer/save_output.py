class SaveOutput:
    '''forward hook：按顺序收集每个被挂载模块的输出，供后续按层取特征。'''

    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

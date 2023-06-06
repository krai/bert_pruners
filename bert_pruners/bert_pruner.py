from transformers import BertModel
from torch.nn.utils import prune

class BertPruner:
    def __init__(self, saved_dir: str, sparsity: float):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.saved_dir = saved_dir
        self.sparsity = sparsity

    def prune_and_save(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.sparsity)

        torch.onnx.export(self.model,               # model being run
                          torch.randn(1, 512),      # model input (or a tuple for multiple inputs)
                          self.saved_dir,           # where to save the model
                          export_params=True,       # store the trained parameter weights inside the model file
                          opset_version=10,         # the ONNX version to export the model to
                          do_constant_folding=True) # whether to execute constant folding for optimization

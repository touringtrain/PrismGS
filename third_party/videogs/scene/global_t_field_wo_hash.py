import torch
import tinycudann as tcnn
import numpy as np
import json

class GlobalTField_wo_hash(torch.nn.Module):
    def __init__(self, config_path = "config/config_wo_hash.json") -> None:
        super().__init__()

        self.config_path = config_path
        with open(self.config_path) as config_file:
            self.config = json.load(config_file)
        config_file.close()

        self.model = tcnn.Network(
            n_input_dims=3, 
            n_output_dims=3,
            network_config=self.config["network"]
        ).to("cuda")

    def forward(self, pcd):
        deform_params = self.model(pcd)
        global_translation = deform_params
        # global_quaternion = deform_params[:, 3:]
        return global_translation.float()
    
    def dump_ckpt(self, output_path):
        print(f"Saving model to {output_path}")
        torch.save(self.model.state_dict(), output_path)

    def load_ckpt(self, input_path):
        print(f"Loading model from {input_path}")
        self.model.load_state_dict(torch.load(input_path))
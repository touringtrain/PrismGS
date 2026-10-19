import torch
import tinycudann as tcnn
import numpy as np
import json

class GlobalRTCField(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.config_path = "config/config_hash.json"
        with open(self.config_path) as config_file:
            self.config = json.load(config_file)
        config_file.close()

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3, 
            n_output_dims=3 + 4 + 3,
            encoding_config=self.config["encoding"],
            network_config=self.config["network"]
        ).to("cuda")

    def forward(self, pcd):
        deform_params = self.model(pcd)
        global_translation = deform_params[:, :3]
        global_quaternion = deform_params[:, 3:7]
        global_scale = deform_params[:, 7:]
        # return global_translation, global_quaternion
        # float16 to float32
        return global_translation.float(), global_quaternion.float(), global_scale.float()
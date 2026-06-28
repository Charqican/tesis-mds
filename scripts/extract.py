from data_loaders.SimplePointCloudLoader import PointCloudLoader
from data_loaders.config import DataLoaderConfig
from feature_extractor.config import FeatureConfig
from pipelines.backprojected_features import extract_features
from pytorch3d.structures import Pointclouds
import torch
import torchvision
from model_wrappers import DINOWrapper
from dotenv import load_dotenv
import os


def main():
    load_dotenv()
    raw_data_path = os.environ["DATA_SYM_PLANE_PROCESSED"]
    features_symm_ori_path = Path(os.environ["FEATURES_SYM_ORI"])

    # load data
    dataload_settings = DataLoaderConfig(processed_dir=raw_data_path, batch_size=1)
    feature_settings = FeatureConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = PointCloudLoader(dataload_settings, sort=False)
    model = DINOWrapper(device=device, small=True, reg=True)

    for name, point_cloud in dataloader:
        point_cloud = point_cloud.to(device)
        pc_features: torch.Tensor = extract_features(
            point_cloud, model, feature_settings
        )
        np.save(features_symm_ori_path, pc_features.cpu().numpy())

import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_path",type=str,default="colmap")
    parser.add_argument("--data_path",type=str,default="/home/auwang/workspace/data/teaser_datasets/hanfu_cali/0")
    args = parser.parse_args()

    colmap_path = args.colmap_path
    data_path = args.data_path

    database_path = os.path.join(data_path,"database.db")
    image_path = os.path.join(data_path,"images")
    sparse_path = os.path.join(data_path,"sparse")
    dense_path = os.path.join(data_path,"dense")
    fused_path = os.path.join(dense_path,"fused.ply")
    os.makedirs(sparse_path,exist_ok=True)
    os.makedirs(dense_path,exist_ok=True)
    
    feature_extraction = "{} feature_extractor --database_path {} --image_path {}".format(colmap_path,database_path,image_path)
    exhaustive_matcher = "{} exhaustive_matcher --database_path {}".format(colmap_path,database_path)
    mapper = "{} mapper --database_path {} --image_path {} --output_path {}".format(colmap_path,database_path,image_path,sparse_path)
    patch_match_stereo = "{} patch_match_stereo --workspace_path {} --workspace_format COLMAP --PatchMatchStereo.geom_consistency true".format(colmap_path,dense_path)
    stereo_fusion = "{} stereo_fusion --workspace_path {} --workspace_format COLMAP --input_type geometric --output_path {}".format(colmap_path,dense_path,fused_path)

    os.system(feature_extraction)
    os.system(exhaustive_matcher)
    os.system(mapper)
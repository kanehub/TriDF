import argparse
import os
import shutil
from datetime import datetime

import pycolmap
import subprocess

from levir2colmap import generate_init_txt
from read_write_model import read_cameras_binary, read_images_binary, read_points3D_binary
from read_write_model import write_cameras_text, write_images_text, write_points3D_text
from set_database import camTodatabase

neighbor_view_ids = ["000", "001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012",
                     "013", "014", "015", "016", "017", "018", "019", "020"]


def main(args):
    ymd = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    work_dir = os.path.join("./runs", args.scene_id, ymd)
    os.makedirs(work_dir, exist_ok=True)

    # 拷贝图片
    img_dir = os.path.join(work_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    for idx in args.train_list:
        src_img_path = os.path.join(args.dataset_path, args.scene_id, "Images/{}.png".format(neighbor_view_ids[idx]))
        shutil.copy(src_img_path, os.path.join(img_dir, "{}.png".format(neighbor_view_ids[idx])))

    # feature extraction
    database_path = os.path.join(work_dir, "database.db")
    # pycolmap.extract_features(database_path, img_dir)
    colmap_extract_features = [
        'COLMAP.bat', 'feature_extractor',
        '--database_path', database_path,
        '--image_path', img_dir,
        '--ImageReader.single_camera', '1',
        '--SiftExtraction.dsp_min_scale', '0.16667',
        '--SiftExtraction.peak_threshold', '0.00668'
    ]
    print(f"Running command: {' '.join(colmap_extract_features)}")
    result_feature_extractor = subprocess.run(colmap_extract_features, capture_output=True, text=True)
    print(f"Command stderr:\n{result_feature_extractor.stderr}")
    print(f"Command stdout:\n{result_feature_extractor.stdout}")
    print(f"Exit status: {result_feature_extractor.returncode}")

    # 生成位姿文件
    posed_sparse_model_path = os.path.join(work_dir, "posed_sparse_model")
    generate_init_txt(args, posed_sparse_model_path)

    # set database camera parameters
    camTodatabase(database_path, os.path.join(posed_sparse_model_path, "cameras.txt"))

    # feature matching
    # pycolmap.match_exhaustive(database_path)

    colmap_exhaustive_matcher = [
        'COLMAP.bat', 'exhaustive_matcher',
        '--database_path', database_path,
        '--SiftMatching.max_distance', '0.8',
    ]
    print(f"Running command: {' '.join(colmap_exhaustive_matcher)}")
    result_exhaustive_matcher = subprocess.run(colmap_exhaustive_matcher, capture_output=True, text=True)
    print(f"Command stderr:\n{result_exhaustive_matcher.stderr}")
    print(f"Command stdout:\n{result_exhaustive_matcher.stdout}")
    print(f"Exit status: {result_exhaustive_matcher.returncode}")

    # 三角化
    triangulated_sparse_model_path = os.path.join(work_dir, "triangulated_sparse_model")
    os.makedirs(triangulated_sparse_model_path)
    colmap_triangulated = [
        'COLMAP.bat', 'point_triangulator',
        '--database_path', database_path,
        '--image_path', img_dir,
        '--input_path', posed_sparse_model_path,
        '--output_path', triangulated_sparse_model_path
    ]

    # 执行 COLMAP 命令
    print(f"Running command: {' '.join(colmap_triangulated)}")
    result_point_triangulator = subprocess.run(colmap_triangulated, capture_output=True, text=True)
    print(f"Command stderr:\n{result_point_triangulator.stderr}")
    print(f"Command stdout:\n{result_point_triangulator.stdout}")
    print(f"Exit status: {result_point_triangulator.returncode}")

    # dense model from sparse model

    dense_model_path = os.path.join(work_dir, "dense")
    os.makedirs(dense_model_path)
    # pycolmap.undistort_images(dense_model_path, triangulated_sparse_model_path, img_dir)
    # pycolmap.patch_match_stereo(dense_model_path) # requires compilation with CUDA

    colmap_image_undistorter = [
        'COLMAP.bat', 'image_undistorter',
        '--image_path', img_dir,
        '--input_path', triangulated_sparse_model_path,
        '--output_path', dense_model_path,
    ]
    # 执行 COLMAP 命令
    print(f"Running command: {' '.join(colmap_image_undistorter)}")
    result_image_undistorter = subprocess.run(colmap_image_undistorter, capture_output=True, text=True)
    print(f"Command stderr:\n{result_image_undistorter.stderr}")
    print(f"Command stdout:\n{result_image_undistorter.stdout}")
    print(f"Exit status: {result_image_undistorter.returncode}")

    if args.fuse_plt:
        colmap_patch_match = [
            'COLMAP.bat', 'patch_match_stereo',
            '--workspace_path', dense_model_path,
        ]
        print(f"Running command: {' '.join(colmap_patch_match)}")
        result_2 = subprocess.run(colmap_patch_match, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        print(f"Command stderr:\n{result_2.stderr}")
        print(f"Command stdout:\n{result_2.stdout}")
        print(f"Exit status: {result_2.returncode}")

        pycolmap.stereo_fusion(os.path.join(dense_model_path, "dense.ply"), dense_model_path)

    cameras_bin = read_cameras_binary(os.path.join(dense_model_path, "sparse", "cameras.bin"))
    images_bin = read_images_binary(os.path.join(dense_model_path, "sparse", "images.bin"))
    points3D_bin = read_points3D_binary(os.path.join(dense_model_path, "sparse", "points3D.bin"))

    sparse_txt_path = os.path.join(dense_model_path, "sparse_txt")
    os.makedirs(sparse_txt_path)
    write_cameras_text(cameras_bin, os.path.join(sparse_txt_path, "cameras.txt"))
    write_images_text(images_bin, os.path.join(sparse_txt_path, "images.txt"))
    write_points3D_text(points3D_bin, os.path.join(sparse_txt_path, "points3D.txt"))

    # 复制到 LEVIR_SPC文件夹
    if args.move_to_dataset:
        spc_dataset_path = args.dataset_path.replace("LEVIR_NVS", "LEVIR_SPC")
        tgt_path = os.path.join(spc_dataset_path, args.scene_id)
        os.makedirs(tgt_path, exist_ok=True)

        shutil.copy(os.path.join(sparse_txt_path, "images.txt"), os.path.join(tgt_path, "images.txt"))
        shutil.copy(os.path.join(sparse_txt_path, "points3D.txt"), os.path.join(tgt_path, "points3D.txt"))
        shutil.copy(os.path.join(dense_model_path, "dense.ply"), os.path.join(tgt_path, "dense.ply"))

    # 复制到其他数据集
    if args.move_to_other:
        tgt_path = os.path.join(args.other_dataset_path, args.scene_id)
        os.makedirs(tgt_path, exist_ok=True)

        src_sparse_path = os.path.join(dense_model_path, "sparse")
        dst_sparse_path = os.path.join(tgt_path, "sparse")

        shutil.copytree(src_sparse_path, dst_sparse_path)

    if args.use_all_images:
        gs_dataset = r"../data/LEVIR_GS"
        gs_scene_path = os.path.join(gs_dataset, args.scene_id)

        if os.path.exists(gs_scene_path):
            shutil.rmtree(gs_scene_path)

        os.makedirs(gs_scene_path, exist_ok=True)

        images_path = os.path.join(dense_model_path, "images")
        tgt_images_path = os.path.join(gs_scene_path, "images")
        shutil.copytree(images_path, tgt_images_path)

        sparse_path = os.path.join(dense_model_path, "sparse")
        tgt_sparse_path = os.path.join(gs_scene_path, "sparse")
        os.makedirs(tgt_sparse_path, exist_ok=True)
        shutil.copytree(sparse_path, os.path.join(tgt_sparse_path, "0"))

        # 替换 ply
        # 删除 points3D.bin
        os.remove(os.path.join(tgt_sparse_path, "0", "points3D.bin"))
        spc_scene_path = gs_scene_path.replace("LEVIR_GS", "LEVIR_SPC")
        shutil.copy(os.path.join(spc_scene_path, "points3D.txt"), os.path.join(tgt_sparse_path, "0", "points3D.txt"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset_path", help="path to dataset", type=str, required=True, default=r".\data\LEVIR_NVS"
    )

    parser.add_argument(
        "-s",
        "--scene_id",
        help="scene id",
        type=str,
        default="scene_000",
    )
    parser.add_argument(
        "-t",
        "--train_list",
        type=int, default=[1, 8, 15], nargs='*', help='training_idx'
    )
    parser.add_argument(
        "--fuse_plt",
        type=bool, default=True, help='whether to fuse point cloud using patch match stereo'
    )

    parser.add_argument(
        "--move_to_dataset",
        type=bool, default=False, help='move to dataset'
    )

    parser.add_argument(
        "--use_all_images",
        type=bool, default=False, help='whether to use all images'
    )

    parser.add_argument(
        "--move_to_other",
        type=bool, default=False, help='move to other dataset'
    )

    parser.add_argument(
        "--other_dataset_path",
        type=str, default=r".\data\LEVIR_NVS", help='other dataset path'
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.use_all_images:
        args.train_list = list(range(1, 21))

    main(args)

    # dense_model_path = r"runs/scene_000/new_4/dense"
    # cameras_bin = read_cameras_binary(os.path.join(dense_model_path, "sparse", "cameras.bin"))
    # images_bin = read_images_binary(os.path.join(dense_model_path, "sparse", "images.bin"))
    # points3D_bin = read_points3D_binary(os.path.join(dense_model_path, "sparse", "points3D.bin"))
    #
    # sparse_txt_path = os.path.join(dense_model_path, "sparse_txt")
    # os.makedirs(sparse_txt_path)
    # write_cameras_text(cameras_bin, os.path.join(sparse_txt_path, "cameras.txt"))
    # write_images_text(images_bin, os.path.join(sparse_txt_path, "images.txt"))
    # write_points3D_text(points3D_bin, os.path.join(sparse_txt_path, "points3D.txt"))

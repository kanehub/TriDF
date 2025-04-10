import numpy as np
import os
import shutil
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

# scene_id = "scene_000"
# train_list = [0, 7, 14]
neighbor_view_ids = ["000", "001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012",
                     "013", "014", "015", "016", "017", "018", "019", "020"]


def read_image(filename):
    image = Image.open(filename)

    return image


def read_camera_parameters(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsic = np.fromstring(",".join(lines[1:5]), dtype=np.float32, sep=",").reshape(4, 4)
    intrinsic = np.fromstring(",".join(lines[7:10]), dtype=np.float32, sep=",").reshape(3, 3)
    # depth_min, depth_max = [float(item) for item in lines[11].split(",")]
    return intrinsic, extrinsic


def get_cameras_txt(args, output_path):
    camera_filepath = os.path.join(output_path, "cameras.txt")
    if os.path.exists(camera_filepath):
        os.remove(camera_filepath)

    with open(camera_filepath, 'w') as f:
        # 共用一个相机模型
        f.write("# Camera list with one line of data per camera:\n\
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n\
# Number of cameras: 1\n"
                )
        neighbor_view_id = neighbor_view_ids[args.train_list[0]]
        intrinsic, _ = read_camera_parameters(
            os.path.join(args.dataset_path, args.scene_id, "Cams/{}.txt".format(neighbor_view_id))
        )

        image = read_image(
            os.path.join(args.dataset_path, args.scene_id, "Images/{}.png".format(neighbor_view_id))
        )
        W, H = image.size
        focal = intrinsic[0, 0]
        cx = int(intrinsic[0, -1])
        cy = int(intrinsic[1, -1])

        # 1 SIMPLE_RADIAL 512 512 609.81482262776569 256 256 -0.10856289114636609
        f.write(f"1 SIMPLE_PINHOLE {W} {H} {focal} {cx} {cy} \n")


def get_images_txt(args, output_path):
    images_filepath = os.path.join(output_path, "images.txt")
    if os.path.exists(images_filepath):
        os.remove(images_filepath)

    with open(images_filepath, 'w') as f:
        # 共用一个相机模型
        f.write("# Image list with two lines of data per image:\n\
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n\
#   POINTS2D[] as (X, Y, POINT3D_ID)\n\
# Number of images: 3\n"
                )
        for i, image_id in enumerate(args.train_list):
            neighbor_view_id = neighbor_view_ids[image_id]
            _, extrinsic = read_camera_parameters(
                os.path.join(args.dataset_path, args.scene_id, "Cams/{}.txt".format(neighbor_view_id))
            )

            # 转换坐标系
            # colmap x->right, y->down, z->forward
            # levir opencv 和 colmap 一样不用转

            T = extrinsic[:3, -1].squeeze()
            Rotation_matrix = extrinsic[:3, :3]
            r = Rot.from_matrix(Rotation_matrix)
            quat = r.as_quat()
            # (x,y,z,w) => (w,x,y,z)
            quat = np.concatenate((quat[-1:], quat[:-1]))

            # R = Rotation_matrix
            # q0 = 0.5 * math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
            # q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
            # q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
            # q3 = (R[1, 0] - R[0, 1]) / (4 * q0)

            # 1 0.695104 0.718385 -0.024566 0.012285 -0.046895 0.005253 -0.199664 1 image0001.png
            f.write(f"{i + 1} ")
            for quat_i in quat:
                f.write(f"{quat_i} ")
            for T_i in T:
                f.write(f"{T_i} ")
            f.write(f"1 {neighbor_view_id}.png\n\n")


def generate_init_txt(args, output_path):
    '''
    获得 cameras.txt, points3D.txt, images.txt, 以便从已知的相机姿势重建稀疏/密集模型
    :return:
    cameras.txt,
    points3D.txt 应为空，
    images.txt 中每隔一行为空
    '''

    os.makedirs(output_path, exist_ok=True)

    points_filename = "points3D.txt"

    with open(os.path.join(output_path, points_filename), 'w') as f:
        pass

    get_cameras_txt(args, output_path)
    get_images_txt(args, output_path)



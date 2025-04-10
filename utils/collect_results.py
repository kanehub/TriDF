import os
import json
import csv


def collect_json_result(root_path, data_type='test', include_names = None):
    """
    Collect all the json files in the root_path and return the result in a list
    :param root_path: the root path to search for json files
    :param type: train or test, the type of the data
    :param exclude_files: the files to exclude
    :return: result
    """

    json_keys = [
        f"{data_type}_mean_psnr",
        f"{data_type}_mean_ssim",
        f"{data_type}_mean_lpips",
    ]
    save_path = os.path.join(root_path, 'collected_results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if include_names is None or len(include_names) == 0:
        # all files
        include_names = ['-']
        result_file_name = f'all_{data_type}_results.csv'
    else:
        result_file_name = '_'.join(include_names) + f'_{data_type}_results.csv'



    result_list = []
    for root, dirs, files in os.walk(root_path):
        if check_include_files(root, include_names):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(root, file), 'r') as json_file:
                        data = json.load(json_file)
                        result_unit = []
                        result_unit.append(root)
                        for key in json_keys:
                            result_unit.append(data[key])
                        result_list.append(result_unit)

    csv_path = os.path.join(save_path, result_file_name)
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['name', 'psnr', 'ssim', 'lpips'])
        for result in result_list:
            csv_writer.writerow(result)




def check_include_files(file_name, include_names):
    for name in include_names:
        if name in file_name:
            return True
    return False

if __name__ == "__main__":

    root_path = r'../exp'
    collect_json_result(root_path, include_names=['loss'])
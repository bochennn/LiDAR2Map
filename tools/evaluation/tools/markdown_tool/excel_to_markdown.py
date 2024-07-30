import os

import pandas as pd
from mdutils.mdutils import MdUtils
from tabulate import tabulate


def parse_excel_to_df(excel_path, sheet_names=None):
    excel_reader = pd.ExcelFile(excel_path)
    sheet_names = sheet_names if sheet_names is not None else excel_reader.sheet_names
    return excel_reader.parse(sheet_names)


def excel_to_markdown(excel_path, out_path):
    name = os.path.splitext(os.path.basename(excel_path))[0]
    table_record = parse_excel_to_df(excel_path)

    md_file_path = os.path.join(out_path, "{}.md".format(name))
    md_file = MdUtils(md_file_path, title=name)
    md_file.new_header(level=1, title="KPI")
    for table_name, table in table_record.items():
        md_file.new_header(level=2, title=table_name)
        table_str = tabulate(table, headers=table.columns)
        md_file.insert_code(table_str, language="python")
    md_file.create_md_file()


def combine_excel_to_markdown(excel_path_list, out_path, test_name, target_sheet_names="AP@50"):
    md_file_path = os.path.join(out_path, "{}.md".format(test_name))
    md_file = MdUtils(md_file_path, title=test_name)
    for excel_path in sorted(excel_path_list):
        file_name = os.path.basename(excel_path)
        file_prefix = os.path.splitext(file_name)[0]
        table_record = parse_excel_to_df(excel_path, sheet_names=target_sheet_names)
        md_file.new_header(level=1, title=file_prefix)
        if isinstance(target_sheet_names, str):
            md_file.insert_code(target_sheet_names + "\n" + tabulate(table_record, headers=table_record.columns),
                                language="python")
        else:
            table_str = ""
            for table_name in target_sheet_names:
                table = table_record[table_name]
                table_str += (table_name + "\n" + tabulate(table, headers=table.columns))
            md_file.insert_code(table_str, language="python")
    md_file.create_md_file()


def create_md_report(excel_root_path, out_path):
    excel_path_list = []
    for root, dirs, files in os.walk(excel_root_path):
        for file_name in files:
            if file_name.endswith("xlsx"):
                excel_path = os.path.join(root, file_name)
                excel_path_list.append(excel_path)
                excel_to_markdown(excel_path, out_path)
    combine_excel_to_markdown(excel_path_list, out_path, "detector_20240124_dbsample")


if __name__ == "__main__":
    excel_root_path = "/mnt/data/lidar_detection/results/m2test/detector_2024013114/evaluate_ret"
    out_path = "/mnt/data/lidar_detection/results/m2test/detector_2024013114/evaluate_ret_md"
    create_md_report(excel_root_path, out_path)

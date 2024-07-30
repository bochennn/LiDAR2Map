import os
import pprint
import json
from collections import OrderedDict

import requests
import pandas as pd
from tabulate import tabulate

from log_mgr import logger
from utils.multiprocess import arg_split_by_bin_size


folder_id = "Lg73fpWItlxyH5dUYSzcmbZenIc"


def parse_excel_to_df(excel_path, sheet_names=None):
    excel_reader = pd.ExcelFile(excel_path)
    sheet_names = sheet_names if sheet_names is not None else excel_reader.sheet_names
    table_record = excel_reader.parse(sheet_names)
    return {table_name: tabulate(table, headers=table.columns, showindex=False)
            for table_name, table in table_record.items()}


def get_tenant_access_token(app_id="cli_a522521d9ff9d00c", app_secret="v4Jy3lWKy4Wo3AYGtj9EiKv5Zn0TOcTt"):
    url = "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal"
    payload = json.dumps({
        "app_id": app_id,
        "app_secret": app_secret
    })

    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    ret = json.loads(response.text)
    if ret["code"] == 0:
        logger.info("get tenant_access_token succeed")
        return ret["tenant_access_token"]
    else:
        logger.error(ret)


def create_doc(folder_id, doc_title, tenant_access_token):
    url = "https://open.feishu.cn/open-apis/docx/v1/documents"
    payload = json.dumps({
        "folder_token": folder_id,
        "title": doc_title
    })

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer {}'.format(tenant_access_token)
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    ret = json.loads(response.text)
    if ret["code"] == 0:
        logger.info("create doc succeed")
        return ret["data"]["document"]["document_id"]
    else:
        logger.error(ret)


def create_block(doc_id, block_id, request_body, tenant_access_token, index=-1):
    if len(request_body) > 45:
        ret_list = []
        for sub_request_body in arg_split_by_bin_size(request_body, 40):
            ret_list.extend(create_block(doc_id, block_id, sub_request_body, tenant_access_token, index=-1))
        return ret_list
    url = "https://open.feishu.cn/open-apis/docx/v1/documents/{}/blocks/{}/children?document_revision_id=-1".format(
        doc_id, block_id
    )

    payload = json.dumps({
        "children": request_body,
        "index": index
    })

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer {}'.format(tenant_access_token)
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    ret = json.loads(response.text)
    if ret["code"] == 0:
        logger.info("create block succeed")
        return [ret]
    else:
        logger.error("length of blocks: {}".format(len(request_body)))
        logger.error("{}".format(pprint.pformat(request_body)))
        logger.error(ret)


def new_header_request_body(level, content, link=None, folded=False, bold=True, text_color=None):
    assert 1 <= level <= 9
    block_id = level + 2
    text_element_style = dict()
    if link is not None:
        text_element_style["link"] = {"url": link}
    if bold:
        text_element_style["bold"] = bold
    if text_color is not None:
        text_element_style["text_color"] = text_color
    request_body = {"block_type": block_id,
                    "heading{}".format(level): {
                        "style": {"folded": folded},
                        "elements": [
                            {"text_run": {
                                "content": content,
                                "text_element_style": text_element_style
                            }}]
                    }}
    return request_body


def new_text_request_body(content, link=None, folded=False, bold=True):
    request_body = {"block_type": 2,
                    "text": {
                        "style": {"folded": folded},
                        "elements": [
                            {"text_run": {
                                "content": content,
                                "text_element_style": {"bold": bold}
                            }} if not link else
                            {"text_run": {
                                "content": content,
                                "text_element_style": {"link": {"url": link},
                                                       "bold": bold}
                            }}]
                    }}
    return request_body


def new_python_code_request_body(content):
    request_body = {"block_type": 14,
                    "code": {
                        "style": {"language": 49},
                        "elements": [
                            {"text_run": {
                                "content": content
                            }}
                        ]
                    }}
    return request_body


def new_json_code_request_body(content):
    request_body = {"block_type": 14,
                    "code": {
                        "style": {"language": 28},
                        "elements": [
                            {"text_run": {
                                "content": content
                            }}
                        ]
                    }}
    return request_body


def create_url_from_doc_id(doc_id, block_id=None):
    doc_url_sample = "https://wx6lpdt35j.feishu.cn/docx/{}"
    block_url_sample = "#{}"
    url = doc_url_sample.format(doc_id)
    if block_id is not None:
        url += block_url_sample.format(block_id)
    return url


class LarkClient:
    def __init__(self, folder_id=folder_id):
        self.tenant_access_token = get_tenant_access_token()
        self.folder_id = folder_id
        self.doc_id = None

    def init_new_doc(self, doc_title):
        doc_id = create_doc(self.folder_id, doc_title, self.tenant_access_token)
        self.doc_id = doc_id
        return doc_id

    def create_epoch_kpi(self, title, tables, need_table_name=True, heading_link=None, heading_color=None):
        request_body_list = [new_header_request_body(level=2,
                                                     content=title,
                                                     link=heading_link,
                                                     text_color=heading_color)]
        for table_name, table in tables.items():
            if need_table_name:
                # request_body_list.append(new_header_request_body(level=3, content=table_name))
                request_body_list.append(new_text_request_body(content=table_name))
            request_body_list.append(new_python_code_request_body(table))
        return request_body_list

    def create_config(self, config_str):
        request_body_list = [new_header_request_body(level=1, content="evaluation_config", folded=True),
                             new_json_code_request_body(config_str)]
        return request_body_list

    def create_combine_kpi(self, all_epoch_tables, doc_name):
        self.init_new_doc(doc_name)
        detail_request_body_list = [new_header_request_body(level=1, content="Detail", folded=True)]
        brief_request_body_list = [new_header_request_body(level=1, content="AP")]
        config = None
        for epoch_name, (tables, config) in all_epoch_tables.items():
            detail_request_body_list.extend(self.create_epoch_kpi(epoch_name, tables, heading_color=5))
        detailed_block_ret = create_block(self.doc_id, self.doc_id, detail_request_body_list, self.tenant_access_token)
        detailed_block_record = self.extract_detail_heading_block_id(detailed_block_ret)
        for epoch_name, (tables, config) in all_epoch_tables.items():
            epoch_block_id = detailed_block_record[epoch_name]
            epoch_url = create_url_from_doc_id(self.doc_id, epoch_block_id)
            brief_request_body_list.extend(self.create_epoch_kpi(epoch_name,
                                                                 {"AP@50": tables["AP@50"]},
                                                                 need_table_name=False,
                                                                 heading_link=epoch_url))
        brief_block_ret = create_block(self.doc_id,
                                       self.doc_id,
                                       brief_request_body_list,
                                       self.tenant_access_token,
                                       index=0)

        if config is not None:
            config_str = pprint.pformat(config)
            create_block(self.doc_id, self.doc_id, self.create_config(config_str), self.tenant_access_token)

    @staticmethod
    def extract_detail_heading_block_id(detailed_block_ret):
        block_id_record = OrderedDict()
        for one_batch in detailed_block_ret:
            for one_block in one_batch["data"]["children"]:
                if "heading2" in one_block:
                    text = one_block["heading2"]["elements"][0]["text_run"]["content"]
                    if text.startswith("epoch_"):
                        block_id_record[text] = one_block["block_id"]
        return block_id_record

    @staticmethod
    def get_all_epoch_tables(root_data_path):
        epoch_tables_list = dict()
        for name in sorted(os.listdir(root_data_path), key=lambda x: x.split("_")[1]):
            epoch_data_path = os.path.join(root_data_path, name)
            if os.path.isdir(epoch_data_path) and "epoch_" in name:
                epoch_name_list = os.listdir(epoch_data_path)
                kpi_data_name = [name for name in epoch_name_list if name.endswith("xlsx")][0]
                kpi_data_path = os.path.join(epoch_data_path, kpi_data_name)
                epoch_tables = parse_excel_to_df(kpi_data_path)

                eval_config_name = [name for name in epoch_name_list if name.endswith(".json")][0]
                eval_config_path = os.path.join(epoch_data_path, eval_config_name)
                with open(eval_config_path, 'r') as f:
                    eval_config = json.load(f)
                epoch_tables_list[name] = (epoch_tables, eval_config)
        return epoch_tables_list

    def start(self, root_data_path, doc_name):
        all_epoch_tables = self.get_all_epoch_tables(root_data_path)
        self.create_combine_kpi(all_epoch_tables, doc_name)


if __name__ == "__main__":
    client = LarkClient(folder_id)
    root_data_path = "/mnt/data/lidar_detection/results/m2test/detector_2024020219/evaluate_ret"
    client.start(root_data_path, "detector_2024020219")


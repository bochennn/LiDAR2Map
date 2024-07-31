from tabulate import tabulate

# from log_mgr import logger


def print_result(result_table, table_name):
    seperator = "=" * 60
    # logger.info(seperator)
    # logger.info(table_name)
    # result_str = tabulate(result_table, headers=result_table.columns)
    # logger.info(result_str)

    # print(seperator)
    # print(table_name)
    result_str = tabulate(result_table, headers=result_table.columns)
    # print(result_str)
    return seperator + "\n" + table_name + "\n" + result_str

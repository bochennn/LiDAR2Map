import os
from collections import OrderedDict
from multiprocessing.pool import Pool

from cyber_record.record import Record
from tqdm import tqdm

from log_mgr import logger
from objects.obstacle.objs.obstacle_clip_gt import ObstacleClipGt
from utils.pointcloud_ops import PointCompensator
from tools.m2_data_convert.update_json import hesai_to_m2


raw_anno_name = "annotation"
visual_anno_name = "annotations_visual"
m2_visual_anno_name = "m2_annotations_visual"


def recreate_dir(target_dir):
    if os.path.exists(target_dir):
        os.system("rm -r {}".format(target_dir))
        os.makedirs(target_dir)


class DataSetMaker:
    def __init__(self, dataset_root_path):
        self.dataset_root_path = dataset_root_path
        self.clip_path_record = self.get_clip_path_record()

    def get_clip_path_record(self):
        clip_path_record = OrderedDict()
        for root, dirs, files in os.walk(self.dataset_root_path):
            for dir_name in dirs:
                if dir_name.startswith("clip_"):
                    clip_path = os.path.join(root, dir_name)
                    clip_path_record[dir_name] = clip_path
        return clip_path_record

    def relocate_clips(self):
        processed_clips = []
        clip_root_path = os.path.join(self.dataset_root_path, "clips")
        for clip_name, clip_path in self.clip_path_record.items():
            dst_clip_path = os.path.join(clip_root_path, clip_name)
            if not os.path.exists(dst_clip_path):
                os.makedirs(clip_root_path, exist_ok=True)
                logger.info("move {} to {}".format(clip_path, dst_clip_path))
                command = "mv {} {}".format(clip_path, dst_clip_path)
                os.system(command)
                self.clip_path_record.update({clip_name: dst_clip_path})
                processed_clips.append(clip_name)
        logger.info("{}/{} clips relocated, all those clips now located under {}".format(len(processed_clips),
                                                                                         len(self.clip_path_record),
                                                                                         clip_root_path))

    @staticmethod
    def search_raw_clip_annotation(clip_path):
        for root, dirs, files in os.walk(clip_path):
            for file in files:
                if file.startswith("clip_") and file.endswith(".json"):
                    return os.path.join(root, file)
        return None

    @staticmethod
    def search_annotation(clip_path, anno_name):
        visual_annotation_list = []
        for root, dirs, files in os.walk(os.path.join(clip_path, anno_name)):
            for file in files:
                if file.endswith(".json"):
                    visual_annotation_list.append(os.path.join(root, file))
        return visual_annotation_list

    @staticmethod
    def search_split_annotation(clip_path):
        for root, dirs, files in os.walk(clip_path):
            for file in files:
                if file.startswith("sample_") and file.endswith(".json"):
                    return root
        return None

    @staticmethod
    def search_bag(clip_path):
        for root, dirs, files in os.walk(clip_path):
            for file in files:
                if ".record." in file and os.path.basename(root) == "bag":
                    return os.path.join(root, file)
        return None

    @staticmethod
    def search_sensor_data(clip_path, sensor_name):
        sensor_data_path_list = []
        for root, dirs, files in os.walk(clip_path):
            for file in files:
                if file.startswith(sensor_name) and (file.endswith(".jpg") or file.endswith(".pcd")):
                    sensor_data_path_list.append(os.path.join(root, file))
        return sensor_data_path_list

    def annotation_convert(self):
        processed_clips = []
        for clip_name, clip_path in self.clip_path_record.items():
            clip_annotation_path = self.search_raw_clip_annotation(clip_path)
            if clip_annotation_path is not None:
                logger.debug("convert annotation of {} into visualization format".format(clip_name))
                clip_visual_anno_path = os.path.join(clip_path, visual_anno_name)
                if os.path.exists(clip_visual_anno_path):
                    os.system("rm -r {}".format(clip_visual_anno_path))
                os.makedirs(clip_visual_anno_path, exist_ok=True)
                gt_obj = ObstacleClipGt(clip_annotation_path)
                gt_obj.to_visual_json(clip_visual_anno_path)
                processed_clips.append(clip_name)
        logger.info("{}/{} clips annotation transformed into visualization format".format(len(processed_clips),
                                                                                          len(self.clip_path_record)))

    def m2_annotation_convert(self):
        processed_clips = []
        for clip_name, clip_path in self.clip_path_record.items():
            clip_visual_annotations_path = os.path.join(clip_path, visual_anno_name)
            if os.path.exists(clip_visual_annotations_path):
                processed_clips.append(clip_name)
                clip_m2_anno_path = os.path.join(clip_path, m2_visual_anno_name)
                os.makedirs(clip_m2_anno_path, exist_ok=True)
                hesai_to_m2(clip_visual_annotations_path, clip_m2_anno_path)
        logger.info("processed clip num: {}".format(len(processed_clips)))

    def symlink_bags(self):
        dataset_bag_path = os.path.join(self.dataset_root_path, "bags")
        for clip_name, clip_path in self.clip_path_record.items():
            clip_record_path = self.search_bag(clip_path)
            if clip_record_path is not None:
                dst_record_path = os.path.join(dataset_bag_path, os.path.basename(clip_record_path))
                if not os.path.exists(dst_record_path):
                    os.makedirs(dataset_bag_path, exist_ok=True)
                    logger.debug("create symlink {} -> {}".format(clip_record_path, dst_record_path))
                    os.symlink(clip_record_path, dst_record_path)

    def symlink_sensor_data(self, sensor_name):
        annotated_sensor_name = "{}_annotated".format(sensor_name)

        dataset_sensor_root_path = os.path.join(self.dataset_root_path, sensor_name)
        dataset_annotated_sensor_root_path = os.path.join(self.dataset_root_path, annotated_sensor_name)
        recreate_dir(dataset_sensor_root_path)
        recreate_dir(dataset_annotated_sensor_root_path)
        processed_clips = []
        processed_samples = []
        processed_annotated_samples = []
        for clip_name, clip_path in self.clip_path_record.items():
            clip_sensor_path_list = self.search_sensor_data(clip_path, sensor_name)
            clip_annotation_list = self.search_annotation(clip_path, visual_anno_name)
            clip_annotation_record = {os.path.splitext(os.path.basename(file_path))[0]: file_path
                                      for file_path in clip_annotation_list}
            clip_sensor_data_path = os.path.join(clip_path, sensor_name)
            clip_annotated_sensor_data_path = os.path.join(clip_path, annotated_sensor_name)
            recreate_dir(clip_sensor_data_path)
            recreate_dir(clip_annotated_sensor_data_path)
            if len(clip_sensor_path_list) != 0:
                logger.info("sensor name: {}, {} files found in {}".format(sensor_name,
                                                                           len(clip_sensor_path_list),
                                                                           clip_name))
                processed_clips.append(clip_name)
                for raw_file_path in clip_sensor_path_list:
                    sample_name = os.path.basename(os.path.dirname(raw_file_path))
                    _, postfix = os.path.splitext(os.path.basename(raw_file_path))
                    sample_ts = sample_name.split("_")[1]
                    sample_ts = sample_ts[:10] + "." + sample_ts[10:]
                    file_name = os.path.join("{}{}".format(sample_ts, postfix))

                    clip_sensor_file_path = os.path.join(clip_sensor_data_path, file_name)
                    if not os.path.exists(clip_sensor_file_path):
                        os.symlink(raw_file_path, clip_sensor_file_path)
                    clip_annotated_sensor_file_path = os.path.join(clip_annotated_sensor_data_path, file_name)
                    if not os.path.exists(clip_annotated_sensor_file_path) and sample_ts in clip_annotation_record:
                        os.symlink(raw_file_path, clip_annotated_sensor_file_path)

                    dataset_sensor_file_path = os.path.join(dataset_sensor_root_path, file_name)
                    if not os.path.exists(dataset_sensor_file_path):
                        os.symlink(raw_file_path, dataset_sensor_file_path)
                        processed_samples.append(dataset_sensor_file_path)
                    dataset_annotated_sensor_file_path = os.path.join(dataset_annotated_sensor_root_path, file_name)
                    if sample_ts in clip_annotation_record and not os.path.exists(dataset_annotated_sensor_file_path):
                        os.symlink(raw_file_path, dataset_annotated_sensor_file_path)
                        processed_annotated_samples.append(dataset_annotated_sensor_file_path)
        logger.info("sensor name: {}".format(sensor_name))
        logger.info("{}/{} clips processed, "
                    "{} samples processed, "
                    "{} annotated samples processed".format(len(processed_clips),
                                                            len(self.clip_path_record),
                                                            len(processed_samples),
                                                            len(processed_annotated_samples)))

    def symlink_fusion_lidar(self):
        self.symlink_sensor_data("combine")

    def symlink_main_lidar(self):
        self.symlink_sensor_data("lidar0")

    def symlink_m2_lidar(self):
        self.symlink_sensor_data("lidar5")

    def symlink_visual_annotation(self):
        dataset_anno_path = os.path.join(self.dataset_root_path, visual_anno_name)
        os.makedirs(dataset_anno_path, exist_ok=True)
        processed_clips = []
        processed_gt_samples = []
        for clip_name, clip_path in self.clip_path_record.items():
            clip_visual_anno_list = self.search_annotation(clip_path, visual_anno_name)
            if len(clip_visual_anno_list) != 0:
                logger.info("{} visual annotation files found in {}".format(len(clip_visual_anno_list), clip_name))
                processed_clips.append(clip_name)
                for clip_visual_anno_file in clip_visual_anno_list:
                    dataset_visual_anno_path = os.path.join(dataset_anno_path,
                                                            os.path.basename(clip_visual_anno_file))
                    if not os.path.exists(dataset_visual_anno_path):
                        os.symlink(clip_visual_anno_file, dataset_visual_anno_path)
                        processed_gt_samples.append(clip_visual_anno_file)
        logger.info(
            "{}/{} clips processed, "
            "{} samples processed, for visual annotation symlink create under {}".format(len(processed_clips),
                                                                                         len(self.clip_path_record),
                                                                                         len(processed_gt_samples),
                                                                                         dataset_anno_path))

    def symlink_m2_annotation(self):
        dataset_m2_anno_path = os.path.join(self.dataset_root_path, m2_visual_anno_name)
        os.makedirs(dataset_m2_anno_path, exist_ok=True)
        processed_clips = []
        processed_gt_samples = []
        for clip_name, clip_path in self.clip_path_record.items():
            clip_m2_anno_list = self.search_annotation(clip_path, m2_visual_anno_name)
            if len(clip_m2_anno_list) != 0:
                logger.info("{} m2 annotation files found in {}".format(len(clip_m2_anno_list), clip_name))
                processed_clips.append(clip_name)
                for clip_m2_anno_file in clip_m2_anno_list:
                    dataset_m2_anno_file_path = os.path.join(dataset_m2_anno_path,
                                                             os.path.basename(clip_m2_anno_file))
                    if not os.path.exists(dataset_m2_anno_file_path):
                        os.symlink(clip_m2_anno_file, dataset_m2_anno_file_path)
                        processed_gt_samples.append(dataset_m2_anno_file_path)
        logger.info(
            "{}/{} clips processed, "
            "{} samples processed, for visual annotation symlink create under {}".format(len(processed_clips),
                                                                                         len(self.clip_path_record),
                                                                                         len(processed_gt_samples),
                                                                                         dataset_m2_anno_path))

    def bag_compensate(self):
        task_args = [(clip_name, clip_path) for clip_name, clip_path in self.clip_path_record.items()]
        pool = Pool(processes=4)
        pool.starmap(self.create_bag_with_compensate, task_args)

    @staticmethod
    def topic_exist(bag_path, topic):
        with Record(bag_path, 'r') as record:
            topic_list = [ele.name for ele in record.get_channel_cache()]
        return topic in topic_list

    def create_bag_with_compensate(self, clip_name, clip_path, target_topic="/apollo/sensor/hesai64/PointCloud2", force=False):
        logger.info("create bag with compensate pointcloud message for {}".format(clip_name))
        clip_compensate_path = os.path.join(clip_path, "bag_compensate")
        if force and os.path.exists(clip_compensate_path):
            os.system("rm -r {}".format(clip_compensate_path))
        clip_bag_path = self.search_bag(clip_path)
        os.makedirs(clip_compensate_path, exist_ok=True)
        compensate_bag_path = os.path.join(clip_compensate_path, os.path.basename(clip_bag_path))
        dst_topic = "/apollo/sensor/hesaiqt128/compensator/PointCloud2"
        if os.path.exists(compensate_bag_path) and self.topic_exist(compensate_bag_path, dst_topic):
            logger.info("bag with target topic already exist, skip")
            return
        compensator = PointCompensator(clip_path)
        logger.info("create bag with compensate lidar message at [{}] for [{}]".format(compensate_bag_path,
                                                                                       clip_bag_path))
        with Record(clip_bag_path, 'r') as record:
            with Record(compensate_bag_path, 'w') as out_record:
                for topic, message, t in record.read_messages():
                    if topic == target_topic:
                        message = compensator.compensate(message)
                        topic = dst_topic
                    out_record.write(topic, message, t)
        logger.info("creation of bag with compensate pointcloud message for {} finished".format(clip_name))

    def create(self):
        # self.relocate_clips()
        # self.annotation_convert()
        # self.m2_annotation_convert()
        # self.symlink_bags()
        # self.symlink_visual_annotation()
        # self.symlink_m2_annotation()
        # self.symlink_main_lidar()
        # self.symlink_m2_lidar()
        # self.symlink_fusion_lidar()
        # for camera_id in range(11):
        #     camera_name = "camera{}".format(camera_id)
        #     self.symlink_sensor_data(camera_name)
        # self.symlink_fusion_lidar_without_filter()
        self.bag_compensate()


if __name__ == "__main__":
    dataset_path = "/mnt/data/lidar_detection/test_datasets/20231028"
    dataset_maker = DataSetMaker(dataset_path)
    dataset_maker.create()

